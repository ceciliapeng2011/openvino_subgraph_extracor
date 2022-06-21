// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>

#include <ie_core.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"

using namespace ov;


void run_model(std::shared_ptr<ov::Model> model) {
    ov::Core core;
    
    // for Dien
    for (auto& parameter : model->get_parameters()) {
        parameter->set_layout("NC");
    }
    ov::set_batch(model, 16); // Dien

    ov::AnyMap config = {
        ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
        ov::hint::inference_precision("FP32"),
        ov::inference_num_threads(4),
        ov::num_streams(1),
        ov::enable_profiling(true)
    };    

    auto exeNetwork = core.compile_model(model, "CPU", config);
    auto infer_request = exeNetwork.create_infer_request();

    std::cout << "Feeding model with random inputs." << std::endl;
    // auto preproc = ov::preprocess::PrePostProcessor(model);
    // for (auto& item : exeNetwork.inputs()) {
    //     auto& in = preproc.input(item.get_any_name());
    //     // Explicitly set inputs layout.
    //     in.model().set_layout("NC");
    // }

    for (auto& input : exeNetwork.inputs()) {
        const auto& inputname = input.get_any_name();     
        auto& ps = input.get_partial_shape();
        std::cout << inputname << ps << std::endl;   

        ov::Tensor input_tensor = infer_request.get_tensor(inputname);
        // if (ps.is_dynamic()) {
        //     for(auto dim = ps.begin(); dim !=ps.end(); dim++) {
        //         if (dim->is_dynamic()) {                            
        //             *dim = 16;
        //         }
        //     }            
        //     input_tensor.set_shape(ps.to_shape());
        // }

        std::vector<float > fake_input(input_tensor.get_size(), 0);
        std::memcpy(input_tensor.data<float>(), fake_input.data(), fake_input.size());
    }

    std::cout << "Running saved model." << std::endl;
    infer_request.infer();
    infer_request.infer();

    for (auto i = 0; i < exeNetwork.outputs().size(); i++) {
        auto output_tensor = infer_request.get_output_tensor(i);
        std::cout << output_tensor.get_shape() << std::endl;
    }
}

int main(int args, char *argv[]) {
    // if (args < 4)
    //     return -1;

    ov::Core core;
    auto model = core.read_model(argv[1]);

    auto ordered_ops = model->get_ordered_ops();
    std::cout << __LINE__ << ": model nodes " << model->get_ops().size() << ", parameter " << model->get_parameters().size() << ", results " << model->get_results().size() << std::endl;

    std::unordered_map<std::string, std::shared_ptr<Node>> name2op = {};

    //collect op mapps
    for(auto& op : ordered_ops) {
        name2op.emplace(op->get_friendly_name(), op);
    }

    // These target inputs and outputs are tensors.
    std::vector<std::string> target_inputs = {
            "dien/rnn_2/gru2/add:0", 
            "dien/rnn_1/gru1/GRUBlockCell_1/GRUBlockCell:0", "dien/rnn_2/gru2/strided_slice_2/Squeeze_shrink:0",
            "dien/rnn_1/gru1/GRUBlockCell_2/GRUBlockCell:0", "dien/rnn_2/gru2/strided_slice_3/Squeeze_shrink:0"
        };
    std::vector<std::string> target_outputs = {"dien/rnn_2/gru2/add_2:0"};

    std::vector<std::shared_ptr<opset8::Parameter> > subgraph_parameters = {};
    std::vector<std::shared_ptr<opset8::Result> > subgraph_results = {};

    auto get_op_name_port = [](std::string& tensor_name) -> std::tuple<std::string, int64_t> {
        const auto pos = tensor_name.find_last_of(":");
        if (pos == std::string::npos) {
            std::cout << "warning, not a tensor name, skip." << std::endl;
            return std::make_tuple(std::string(), -1);
        }
        const auto op_name = tensor_name.substr(0, pos-0);
        const auto port_name = tensor_name.substr(pos+1);
        const auto port_id = std::stoi(port_name);
        std::cout << "target : " << tensor_name << ", op " << op_name << ", port " << port_id << std::endl;
        return std::make_tuple(op_name, port_id);
    };

    for (auto& input_tensor : target_inputs) {
        std::string op_name;
        int64_t port_id;
        std::tie(op_name, port_id) = get_op_name_port(input_tensor);
        if (op_name.empty() || port_id == -1) continue;

        auto input_op = name2op.at(op_name);
        if (input_op->get_output_tensor(port_id).get_names().size() < 1) {
            input_op->get_output_tensor(port_id).add_names({input_tensor});
        }

        if (const auto node = ov::as_type_ptr<opset8::Parameter>(input_op)) {
            std::cout << "keep original parameter " << op_name << std::endl;
            subgraph_parameters.push_back(node);
        } else {
            auto output = input_op->output(port_id);

            // each non-constant inputs will be replaced by a Parameter node.
            auto new_param = std::make_shared<opset8::Parameter>(output.get_element_type(),
                                                                output.get_partial_shape());
            const auto new_name = input_tensor;
            new_param->set_friendly_name(new_name);
            new_param->get_output_tensor(0).add_names({new_name});

            for (auto &input : output.get_target_inputs()) {
                input.replace_source_output(new_param);
            }

            subgraph_parameters.push_back(new_param);            
        }
    }

    model->add_parameters(subgraph_parameters);
    model->validate_nodes_and_infer_types();
    std::cout << __LINE__ << ": model nodes " << model->get_ops().size() << ", parameter " << model->get_parameters().size() << ", results " << model->get_results().size() << std::endl;

    for(auto& output_tensor : target_outputs) {
        std::string op_name;
        int64_t port_id;
        std::tie(op_name, port_id) = get_op_name_port(output_tensor);
        if (op_name.empty() || port_id == -1) continue;

        auto output_op = name2op.at(op_name);
        std::cout << "tensor port "<<  port_id << ": " << output_op->get_output_tensor(port_id).get_any_name() << std::endl;

        if (const auto node = ov::as_type_ptr<opset8::Result>(output_op)) {
            std::cout << "keep original Result " << op_name << std::endl;
            subgraph_results.push_back(node);
        } else {
            std::shared_ptr<opset8::Result> new_res;
            if (output_op->get_output_size() == 1) {
                auto node_copy = output_op->clone_with_new_inputs(output_op->input_values());
                new_res = std::make_shared<opset8::Result>(node_copy);
                ov::replace_node(output_op, new_res);
            } else {
                auto output = output_op->output(port_id);

                // each non-constant inputs will be replaced by a Parameter node.
                new_res = std::make_shared<opset8::Result>(output);
                const auto new_name = output_tensor;
                new_res->set_friendly_name(new_name);
                new_res->get_output_tensor(0).add_names({new_name});

                for (auto &input : output.get_target_inputs()) {
                    input.replace_source_output(new_res);
                }                
            }

            subgraph_results.push_back(new_res);
        }
    }
    std::cout << __LINE__ << std::endl;
    model->add_results(subgraph_results);
    //model->validate_nodes_and_infer_types();
    std::cout << __LINE__ << ": model nodes " << model->get_ops().size() << ", parameter " << model->get_parameters().size() << ", results " << model->get_results().size() << std::endl;

    /* remove the original parameters... otherwise exeption that they are not registered. */
    const auto parameters = model->get_parameters();
    for (auto node : parameters) {
        if (std::find(subgraph_parameters.begin(), subgraph_parameters.end(), node) == subgraph_parameters.end()) {
                // auto ps = node->get_output_partial_shape(0);
                // if (ps.is_dynamic()) {
                //     std::cout << ps << std::endl;
                //     for(auto dim = ps.begin(); dim !=ps.end(); dim++) {
                //         if (dim->is_dynamic()) {                            
                //             *dim = 100;
                //         }
                //     }
                // }
                // auto init_const = opset8::Constant::create(node->get_output_element_type(0), ps.to_shape(), {0});
                // //auto read = std::make_shared<opset3::ReadValue>(init_const, "v0");    
                // //model->input(1).replace_source_output(read->output(0));
                // model->replace_node(node, init_const);

                model->remove_parameter(node);
                std::cout << "remove parameter " << node->get_friendly_name() << std::endl;
        }  
    }
    const auto results = model->get_results();
    for (auto node : results) {
        if (std::find(subgraph_results.begin(), subgraph_results.end(), node) == subgraph_results.end()) {
                model->remove_result(node);
                std::cout << "remove result " << node->get_friendly_name() << std::endl;
        }  
    }    
    std::cout << __LINE__ << ": model nodes " << model->get_ops().size() << ", parameter " << model->get_parameters().size() << ", results " << model->get_results().size() << std::endl;

    model->validate_nodes_and_infer_types();
    

    //auto subgraph = std::make_shared<ov::Model>(subgraph_results, subgraph_parameters);

    ov::pass::Serialize serializer("simple_model.xml", "simple_model.bin");
    serializer.run_on_model(model);
    std::cout << "Saved subgraph IR." << std::endl;
    
    run_model(model);
}
