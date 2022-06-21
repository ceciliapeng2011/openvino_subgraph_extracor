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
    std::vector<std::string> target_input = {"dien/rnn_2/gru2/concat_2", "dien/rnn_2/gru2/sub_2"};
    std::vector<std::string> target_output = {"dien/rnn_2/gru2/add_1"};

    std::vector<std::shared_ptr<opset8::Parameter> > subgraph_parameters = {};
    std::vector<std::shared_ptr<opset8::Result> > subgraph_results = {};

    for(auto& input_name : target_input) {
        auto input_op = name2op.at(input_name);
        if (auto node = ov::as_type_ptr<opset8::Parameter>(input_op)) {
            std::cout << "keep original parameter " << input_name << std::endl;
            subgraph_parameters.push_back(node);
            continue;
        }

        for(size_t i = 0; i < input_op->get_input_size(); i++) {
            auto parent = input_op->get_input_node_shared_ptr(i);
            if(ov::as_type_ptr<ov::opset8::Constant>(parent)) {
                continue;
            }
            // each non-constant inputs will be replaced by a Parameter node.
            auto new_param = std::make_shared<opset8::Parameter>(input_op->get_input_element_type(i),
                                                                input_op->get_input_partial_shape(i));
            const auto new_name = input_name+"/"+std::to_string(i);                                                                
            new_param->set_friendly_name(new_name);
            new_param->get_output_tensor(0).add_names({new_name});

            subgraph_parameters.push_back(new_param);

            for (auto child : parent->outputs()) {
                for (auto &input : child.get_target_inputs()) {
                    input.replace_source_output(new_param);
                }
            }
            //input_op->input_value(i).replace(new_param->output(i));            
        }
    }
    model->add_parameters(subgraph_parameters);
    model->validate_nodes_and_infer_types();
    std::cout << __LINE__ << ": model nodes " << model->get_ops().size() << ", parameter " << model->get_parameters().size() << ", results " << model->get_results().size() << std::endl;

    for(auto& output_name : target_output) {
        auto output_op = name2op.at(output_name);
        if (output_op->get_output_size() !=1) {
            throw std::runtime_error("output must has 1 child");
        }
        auto node_copy = output_op->clone_with_new_inputs(output_op->input_values());
        auto new_result = std::make_shared<opset8::Result>(node_copy);
        ov::replace_node(output_op, new_result);
        subgraph_results.push_back(new_result);
    }
    model->add_results(subgraph_results);

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
