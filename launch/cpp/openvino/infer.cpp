// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        std::cout << ov::get_openvino_version() << std::endl;
        
        
        // -------- Parsing and validation of input arguments --------
        if (argc != 2) {
            std::cout << "Usage : " << argv[0] << " <path_to_model>"
                       << std::endl;
            return EXIT_FAILURE;
        }

        const std::string args = argv[0];
        const std::string model_path = argv[1];
        
        std::vector<int> text_inputs = {178, 56, 178, 156, 178, 57, 178, 135, 178, 3, 178, 16, 178, 61, 178, 157, 178, 57
                                        , 178, 135, 178, 16, 178, 44, 178, 157, 178, 51, 178, 158, 178, 102, 178, 112, 178, 16
                                        , 178, 156, 178, 47, 178, 102, 178, 44, 178, 83, 178, 54, 178, 16, 178, 62, 178, 63
                                        , 178, 158, 178, 3, 178, 16, 178, 54, 178, 156, 178, 43, 178, 102, 178, 53, 178, 3
                                        , 178, 16, 178, 92, 178, 86, 178, 62, 178, 16, 178, 102, 178, 56, 178, 16, 178, 46
                                        , 178, 156, 178, 102, 178, 48, 178, 123, 178, 83, 178, 56, 178, 62, 178, 16, 178, 58
                                        , 178, 83, 178, 68, 178, 156, 178, 102, 178, 131, 178, 83, 178, 56, 178, 68, 178, 16
                                        , 178, 48, 178, 76, 178, 158, 178, 123, 178, 16, 178, 52, 178, 63, 178, 158, 178, 3
                                        , 178, 16, 178, 54, 178, 156, 178, 43, 178, 102, 178, 53, 178, 3, 178, 16, 178, 61
                                        , 178, 58, 178, 83, 178, 61, 178, 156, 178, 102, 178, 48, 178, 102, 178, 53, 178, 54
                                        , 178, 51, 178, 16, 178, 46, 178, 156, 178, 69, 178, 158, 178, 92, 178, 51, 178, 3
                                        , 178, 16, 178, 157, 178, 86, 178, 55, 178, 157, 178, 86, 178, 55, 178, 156, 178, 86
                                        , 178, 55, 178, 3, 178, 16, 178, 43, 178, 102, 178, 16, 178, 48, 178, 156, 178, 138
                                        , 178, 53, 178, 16, 178, 55, 178, 43, 178, 102, 178, 61, 178, 156, 178, 86, 178, 54
                                        , 178, 48, 178, 4, 178};

        ov::Core core;

        std::cout << "Loading model files: " << model_path << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        ov::CompiledModel compiled_model = core.compile_model(model);

        ov::InferRequest infer_request = compiled_model.create_infer_request();

        ov::Tensor input_tensor_0 = infer_request.get_input_tensor(0);
        ov::Shape tensor_shape = input_tensor_0.get_shape();
        //std::cout << tensor_shape << std::endl;
        input_tensor_0.set_shape({1, text_inputs.size()});
        //std::cout << input_tensor_0.get_shape() << std::endl;
        int64_t* x0 = input_tensor_0.data<int64_t>();
        for (auto t : text_inputs) {
            (*x0) = t;
            x0 += 1;
        }

        /*
        int64_t* x1 = input_tensor_0.data<int64_t>();
        for (int j = 0; j < text_inputs.size(); j++){
            std::cout << *(x1x+j) << std::endl;
        }*/


        ov::Tensor input_tensor_1 = infer_request.get_input_tensor(1);
        input_tensor_1.set_shape({1});
        int64_t* x1 = input_tensor_1.data<int64_t>();
        *x1 = text_inputs.size();

        /*
        int64_t* yy = input_tensor_1.data<int64_t>();
        std::cout << *(yy) << std::endl;*/

        ov::Tensor input_tensor_2 = infer_request.get_input_tensor(2);
        //std::cout << "input_tensor_2.get_shape():" << input_tensor_2.get_shape() << std::endl;
        float* x2 = input_tensor_2.data<float>();
        std::vector<float> scales = {0.667, 1.0, 1.0};
        //*z = scales[0];
        //*(z+1) = scales[1];
        //*(z+2) = scales[2];
        for (auto s: scales) {
            *x2 = s;
            x2 += 1;
        }

        /*
        for (auto idx =0; idx < input_tensor_2.get_shape()[0]; idx++) {
            std::cout << *(input_tensor_2.data<float>()+idx) << "; ";
        }
        std::cout << std::endl;
        std::cout << "input_tensor_2.get_shape():" << input_tensor_2.get_shape() << std::endl;
        std::cout << "input is ok\n";

        std::cout << (infer_request.get_input_tensor(0).get_shape()) << std::endl;
        std::cout << *(infer_request.get_input_tensor(1).data<int64_t>()) << std::endl;

        std::cout << "check input\n";
        ov::Tensor a = infer_request.get_input_tensor(0);
        std::cout << a.get_shape() << std::endl;*/


        infer_request.infer();

        /*
        std::cout << "input\n";
        // -------- Step 9. Process output
        const ov::Tensor& input_0 = infer_request.get_tensor("input");
        std::cout << input_0.get_shape() << std::endl;
        for (auto y0 = 0; y0 < input_0.get_size(); y0++) {
            std::cout << input_0.data<int64_t>()[y0] << ";";
        }
        std::cout << std::endl;

        std::cout << "input_lengths:\n";
        const ov::Tensor& input_1 = infer_request.get_tensor("input_lengths");
        std::cout << input_1.get_shape() << std::endl;
        for (auto y1 = 0; y1 < input_1.get_size(); y1++) {
            std::cout << input_1.data<int64_t>()[y1] << ";";
        }
        std::cout << std::endl;
        
        std::cout << "scales:\n";
        const ov::Tensor& input_2 = infer_request.get_tensor("scales");
        std::cout << input_2.get_shape() << std::endl;
        for (auto y2 = 0; y2 < input_2.get_size(); y2++) {
            std::cout << input_2.data<float>()[y2] << ";";
        }
        std::cout << std::endl;*/

        std::cout << "output:\n";
        const ov::Tensor& output_tensor1 = infer_request.get_output_tensor();
        std::cout << output_tensor1.get_shape() << std::endl;
        for (auto yo = 0; yo < output_tensor1.get_size(); yo++) {
            std::cout << output_tensor1.data<float>()[yo] << ",";
        }
        std::cout << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}