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

        std::vector<int> text_inputs = {178, 156, 178,  57, 178, 135, 178,   3, 178,  16, 178,  44, 178, 156,
        178,  47, 178, 102, 178,  44, 178,  51, 178,   3, 178,  16, 178,  52,
        178,  63, 178, 158, 178,  16, 178,  69, 178, 158, 178, 123, 178,  16,
        178,  61, 178, 157, 178,  57, 178, 135, 178,  16, 178,  61, 178, 156,
        178,  86, 178,  53, 178,  61, 178,  51, 178,   3, 178,  16, 178,  43,
        178, 102, 178,  16, 178,  54, 178, 156, 178, 138, 178,  64, 178,  16,
        178, 102, 178,  62, 178,   5, 178};
        

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        //ov::Core core = ov::Core("/data/coastcao/openvino_2024.4.0/runtime/lib/intel64/pkgconfig/openvino.pc");
        //ov::Core core = ov::Core("/data/coastcao/openvino_2024.4.0/runtime/lib/intel64/pkgconfig/openvino.pc");
        // const std::string config_path = "/data/coastcao/openvino_2024.4.0/runtime/lib/intel64/pkgconfig/openvino.pc";
        ov::Core core;


        // -------- Step 2. Read a model --------
        std::cout << "Loading model files: " << model_path << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        //printInputAndOutputsInfo(*model);
        std::cout << "load model successfully!" << std::endl;

        /*
        auto inputs = model->inputs();
        std::cout << "input num:" << inputs.size() << std::endl;
        // change the input as dynamic shape support

        for(auto input_one : inputs){         
            auto input_shape = input_one.get_partial_shape();    
            std::cout << "[0] input_shape:" << input_shape << std::endl;    
            //input_shape[0] = 1;         
            //input_shape[1] = text_inputs.size();       
        }

        auto input_0 = inputs[0];
        auto input_shape_0 = input_0.get_partial_shape();
        input_shape_0[0] = 1;
        input_shape_0[1] = text_inputs.size();
        std::cout << input_0 << std::endl;
        std::cout << input_shape_0 << std::endl;

        //inputs[0].set_partial_shape(input_shape_0);
        //std::cout << input_0 << std::endl;

        

        auto input_1 = inputs[0];
        auto input_shape_1 = input_1.get_partial_shape(); 
        input_shape_1[0] = 1;


        for(auto input_one : inputs){         
            auto input_shape = input_one.get_partial_shape();    
            std::cout << "input_shape:" << input_shape << std::endl;    
            //input_shape[0] = 1;         
            //input_shape[1] = text_inputs.size();       
        }

        std::cout << "ok 0\n";
        */
        
        //OPENVINO_ASSERT(model->inputs().size() == 3, "Sample supports models with 3 input only");
        //OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        ov::CompiledModel compiled_model = core.compile_model(model);

        ov::InferRequest infer_request = compiled_model.create_infer_request();

        ov::Tensor input_tensor_0 = infer_request.get_input_tensor(0);
        ov::Shape tensor_shape = input_tensor_0.get_shape();
        std::cout << tensor_shape << std::endl;
        input_tensor_0.set_shape({1, text_inputs.size()});
        std::cout << input_tensor_0.get_shape() << std::endl;

        std::cout << input_tensor_0.data<int64_t>() << std::endl;
        std::cout << std::endl;

        int64_t* x = input_tensor_0.data<int64_t>();
        std::cout << *x << std::endl;

        for (auto t : text_inputs) {
            (*x) = t;
            x += 1;
        }

        /*
        int64_t* xx = input_tensor_0.data<int64_t>();
        for (int j = 0; j < text_inputs.size(); j++){
            std::cout << *(xx+j) << std::endl;
        }*/


        ov::Tensor input_tensor_1 = infer_request.get_input_tensor(1);
        input_tensor_1.set_shape({1});
        int64_t* y = input_tensor_1.data<int64_t>();
        *y = text_inputs.size();

        int64_t* yy = input_tensor_1.data<int64_t>();
        std::cout << *(yy) << std::endl;

        ov::Tensor input_tensor_2 = infer_request.get_input_tensor(2);
        std::cout << "input_tensor_2.get_shape():" << input_tensor_2.get_shape() << std::endl;

        float* z = input_tensor_2.data<float>();
        std::vector<float> scales = {0.667, 1.0, 1.0};
        /*
        for (auto s : scales) {
            (*z) = s;
            z += 1;
        }*/


        for (auto idx =0; idx < input_tensor_2.get_shape()[0]; idx++) {
            std::cout << *(input_tensor_2.data<float>()+idx) << "; ";
        }
        std::cout << std::endl;
        std::cout << "input_tensor_2.get_shape():" << input_tensor_2.get_shape() << std::endl;
        std::cout << "input is ok\n";

        /*
        ov::element::Type input_type = ov::element::u8;
        ov::Shape input_shape = {1, 1, text_inputs.size()};
        std::shared_ptr<unsigned char> input_data = text_inputs;

        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor input_tensor_0 = ov::Tensor(input_type, input_shape, input_data.get());*/

        
        //const ov::Layout tensor_layout{"NHWC"};

        // -------- Step 4. Configure preprocessing --------

        //ov::preprocess::PrePostProcessor ppp(model);

        // 1) Set input tensor information:
        // - input() provides information about a single model input
        // - reuse precision and shape from already available `input_tensor_0`
        // - layout of data is 'NHWC'
        // ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        // 2) Adding explicit preprocessing steps:
        // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
        // - apply linear resize from tensor spatial dims to model spatial dims
        //ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        // 4) Suppose model has 'NCHW' layout for input
        //ppp.input().model().set_layout("NCHW");
        // 5) Set output tensor information:
        // - precision of tensor is supposed to be 'f32'
        //ppp.output().tensor().set_element_type(ov::element::f32);

        // 6) Apply preprocessing modifying the original 'model'
        //model = ppp.build();

        // -------- Step 5. Loading a model to the device --------
        //ov::CompiledModel compiled_model = core.compile_model(model);

        // -------- Step 6. Create an infer request --------
        //ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------- Step 7. Prepare input --------
        //infer_request.set_input_tensor(input_tensor_0);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();
        
        //float* o = output_tensor.data<float>();
        
        //std::cout << *o << std::endl;
        std::cout << output_tensor.get_shape() << std::endl;


        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}