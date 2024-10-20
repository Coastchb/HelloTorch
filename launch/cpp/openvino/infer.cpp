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
int tmain(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        std::cout << ov::get_openvino_version() << std::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 4) {
            std::cout << "Usage : " << argv[0] << " <path_to_model>"
                       << std::endl;
            return EXIT_FAILURE;
        }

        const std::string args = argv[0];
        const std::string model_path = argv[1];

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        std::cout << "Loading model files: " << model_path << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        //printInputAndOutputsInfo(*model);

        OPENVINO_ASSERT(model->inputs().size() == 3, "Sample supports models with 3 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 3. Set up input

        std::vector text_inputs = {178, 156, 178,  57, 178, 135, 178,   3, 178,  16, 178,  44, 178, 156,
        178,  47, 178, 102, 178,  44, 178,  51, 178,   3, 178,  16, 178,  52,
        178,  63, 178, 158, 178,  16, 178,  69, 178, 158, 178, 123, 178,  16,
        178,  61, 178, 157, 178,  57, 178, 135, 178,  16, 178,  61, 178, 156,
        178,  86, 178,  53, 178,  61, 178,  51, 178,   3, 178,  16, 178,  43,
        178, 102, 178,  16, 178,  54, 178, 156, 178, 138, 178,  64, 178,  16,
        178, 102, 178,  62, 178,   5, 178};

        ov::element::Type input_type = ov::element::u8;
        ov::Shape input_shape = {1, 1, text_inputs.size()};
        std::shared_ptr<unsigned char> input_data = text_inputs;

        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

        //const ov::Layout tensor_layout{"NHWC"};

        // -------- Step 4. Configure preprocessing --------

        ov::preprocess::PrePostProcessor ppp(model);

        // 1) Set input tensor information:
        // - input() provides information about a single model input
        // - reuse precision and shape from already available `input_tensor`
        // - layout of data is 'NHWC'
        // ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        // 2) Adding explicit preprocessing steps:
        // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
        // - apply linear resize from tensor spatial dims to model spatial dims
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        // 4) Suppose model has 'NCHW' layout for input
        //ppp.input().model().set_layout("NCHW");
        // 5) Set output tensor information:
        // - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 6) Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // -------- Step 5. Loading a model to the device --------
        ov::CompiledModel compiled_model = core.compile_model(model);

        // -------- Step 6. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------- Step 7. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();



        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}