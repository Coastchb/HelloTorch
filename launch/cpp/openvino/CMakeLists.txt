cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcoroutines")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-tree-slp-vectorize")
project(infer)
set(OpenVINO_DIR /opt/intel/openvino_2024/runtime/cmake/)
set(OpenVINO_LIBs 
        #"/root/coastcao/openvino_2024.4.0/runtime/lib/intel64/libopenvino_c.so"
        #"/root/coastcao/openvino_2024.4.0/runtime/lib/intel64/libopenvino_hetero_plugin.so"
        #"/root/coastcao/openvino_2024.4.0/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so"
        #"/root/coastcao/openvino_2024.4.0/runtime/lib/intel64/libopenvino_onnx_frontend.so"
        #"/root/coastcao/openvino_2024.4.0/runtime/lib/intel64/libopenvino_tensorflow_frontend.so"
        "/opt/intel/openvino_2024/runtime/lib/intel64/libopenvino.so"
    )
set(ESpeak_LIBs 
    "/data/coastcao/tools/espeak-1.48.04-source/src/libespeak.so"
    )
#set(openvino_lib_dir "/root/coastcao/openvino_2024.4.0/runtime/lib/intel64/")
#file(GLOB OpenVINO_LIBs "${openvino_lib_dir}/*.so")

#set(TDD_LIBs "/data/coastcao/oneTBB/build/gnu_12.3_cxx11_64_relwithdebinfo/libtbb.so")
set(TDD_LIBs "/data/coastcao/tools/oneTBB/build/gnu_12.3_cxx11_64_relwithdebinfo/libtbb.so.12.14")

#set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
include_directories(
    /opt/intel/openvino_2024/runtime/include/
    /opt/intel/openvino_2024/samples/cpp/common/utils/include
    /opt/intel/openvino_2024/samples/cpp/common/format_reader/include
    /data/coastcao/tools/espeak-1.48.04-source/src
    /data/coastcao/tools/boost_1_86_0
    /data/coastcao/HelloTorch/launch/cpp/openvino/third-party/httplib
)
#add_executable(infer infer.cpp )
#target_link_libraries(infer ${TDD_LIBs} ${OpenVINO_LIBs} ${ESpeak_LIBs} -lsndfile -lssl -lcrypto)

add_executable(infer_v1 infer_v1.cpp )
target_link_libraries(infer_v1 ${TDD_LIBs} ${OpenVINO_LIBs} ${ESpeak_LIBs} -lsndfile -lssl -lcrypto)

