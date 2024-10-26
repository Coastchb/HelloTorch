cmake_minimum_required(VERSION 3.26)
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

#set(openvino_lib_dir "/root/coastcao/openvino_2024.4.0/runtime/lib/intel64/")
#file(GLOB OpenVINO_LIBs "${openvino_lib_dir}/*.so")

#set(TDD_LIBs "/data/coastcao/oneTBB/build/gnu_12.3_cxx11_64_relwithdebinfo/libtbb.so")
set(TDD_LIBs "/data/coastcao/tools/oneTBB/build/gnu_12.3_cxx11_64_relwithdebinfo/libtbb.so.12.14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
include_directories(
    /opt/intel/openvino_2024/runtime/include/
    /opt/intel/openvino_2024/samples/cpp/common/utils/include
    /opt/intel/openvino_2024/samples/cpp/common/format_reader/include
)
add_executable(infer infer.cpp )
target_link_libraries(infer ${TDD_LIBs} ${OpenVINO_LIBs})
