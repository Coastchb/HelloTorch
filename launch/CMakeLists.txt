cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

#set(CMAKE_C_COMPILER "/opt/rh/devtoolset-9/root/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/opt/rh/devtoolset-9/root/usr/bin/g++")
add_executable(example example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 17)

#set(CMAKE_CXX_STANDARD 17) 
#set_property(TARGET example PROPERTY CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_COMPILER "/root/gcc-8.2.0/bin/c++")
#set_property(TARGET example PROPERTY CXX_COMPILER "/root/gcc-8.2.0/bin/c++")

