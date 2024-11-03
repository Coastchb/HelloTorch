cmake_minimum_required(VERSION 3.26)
project(text2phoneme)
set(ESpeak_LIBs 
    "/data/coastcao/tools/espeak-1.48.04-source/src/libespeak.so"
    )

#set(TDD_LIBs "/data/coastcao/tools/oneTBB/build/gnu_12.3_cxx11_64_relwithdebinfo/libtbb.so.12.14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
include_directories(
    /data/coastcao/tools/espeak-1.48.04-source/src
    /data/coastcao/tools/boost_1_86_0
)
add_executable(text2phoneme text2phoneme_v1.cpp )
target_link_libraries(text2phoneme ${ESpeak_LIBs})
