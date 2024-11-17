// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <sndfile.hh>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <fstream>
#include <iostream>
#include <thread>
#include <ctime>
#include <random>
#include <functional>


#include "boost/regex.hpp"
#include "speak_lib.h"
#include "openvino/openvino.hpp"

#include "cinatra.hpp"
#include "cinatra/metric_conf.hpp"
#include "utils.hpp"

using namespace cinatra;
using namespace ylt::metric;
using namespace std::chrono_literals;


async_simple::coro::Lazy<void> tts_http_server(std::map<std::string, int>& char2id, ov::CompiledModel& model, const status_type& init_status) {
    int max_thread_num = std::thread::hardware_concurrency();
    coro_http_server server(max_thread_num, 8080);
    std::cout << "max_thread_num:" << max_thread_num << std::endl;       

    server.set_http_handler<GET, POST>("/TTS",
        [&char2id,&model,&init_status](coro_http_request& req, coro_http_response& res) -> async_simple::coro::Lazy<void> {
        //std::cout << static_cast<int>(init_status) << std::endl;
        if (init_status != status_type::ok) {
            res.set_status_and_content(init_status, "Failed to init TTS model!\n");
            co_return;
        }
        std::string input = "Hello,world!";
        auto req_queris = req.get_queries();

        if (req_queris.contains("text")) {
            input = std::string(req.get_query_value("text"));
        }

        // in curl, query value cannot contains white space, use '%20' instead.
        replaceAll(input, "%20", " ");
        //std::cout << "input:" << input << std::endl; 

        std::string now = std::to_string(static_cast<int>(time(0)));
        int input_number = 0;
        get_random_number(input, input_number);
        std::string wav_filename = now.append("_").append(std::to_string(input_number)).append(".wav");
        //std::string wav_filename = "../output.wav";
        status_type tts_ret_status;
        tts(char2id, model, input, wav_filename, tts_ret_status);

        if (tts_ret_status != status_type::ok) {
            res.set_status_and_content(tts_ret_status, "Failed to run TTS!\n");
            co_return;
        }

        std::ifstream file(wav_filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file:" << wav_filename << std::endl;
            res.set_status_and_content(status_type::open_file_failed, "Failed to open wav file!\n");
            co_return;
        }
    
        // 读取文件全部内容到字符串流
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        std::string content = buffer.str();

        // 使用读取到的数据进行操作
        res.set_status_and_content(status_type::ok, content);

        std::remove(wav_filename.c_str());
    });

    server.sync_start();
}

/**
 * @brief Main with support Unicode paths, wide strings
 */
int main(int argc, char* argv[]) {
    // -------- Get OpenVINO runtime version --------
    // std::cout << ov::get_openvino_version() << std::endl;

    // -------- Parsing and validation of input arguments --------
    if (argc != 2) {
        std::cout << "Usage : " << argv[0] << " <path_to_model>"
                    << std::endl;
        return EXIT_FAILURE;
    }
    const std::string model_path = argv[1];
    std::cout << "Loading model files: " << model_path << std::endl;
        // initialization
    std::map<std::string, int> char2id = {{"_",0}, {";",1}, {":",2}, {",",3}, {".",4}, {"!",5}, {"?",6}, {"¡",7}, {"¿",8}, {"—",9}, {"…",10}, {"\"",11}, {"«",12}, {"»",13}, {"“",14}, {"”",15}, {" ",16}, {"A",17}, {"B",18}, {"C",19}, {"D",20}, {"E",21}, {"F",22}, {"G",23}, {"H",24}, {"I",25}, {"J",26}, {"K",27}, {"L",28}, {"M",29}, {"N",30}, {"O",31}, {"P",32}, {"Q",33}, {"R",34}, {"S",35}, {"T",36}, {"U",37}, {"V",38}, {"W",39}, {"X",40}, {"Y",41}, {"Z",42}, {"a",43}, {"b",44}, {"c",45}, {"d",46}, {"e",47}, {"f",48}, {"g",49}, {"h",50}, {"i",51}, {"j",52}, {"k",53}, {"l",54}, {"m",55}, {"n",56}, {"o",57}, {"p",58}, {"q",59}, {"r",60}, {"s",61}, {"t",62}, {"u",63}, {"v",64}, {"w",65}, {"x",66}, {"y",67}, {"z",68}, {"ɑ",69}, {"ɐ",70}, {"ɒ",71}, {"æ",72}, {"ɓ",73}, {"ʙ",74}, {"β",75}, {"ɔ",76}, {"ɕ",77}, {"ç",78}, {"ɗ",79}, {"ɖ",80}, {"ð",81}, {"ʤ",82}, {"ə",83}, {"ɘ",84}, {"ɚ",85}, {"ɛ",86}, {"ɜ",87}, {"ɝ",88}, {"ɞ",89}, {"ɟ",90}, {"ʄ",91}, {"ɡ",92}, {"ɠ",93}, {"ɢ",94}, {"ʛ",95}, {"ɦ",96}, {"ɧ",97}, {"ħ",98}, {"ɥ",99}, {"ʜ",100}, {"ɨ",101}, {"ɪ",102}, {"ʝ",103}, {"ɭ",104}, {"ɬ",105}, {"ɫ",106}, {"ɮ",107}, {"ʟ",108}, {"ɱ",109}, {"ɯ",110}, {"ɰ",111}, {"ŋ",112}, {"ɳ",113}, {"ɲ",114}, {"ɴ",115}, {"ø",116}, {"ɵ",117}, {"ɸ",118}, {"θ",119}, {"œ",120}, {"ɶ",121}, {"ʘ",122}, {"ɹ",123}, {"ɺ",124}, {"ɾ",125}, {"ɻ",126}, {"ʀ",127}, {"ʁ",128}, {"ɽ",129}, {"ʂ",130}, {"ʃ",131}, {"ʈ",132}, {"ʧ",133}, {"ʉ",134}, {"ʊ",135}, {"ʋ",136}, {"ⱱ",137}, {"ʌ",138}, {"ɣ",139}, {"ɤ",140}, {"ʍ",141}, {"χ",142}, {"ʎ",143}, {"ʏ",144}, {"ʑ",145}, {"ʐ",146}, {"ʒ",147}, {"ʔ",148}, {"ʡ",149}, {"ʕ",150}, {"ʢ",151}, {"ǀ",152}, {"ǁ",153}, {"ǂ",154}, {"ǃ",155}, {"ˈ",156}, {"ˌ",157}, {"ː",158}, {"ˑ",159}, {"ʼ",160}, {"ʴ",161}, {"ʰ",162}, {"ʱ",163}, {"ʲ",164}, {"ʷ",165}, {"ˠ",166}, {"ˤ",167}, {"˞",168}, {"↓",169}, {"↑",170}, {"→",171}, {"↗",172}, {"↘",173}, {"̩",175}, {"'",176}, {"ᵻ",177}, {"<BLNK>",178}};
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;

    status_type init_status = status_type::ok;
    try {
        model = core.read_model(model_path);
        compiled_model = core.compile_model(model);
        espeak_init();
    } catch (const std::exception& ex) {
        std::cerr << "Failed to init TTS model!\n";
        std::cerr << ex.what() << std::endl;
        init_status = status_type::init_tts_model_failed;
        //std::cout << static_cast<int>(init_status) << std::endl;
        //std::cout << static_cast<int>(status_type::init_tts_model_failed) << std::endl;
    }

    async_simple::coro::syncAwait(tts_http_server(char2id, compiled_model, init_status));

    std::cout << "Finished successfully!\n";
    return EXIT_SUCCESS;
}