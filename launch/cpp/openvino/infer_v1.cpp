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
#include <string>
#include "boost/regex.hpp"
#include "speak_lib.h"
#include "openvino/openvino.hpp"
#include "httplib.h"
using namespace httplib;

void decode_character(std::map<std::string, int>& char2id_map,
                    const std::string& input_str,
                    std::vector<int>& ids) {
    size_t idx = 0;
    while (idx < input_str.size()) {
		if ((input_str[idx] & 0x80) != 0){
            std::string ch = input_str.substr(idx, 2);
            ids.push_back(char2id_map[ch]);
            idx += 2;
        } else {
            std::string ch = input_str.substr(idx, 1);
            ids.push_back(char2id_map[ch]);
            idx += 1;
        }

    }      
}

void  insert_blank(const std::vector<int>& input_ids,
                 const int& blank_id,
                 std::vector<int>& output_ids) {
    output_ids.push_back(blank_id);

    for(auto id : input_ids) {
        output_ids.push_back(id);
        output_ids.push_back(blank_id);
    }                
}


int strip_to_restore(const std::string& text,
                    std::vector<std::tuple<std::string, std::string>>& puncs,
                    std::vector<std::string>& sub_texts) {
    std::cout << "get text:" << text << std::endl;
    boost::regex reg("[«»,.!]+");
    boost::sregex_iterator it(text.begin(),text.end(),reg);
    boost::sregex_iterator begin, end;

    while (it!=end) {
        std::string punc_label = (it++)->str();
        std::string position = "MIDDLE";
        if (text.substr(0, punc_label.size()) == punc_label)
            position = "BEGIN";
        else if (text.substr(text.size()-punc_label.size(), text.size()) == punc_label) {
            position = "END";
        }
            
        puncs.push_back(std::tuple<std::string,std::string>(punc_label, position));
    }

    boost::sregex_token_iterator tit(text.begin(),text.end(),reg,-1);
    boost::sregex_token_iterator tend;
    while (tit!=tend) {
        std::string t = *tit++;
        if (t != "")
            sub_texts.push_back(t);
    }
}


int restore(std::vector<std::tuple<std::string, std::string>>& puncs,
            std::vector<std::string>& sub_texts,
            std::vector<std::string>& text) {
    if (puncs.size() <= 0) {
        std::string text_s = "";
        for (auto s : sub_texts) {
            text_s += s;
        }
        text.push_back(text_s);
        std::cout << text[0] << std::endl;
    } else if (sub_texts.size() <= 0) {
        std::string punc_str = "";
        for (auto p : puncs) {
            punc_str += std::get<0>(p);
        }
        text.push_back(punc_str);
        std::cout << text[0] << std::endl;
    } else {

        std::tuple<std::string, std::string> current_punc = puncs[0];
        if (std::get<1>(current_punc) == "BEGIN") {
            std::string prefix_str = std::get<0>(current_punc) + sub_texts[0];
            puncs.erase(puncs.begin());
            sub_texts[0] = prefix_str;
            restore(puncs, sub_texts, text);
        } else  if (std::get<1>(current_punc) == "END") {
            std::string prefix_str = sub_texts[0] + std::get<0>(current_punc);
            std::vector<std::string> suffix_str;
            puncs.erase(puncs.begin());
            sub_texts.erase(sub_texts.begin());
            restore(puncs, sub_texts, suffix_str);
            text.push_back(prefix_str + suffix_str[0]);
        } else  if (sub_texts.size() == 1) {
            sub_texts[0] += std::get<0>(current_punc);
            puncs.erase(puncs.begin());
            restore(puncs, sub_texts, text);
        } else {
            sub_texts[1] = sub_texts[0] + std::get<0>(current_punc) + sub_texts[1];
            sub_texts.erase(sub_texts.begin());
            puncs.erase(puncs.begin());
            restore(puncs, sub_texts, text);
        }
    }
}

void removeChars(std::string str, const std::string& charsToRemove, std::string& output_str) {
    output_str = str;
    output_str.erase(std::remove_if(output_str.begin(), output_str.end(), [&](char c) {
        return charsToRemove.find(c) != std::string::npos;
    }), output_str.end());
}


class A {
  private:
    std::string content = " AAAA";
  public:
    A(){};
    void get_content(const std::string input, std::string& output) {output = input + content;}
};

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

        const std::string model_path = argv[1];

        ov::Core core;

        std::cout << "Loading model files: " << model_path << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        ov::CompiledModel compiled_model = core.compile_model(model);

        ov::InferRequest infer_request = compiled_model.create_infer_request();

        ov::Tensor input_tensor_0 = infer_request.get_input_tensor(0);
        
        // initialization
        std::map<std::string, int> char2id = {{"_",0}, {";",1}, {":",2}, {",",3}, {".",4}, {"!",5}, {"?",6}, {"¡",7}, {"¿",8}, {"—",9}, {"…",10}, {"\"",11}, {"«",12}, {"»",13}, {"“",14}, {"”",15}, {" ",16}, {"A",17}, {"B",18}, {"C",19}, {"D",20}, {"E",21}, {"F",22}, {"G",23}, {"H",24}, {"I",25}, {"J",26}, {"K",27}, {"L",28}, {"M",29}, {"N",30}, {"O",31}, {"P",32}, {"Q",33}, {"R",34}, {"S",35}, {"T",36}, {"U",37}, {"V",38}, {"W",39}, {"X",40}, {"Y",41}, {"Z",42}, {"a",43}, {"b",44}, {"c",45}, {"d",46}, {"e",47}, {"f",48}, {"g",49}, {"h",50}, {"i",51}, {"j",52}, {"k",53}, {"l",54}, {"m",55}, {"n",56}, {"o",57}, {"p",58}, {"q",59}, {"r",60}, {"s",61}, {"t",62}, {"u",63}, {"v",64}, {"w",65}, {"x",66}, {"y",67}, {"z",68}, {"ɑ",69}, {"ɐ",70}, {"ɒ",71}, {"æ",72}, {"ɓ",73}, {"ʙ",74}, {"β",75}, {"ɔ",76}, {"ɕ",77}, {"ç",78}, {"ɗ",79}, {"ɖ",80}, {"ð",81}, {"ʤ",82}, {"ə",83}, {"ɘ",84}, {"ɚ",85}, {"ɛ",86}, {"ɜ",87}, {"ɝ",88}, {"ɞ",89}, {"ɟ",90}, {"ʄ",91}, {"ɡ",92}, {"ɠ",93}, {"ɢ",94}, {"ʛ",95}, {"ɦ",96}, {"ɧ",97}, {"ħ",98}, {"ɥ",99}, {"ʜ",100}, {"ɨ",101}, {"ɪ",102}, {"ʝ",103}, {"ɭ",104}, {"ɬ",105}, {"ɫ",106}, {"ɮ",107}, {"ʟ",108}, {"ɱ",109}, {"ɯ",110}, {"ɰ",111}, {"ŋ",112}, {"ɳ",113}, {"ɲ",114}, {"ɴ",115}, {"ø",116}, {"ɵ",117}, {"ɸ",118}, {"θ",119}, {"œ",120}, {"ɶ",121}, {"ʘ",122}, {"ɹ",123}, {"ɺ",124}, {"ɾ",125}, {"ɻ",126}, {"ʀ",127}, {"ʁ",128}, {"ɽ",129}, {"ʂ",130}, {"ʃ",131}, {"ʈ",132}, {"ʧ",133}, {"ʉ",134}, {"ʊ",135}, {"ʋ",136}, {"ⱱ",137}, {"ʌ",138}, {"ɣ",139}, {"ɤ",140}, {"ʍ",141}, {"χ",142}, {"ʎ",143}, {"ʏ",144}, {"ʑ",145}, {"ʐ",146}, {"ʒ",147}, {"ʔ",148}, {"ʡ",149}, {"ʕ",150}, {"ʢ",151}, {"ǀ",152}, {"ǁ",153}, {"ǂ",154}, {"ǃ",155}, {"ˈ",156}, {"ˌ",157}, {"ː",158}, {"ˑ",159}, {"ʼ",160}, {"ʴ",161}, {"ʰ",162}, {"ʱ",163}, {"ʲ",164}, {"ʷ",165}, {"ˠ",166}, {"ˤ",167}, {"˞",168}, {"↓",169}, {"↑",170}, {"→",171}, {"↗",172}, {"↘",173}, {"̩",175}, {"'",176}, {"ᵻ",177}, {"<BLNK>",178}};
    

        Server svr;
        svr.Get("/get-content", [&](const Request &req, Response &res) {
            std::string input = "Hello,world!";
            if (req.has_param("text")) {
            input = req.get_param_value("text");
            }
            std::cout << "input:" << input << std::endl;
            /*A a;
            std::string c = "";
            a.get_content(input, c);

            res.set_content(c + " sent!", "text/plain");
            //res.set_file_content("/data/coastcao/HelloTorch/launch/cpp/openvino/output.wav", "audio/wave");
            res.status = StatusCode::OK_200;*/
        
            //std::string input_text = "«Hello,world». I'm Coast, I am very happy to see that it works!Love you all!";

            std::string input_text = input;

            std::vector<std::tuple<std::string, std::string>> puncs;
            std::vector<std::string> sub_texts;
            strip_to_restore(input_text, puncs, sub_texts);

            std::vector<std::string> sub_phonemes;

            for (auto s : sub_texts) {
                const std::string* s_ptr = &s;
                //std::cout << "processing:" << *s_ptr << std::endl;
                std::string output = "";
                espeak_Text2Phonemes((const void**)&s_ptr, 1, 19, &output);
                if ((s[0] != ' ') && (output[0] == ' ')) {
                    output = output.substr(1,output.size()-1);
                }
                std::string output1 = "";
                removeChars(output, "_", output1);
                sub_phonemes.push_back(output1);
            }


            std::vector<std::string> restored_text;
            restore(puncs, sub_phonemes, restored_text);

            std::vector<int> ids;
            decode_character(char2id, restored_text[0], ids);

            std::vector<int> phoneme_ids;
            insert_blank(ids, 178, phoneme_ids);


            //ov::Shape tensor_shape = input_tensor_0.get_shape();
            //std::cout << tensor_shape << std::endl;
            input_tensor_0.set_shape({1, phoneme_ids.size()});
            //std::cout << input_tensor_0.get_shape() << std::endl;
            int64_t* x0 = input_tensor_0.data<int64_t>();
            for (auto t : phoneme_ids) {
                (*x0) = t;
                x0 += 1;
            }

            ov::Tensor input_tensor_1 = infer_request.get_input_tensor(1);
            input_tensor_1.set_shape({1});
            int64_t* x1 = input_tensor_1.data<int64_t>();
            *x1 = phoneme_ids.size();

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


            infer_request.infer();
            const ov::Tensor& output_tensor1 = infer_request.get_output_tensor();
            std::vector<float> wav_data;
            for (auto yo = 0; yo < output_tensor1.get_size(); yo++) {
                //std::cout << output_tensor1.data<float>()[yo] << ",";
                wav_data.push_back(output_tensor1.data<float>()[yo]);
            }

            // 数据归一化
            std::vector<float> wav_data_abs(wav_data.size());
            std::transform(wav_data.begin(), wav_data.end(), wav_data_abs.begin(), [](float x) { return std::abs(x); });
            auto max_it = std::max_element(wav_data_abs.begin(), wav_data_abs.end());
            auto max_val = std::max(0.01f, *max_it);
            std::transform(wav_data.begin(), wav_data.end(), wav_data.begin(), [max_val](float x) { return x * (32767 / max_val); });
            std::vector<int16_t> wav_data_int(wav_data.size());
            std::transform(wav_data.begin(), wav_data.end(), wav_data_int.begin(), [](float x) { return int16_t(x); });

            // 写入数据到文件
            SndfileHandle file("output.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, 22050);
            file.write(wav_data_int.data(), wav_data_int.size());

            if (input_text == "STOP") {
                res.set_content("OK, stop!", "text/plain");
                res.status = StatusCode::Locked_423;
            } else {
                res.set_file_content("/data/coastcao/HelloTorch/launch/cpp/openvino/build/output.wav", "audio/wave");
                res.status = StatusCode::OK_200;
            }
        });

        svr.listen("localhost", 1234);
    } catch (const std::exception& ex) {
            std::cerr << ex.what() << std::endl;
            return EXIT_FAILURE;
    }

    std::cout << "Finished successfully!\n";
    return EXIT_SUCCESS;
}