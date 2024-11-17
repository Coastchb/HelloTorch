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


#include "boost/regex.hpp"
#include "speak_lib.h"
#include "openvino/openvino.hpp"
#include "cinatra.hpp"
#include "cinatra/metric_conf.hpp"
using namespace cinatra;

enum class tts_status {
  init,
  ok = 200,
  init_failed = 300,
  inference_failed = 400
};

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
    //std::cout << "get text:" << text << std::endl;
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
        //std::cout << text[0] << std::endl;
    } else if (sub_texts.size() <= 0) {
        std::string punc_str = "";
        for (auto p : puncs) {
            punc_str += std::get<0>(p);
        }
        text.push_back(punc_str);
        //std::cout << text[0] << std::endl;
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


void tts(std::map<std::string, int>& char2id, ov::CompiledModel& model,
         const std::string& input_text, const std::string& output_wav_filename,
         status_type& return_status) {
    try{
        std::vector<std::tuple<std::string, std::string>> puncs;
        std::vector<std::string> sub_texts;
        strip_to_restore(input_text, puncs, sub_texts);

        std::vector<std::string> sub_phonemes;

        for (auto s : sub_texts) {
            const std::string* s_ptr = &s;
            //std::cout << "processing:" << *s_ptr << std::endl;
            std::string output = "";
            espeak_Text2Phonemes((const void**)&s_ptr, &output);
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

        ov::InferRequest infer_request = model.create_infer_request();

        ov::Tensor input_tensor_0 = infer_request.get_input_tensor(0);

        input_tensor_0.set_shape({1, phoneme_ids.size()});
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
        SndfileHandle file(output_wav_filename, SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, 22050);
        file.write(wav_data_int.data(), wav_data_int.size());
    } catch(const std::exception& ex) {
        std::cerr << "Failed to run TTS inference\n";
        std::cerr << ex.what() << std::endl;
        return_status = status_type::tts_inference_failed;
    }
    return_status = status_type::ok;
}

void get_random_number(const std::string& str, int& random_int) {
    std::hash<std::string> hash_fn;
    std::mt19937 rng(hash_fn(str));  // 初始化随机数生成器
    std::uniform_int_distribution<int> distr(0, std::numeric_limits<int>::max());
    random_int = distr(rng);  // 生成随机数
}

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // 防止无限循环，如果`to`包含`from`
    }
}