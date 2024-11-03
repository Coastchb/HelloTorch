#include <iostream>
#include <string>
#include <vector>
#include "speak_lib.h"
#include <string>
#include "boost/regex.hpp"
#include <map>

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


int print_punc(std::vector<std::tuple<std::string, std::string>>& puncs, const std::string& descp) {
    for (auto x: puncs) {
        std::cout << "[" << descp << "]" << std::get<0>(x) << "\t";
    }
    std::cout << std::endl;
}

int print_text(std::vector<std::string>& text, const std::string& descp) {
    for (auto x: text) {
        std::cout << "[" << descp << "]" << x << "\t";
    }
    std::cout << std::endl;
}

int restore(std::vector<std::tuple<std::string, std::string>>& puncs,
            std::vector<std::string>& sub_texts,
            std::vector<std::string>& text) {
    //std::cout << "puncs.size:" << puncs.size() << "\tsub_texts.size:" << sub_texts.size() << std::endl;
    if (puncs.size() <= 0) {
        /*std::cout << "0:\n";
        print_punc(puncs, "0");
        print_text(sub_texts, "0");*/
        std::string text_s = "";
        for (auto s : sub_texts) {
            text_s += s;
        }
        text.push_back(text_s);
        /*print_punc(puncs, "00");
        print_text(sub_texts, "00");*/
        std::cout << text[0] << std::endl;
    } else if (sub_texts.size() <= 0) {
        /*std::cout << "1:\n";
        print_punc(puncs, "1");
        print_text(sub_texts, "1");*/
        std::string punc_str = "";
        for (auto p : puncs) {
            punc_str += std::get<0>(p);
        }
        /*print_punc(puncs, "11");
        print_text(sub_texts, "11");*/
        text.push_back(punc_str);
        std::cout << text[0] << std::endl;
    } else {

        std::tuple<std::string, std::string> current_punc = puncs[0];
        if (std::get<1>(current_punc) == "BEGIN") {
            /*std::cout << "2:\n";
            print_punc(puncs, "2");
            print_text(sub_texts, "2");*/
            std::string prefix_str = std::get<0>(current_punc) + sub_texts[0];
            puncs.erase(puncs.begin());
            sub_texts[0] = prefix_str;
            /*print_punc(puncs, "22");
            print_text(sub_texts, "22");*/
            restore(puncs, sub_texts, text);
        } else  if (std::get<1>(current_punc) == "END") {
            /*std::cout << "3:\n";
            print_punc(puncs, "3");
            print_text(sub_texts, "3");*/
            std::string prefix_str = sub_texts[0] + std::get<0>(current_punc);
            std::vector<std::string> suffix_str;
            puncs.erase(puncs.begin());
            sub_texts.erase(sub_texts.begin());
            /*print_punc(puncs, "33");
            print_text(sub_texts, "33");*/
            restore(puncs, sub_texts, suffix_str);
            text.push_back(prefix_str + suffix_str[0]);
        } else  if (sub_texts.size() == 1) {
            /*std::cout << "4:\n";
            print_punc(puncs, "4");
            print_text(sub_texts, "4");*/
            sub_texts[0] += std::get<0>(current_punc);
            puncs.erase(puncs.begin());
            /*print_punc(puncs, "44");
            print_text(sub_texts, "44");*/
            restore(puncs, sub_texts, text);
        } else {
            /*std::cout << "5:\n";
            print_punc(puncs, "5");
            print_text(sub_texts, "5");*/
            sub_texts[1] = sub_texts[0] + std::get<0>(current_punc) + sub_texts[1];
            sub_texts.erase(sub_texts.begin());
            puncs.erase(puncs.begin());
            /*print_punc(puncs, "55");
            print_text(sub_texts, "55");*/
            restore(puncs, sub_texts, text);
        }
    }
}

std::string removeChars(std::string str, const std::string& charsToRemove) {
    str.erase(std::remove_if(str.begin(), str.end(), [&](char c) {
        return charsToRemove.find(c) != std::string::npos;
    }), str.end());
    return str;
}

int main(int argc, char* argv[]) {
    // initialization
    std::map<std::string, int> char2id = {{"_",0}, {";",1}, {":",2}, {",",3}, {".",4}, {"!",5}, {"?",6}, {"¡",7}, {"¿",8}, {"—",9}, {"…",10}, {"\"",11}, {"«",12}, {"»",13}, {"“",14}, {"”",15}, {" ",16}, {"A",17}, {"B",18}, {"C",19}, {"D",20}, {"E",21}, {"F",22}, {"G",23}, {"H",24}, {"I",25}, {"J",26}, {"K",27}, {"L",28}, {"M",29}, {"N",30}, {"O",31}, {"P",32}, {"Q",33}, {"R",34}, {"S",35}, {"T",36}, {"U",37}, {"V",38}, {"W",39}, {"X",40}, {"Y",41}, {"Z",42}, {"a",43}, {"b",44}, {"c",45}, {"d",46}, {"e",47}, {"f",48}, {"g",49}, {"h",50}, {"i",51}, {"j",52}, {"k",53}, {"l",54}, {"m",55}, {"n",56}, {"o",57}, {"p",58}, {"q",59}, {"r",60}, {"s",61}, {"t",62}, {"u",63}, {"v",64}, {"w",65}, {"x",66}, {"y",67}, {"z",68}, {"ɑ",69}, {"ɐ",70}, {"ɒ",71}, {"æ",72}, {"ɓ",73}, {"ʙ",74}, {"β",75}, {"ɔ",76}, {"ɕ",77}, {"ç",78}, {"ɗ",79}, {"ɖ",80}, {"ð",81}, {"ʤ",82}, {"ə",83}, {"ɘ",84}, {"ɚ",85}, {"ɛ",86}, {"ɜ",87}, {"ɝ",88}, {"ɞ",89}, {"ɟ",90}, {"ʄ",91}, {"ɡ",92}, {"ɠ",93}, {"ɢ",94}, {"ʛ",95}, {"ɦ",96}, {"ɧ",97}, {"ħ",98}, {"ɥ",99}, {"ʜ",100}, {"ɨ",101}, {"ɪ",102}, {"ʝ",103}, {"ɭ",104}, {"ɬ",105}, {"ɫ",106}, {"ɮ",107}, {"ʟ",108}, {"ɱ",109}, {"ɯ",110}, {"ɰ",111}, {"ŋ",112}, {"ɳ",113}, {"ɲ",114}, {"ɴ",115}, {"ø",116}, {"ɵ",117}, {"ɸ",118}, {"θ",119}, {"œ",120}, {"ɶ",121}, {"ʘ",122}, {"ɹ",123}, {"ɺ",124}, {"ɾ",125}, {"ɻ",126}, {"ʀ",127}, {"ʁ",128}, {"ɽ",129}, {"ʂ",130}, {"ʃ",131}, {"ʈ",132}, {"ʧ",133}, {"ʉ",134}, {"ʊ",135}, {"ʋ",136}, {"ⱱ",137}, {"ʌ",138}, {"ɣ",139}, {"ɤ",140}, {"ʍ",141}, {"χ",142}, {"ʎ",143}, {"ʏ",144}, {"ʑ",145}, {"ʐ",146}, {"ʒ",147}, {"ʔ",148}, {"ʡ",149}, {"ʕ",150}, {"ʢ",151}, {"ǀ",152}, {"ǁ",153}, {"ǂ",154}, {"ǃ",155}, {"ˈ",156}, {"ˌ",157}, {"ː",158}, {"ˑ",159}, {"ʼ",160}, {"ʴ",161}, {"ʰ",162}, {"ʱ",163}, {"ʲ",164}, {"ʷ",165}, {"ˠ",166}, {"ˤ",167}, {"˞",168}, {"↓",169}, {"↑",170}, {"→",171}, {"↗",172}, {"↘",173}, {"̩",175}, {"'",176}, {"ᵻ",177}, {"<BLNK>",178}};


    std::string input_text = "«Hello,world». I'm Coast, I am very happy to see that it works!";
    std::vector<std::tuple<std::string, std::string>> puncs;
    std::vector<std::string> sub_texts;
    strip_to_restore(input_text, puncs, sub_texts);

    for (auto x : puncs)
        std::cout << std::get<0>(x) << "\t" << std::get<1>(x) << "\n";

    for (auto x : sub_texts)
        std::cout << x << "\t";
    std::cout << std::endl;


    std::vector<std::string> sub_phonemes;

    for (auto s : sub_texts) {
        const std::string* s_ptr = &s;
        std::cout << "processing:" << *s_ptr << std::endl;
        std::string output = "";
        espeak_Text2Phonemes((const void**)&s_ptr, 1, 19, &output);
        std::cout << "output:" << output << std::endl;
        if ((s[0] != ' ') && (output[0] == ' ')) {
            output = output.substr(1,output.size()-1);
        }
        std::cout << "after processed output:" << output << std::endl;
        output = removeChars(output, "_");
        std::cout << "after removed output:" << output << std::endl;
        sub_phonemes.push_back(output);
    }


    std::vector<std::string> restored_text;
    restore(puncs, sub_phonemes, restored_text);
    std::cout << "raw text:" << input_text << std::endl;
    std::cout << "restored text:" << restored_text[0] << std::endl;

    std::vector<int> ids;
    decode_character(char2id, restored_text[0], ids);
    for (auto x: ids) {
        std::cout << x << ",";
    }
    std::cout << std::endl;

    std::vector<int> inserted_ids;
    insert_blank(ids, 178, inserted_ids);

    std::cout << "raw input:" << input_text << std::endl;
    for (auto x: inserted_ids) {
        std::cout << x << ", ";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}