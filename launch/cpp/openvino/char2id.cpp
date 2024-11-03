// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "iostream"
#include <map>
#include <string>
#include <vector>
// clang-format on


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
int main() {
    // initialization
    std::map<std::string, int> char2id = {{"_",0}, {";",1}, {":",2}, {",",3}, {".",4}, {"!",5}, {"?",6}, {"¡",7}, {"¿",8}, {"—",9}, {"…",10}, {"\"",11}, {"«",12}, {"»",13}, {"“",14}, {"”",15}, {" ",16}, {"A",17}, {"B",18}, {"C",19}, {"D",20}, {"E",21}, {"F",22}, {"G",23}, {"H",24}, {"I",25}, {"J",26}, {"K",27}, {"L",28}, {"M",29}, {"N",30}, {"O",31}, {"P",32}, {"Q",33}, {"R",34}, {"S",35}, {"T",36}, {"U",37}, {"V",38}, {"W",39}, {"X",40}, {"Y",41}, {"Z",42}, {"a",43}, {"b",44}, {"c",45}, {"d",46}, {"e",47}, {"f",48}, {"g",49}, {"h",50}, {"i",51}, {"j",52}, {"k",53}, {"l",54}, {"m",55}, {"n",56}, {"o",57}, {"p",58}, {"q",59}, {"r",60}, {"s",61}, {"t",62}, {"u",63}, {"v",64}, {"w",65}, {"x",66}, {"y",67}, {"z",68}, {"ɑ",69}, {"ɐ",70}, {"ɒ",71}, {"æ",72}, {"ɓ",73}, {"ʙ",74}, {"β",75}, {"ɔ",76}, {"ɕ",77}, {"ç",78}, {"ɗ",79}, {"ɖ",80}, {"ð",81}, {"ʤ",82}, {"ə",83}, {"ɘ",84}, {"ɚ",85}, {"ɛ",86}, {"ɜ",87}, {"ɝ",88}, {"ɞ",89}, {"ɟ",90}, {"ʄ",91}, {"ɡ",92}, {"ɠ",93}, {"ɢ",94}, {"ʛ",95}, {"ɦ",96}, {"ɧ",97}, {"ħ",98}, {"ɥ",99}, {"ʜ",100}, {"ɨ",101}, {"ɪ",102}, {"ʝ",103}, {"ɭ",104}, {"ɬ",105}, {"ɫ",106}, {"ɮ",107}, {"ʟ",108}, {"ɱ",109}, {"ɯ",110}, {"ɰ",111}, {"ŋ",112}, {"ɳ",113}, {"ɲ",114}, {"ɴ",115}, {"ø",116}, {"ɵ",117}, {"ɸ",118}, {"θ",119}, {"œ",120}, {"ɶ",121}, {"ʘ",122}, {"ɹ",123}, {"ɺ",124}, {"ɾ",125}, {"ɻ",126}, {"ʀ",127}, {"ʁ",128}, {"ɽ",129}, {"ʂ",130}, {"ʃ",131}, {"ʈ",132}, {"ʧ",133}, {"ʉ",134}, {"ʊ",135}, {"ʋ",136}, {"ⱱ",137}, {"ʌ",138}, {"ɣ",139}, {"ɤ",140}, {"ʍ",141}, {"χ",142}, {"ʎ",143}, {"ʏ",144}, {"ʑ",145}, {"ʐ",146}, {"ʒ",147}, {"ʔ",148}, {"ʡ",149}, {"ʕ",150}, {"ʢ",151}, {"ǀ",152}, {"ǁ",153}, {"ǂ",154}, {"ǃ",155}, {"ˈ",156}, {"ˌ",157}, {"ː",158}, {"ˑ",159}, {"ʼ",160}, {"ʴ",161}, {"ʰ",162}, {"ʱ",163}, {"ʲ",164}, {"ʷ",165}, {"ˠ",166}, {"ˤ",167}, {"˞",168}, {"↓",169}, {"↑",170}, {"→",171}, {"↗",172}, {"↘",173}, {"̩",175}, {"'",176}, {"ᵻ",177}, {"<BLNK>",178}};

    /*
    //std::map<std::string, int> char2id = {{u8"_",10}, {u8"«",12}, {u8"h", 128} };
    for(auto it=char2id.begin();it!=char2id.end();it++)
   {
       std::cout<<it->first<<" "<<it->second<< std::endl;
   }*/

    std::vector<int> ids;
    std::string input_str = "«həlˈoʊ,wˈɜːld». aɪm kˈoʊst, aɪɐm vˈɛɹi hˈæpi tə sˈiː ðˌɐɾɪt wˈɜːks!";
    //std::string input_str = "«_«h";
    //std::cout << input_str.substr(0,2) << std::endl;
    decode_character(char2id, input_str, ids);
    for (auto x: ids) {
        std::cout << x << ",";
    }
    std::cout << std::endl;

    std::vector<int> inserted_ids;
    insert_blank(ids, 178, inserted_ids);
    for (auto x: inserted_ids) {
        std::cout << x << ",";
    }
    std::cout << std::endl;

}