//
// Created by coastcao(操海兵) on 2019-08-26.
//

#include "boost/regex.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


int strip_to_restore(const std::string& text,
                    std::vector<std::tuple<std::string, std::string>>& puncs,
                    std::vector<std::string>& sub_texts) {
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


int main() {
    std::vector<std::string> text_list = {"«Hello,world». I'm Coast, I am very happy to see that it works!",
                                            "«.!Hello,world»., I'm Coast,! I am very happy to see that it works",
                                            "Hello, world  I'm Coast  I am very happy to see that it works",
                                            "Hello world  I'm Coast  I am very happy to see that it works",
                                            ",.!"
    };

    for (auto s : text_list) {
        std::vector<std::tuple<std::string, std::string>> puncs;
        std::vector<std::string> sub_texts;

        strip_to_restore(s, puncs, sub_texts);


        for (auto x : puncs)
            std::cout << std::get<0>(x) << "\t" << std::get<1>(x) << "\n";

        for (auto x : sub_texts)
            std::cout << x << "\t";
        std::cout << std::endl;

        std::vector<std::string> restored_text;
        restore(puncs, sub_texts, restored_text);
        std::cout << "raw text:" << s << std::endl;
        std::cout << "restored text:" << restored_text[0] << std::endl;
    }
}
