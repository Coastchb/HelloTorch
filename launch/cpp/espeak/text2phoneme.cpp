#include <iostream>
#include <string>
#include <vector>
#include "speak_lib.h"
#include <string>

int main(int argc, char* argv[]) {
    std::string input_text = "Hello,world. I am Coast, I am very happy to see that it works!";
    const std::string* input_text_ptr = &input_text;
    //const void** input_text_ptr_ptr = &input_text_ptr;
    //auto samplerate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,0,NULL,0);
    std::string output = "";
    const char* ret = espeak_Text2Phonemes((const void**)&input_text_ptr, 1, 19, &output);
    std::cout << "output:" << output << std::endl;
    std::cout << ret << std::endl;
    return EXIT_SUCCESS;
}