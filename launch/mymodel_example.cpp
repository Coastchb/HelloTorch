#include <torch/script.h>
#include <memory>

#include <iostream>
using namespace std;

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "loaded model " << argv[1] << " ok\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::tensor({{3, 2, 8}, {1,6,3}}));

  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  at::Tensor output1 = module.get_method("func1")(inputs).toTensor();
  std::cout << output1.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}