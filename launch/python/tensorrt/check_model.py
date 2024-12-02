import sys
import onnx
filename = "../models/coqui_vits.onnx"
model = onnx.load(filename)
onnx.checker.check_model(model)
