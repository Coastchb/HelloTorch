import tensorrt as trt
import sys

def load_engine(trt_runtime, engine_path):
    trt.init_libnvinfer_plugins(None, "")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def print_bindings(engine):
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        #is_input = engine.bindings_is_input(i)
        print(f"Tensor {i}:")
        print(f"  Name: {tensor_name}")
        print(f"  Shape: {tensor_shape}")
        print(f"  Dtype: {tensor_dtype}")
        #print(f"  Is input: {is_input}")

engine_path = sys.argv[1]
engine = load_engine(trt.Runtime(trt.Logger(trt.Logger.INFO)), engine_path)

print_bindings(engine)
