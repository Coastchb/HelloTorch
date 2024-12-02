import openvino as ov
from pathlib import Path
import numpy as np
import sys
from typing import List
import torch
import time

sys.path.append('/root/coastcao/HelloTorch/TTS')
from TTS.tts.utils.text.cleaners import basic_cleaners
from TTS.tts.models.vits import Vits
from TTS.config import load_config
from TTS.utils.audio.numpy_transforms import save_wav


onnx_model_path = Path('../models/coqui_vits_G.xml')
input_text = 'No, so being able to, like, get in different positions for you, Like, specifically doggie, Mmm, I fuck myself.'
input_text = 'Thank you for your support, looking foward to continuing our cooperation next time'

# convert text to sequence of token IDs
config_file_path = '../models/G_config.json'
config = load_config(config_file_path)
vits_model = Vits.init_from_config(config)
input_tokens = np.asarray(
    vits_model.tokenizer.text_to_ids(input_text),
    dtype=np.int32,
)
print('input_text:', input_text)
print('input_tokens:', input_tokens)
print('len:', len(input_tokens))
# infer
core = ov.Core()
model = core.read_model(model=onnx_model_path)
compiled_model = core.compile_model(model=model)

output_layer = compiled_model.output(0)

start = time.time()
scale = torch.FloatTensor([0.667, 1.0, 1.0])
print(np.expand_dims(input_tokens, 0).shape)
print([len(input_tokens)])
print(scale)
# sid is valid only if it is multi-speaker model
model_outputs = compiled_model({
    "input": np.expand_dims(input_tokens, 0),
    "input_lengths": [len(input_tokens)],
    "scales": scale,
    "sid": torch.tensor([94])})[output_layer]
print('model_output:', model_outputs)
print('model_outputs.shape:', model_outputs.shape)


wav = None
def inv_spectrogram(postnet_output, ap, CONFIG):
    if CONFIG.model.lower() in ["tacotron"]:
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_melspectrogram(postnet_output.T)
    return wav
def save_wav_to_file(wav: List[int], path: str, pipe_out=None) -> None:
    """Save the waveform as a file.

    Args:
        wav (List[int]): waveform as a list of values.
        path (str): output path to save the waveform.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    # if tensor convert to numpy
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    save_wav(wav=wav, path=path, sample_rate=22050, pipe_out=pipe_out)
model_outputs = model_outputs.squeeze()
print('model_outputs:', model_outputs)
print('model_outputs.shape:', model_outputs.shape)
wav = model_outputs
save_wav_to_file(wav=wav, path='output.wav')

end = time.time()
print('consumed:', end - start)
