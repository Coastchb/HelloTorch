import onnxruntime as ort
import numpy as np
import torch
import time
from typing import List
import sys
import scipy


use_gpu = False
print('use_gpu:', use_gpu)

ort.set_default_logger_severity(0)

options = ort.SessionOptions()
options.log_severity_level=0
ort_sess = ort.InferenceSession('../models/coqui_vits_G.onnx', providers=['CUDAExecutionProvider' if use_gpu else 'CPUExecutionProvider'], sess_options=options)
#ort_sess.log_severity_level(1)
#ort_sess.set_run_options(ort.RunOptions(log_severity_level=1))
providers=['CUDAExecutionProvider']
print(ort_sess.get_providers())

print('ort.version:', ort.__version__)
start = time.time()
input_tokens = [178,56,178,156,178,57,178,135,178,3,178,16,178,61,178,157,178,57,178,135,178,16,178,44,178,157,178,51,178,158,178,102,178,112,178,16,178,156,178,47,178,102,178,44,178,83,178,54,178,16,178,62,178,63,178,158,178,3,178,16,178,54,178,156,178,43,178,102,178,53,178,3,178,16,178,92,178,86,178,62,178,16,178,102,178,56,178,16,178,46,178,156,178,102,178,48,178,123,178,83,178,56,178,62,178,16,178,58,178,83,178,68,178,156,178,102,178,131,178,83,178,56,178,68,178,16,178,48,178,76,178,158,178,123,178,16,178,52,178,63,178,158,178,3,178,16,178,54,178,156,178,43,178,102,178,53,178,3,178,16,178,61,178,58,178,83,178,61,178,156,178,102,178,48,178,102,178,53,178,54,178,51,178,16,178,46,178,156,178,69,178,158,178,92,178,51,178,3,178,16,178,157,178,86,178,55,178,157,178,86,178,55,178,156,178,86,178,55,178,3,178,16,178,43,178,102,178,16,178,48,178,156,178,138,178,53,178,16,178,55,178,43,178,102,178,61,178,156,178,86,178,54,178,48,178,4,178]
input_tokens = [178,119,178,156,178,72,178,112,178,53,178,16,178,52,178,63,178,158
                ,178,16,178,48,178,76,178,158,178,123,178,16,178,52,178,135,178,123
                ,178,16,178,61,178,83,178,58,178,156,178,57,178,158,178,123,178,62
                ,178,3,178,16,178,54,178,156,178,135,178,53,178,102,178,112,178,16
                ,178,48,178,156,178,43,178,135,178,85,178,46,178,16,178,62,178,83
                ,178,16,178,53,178,83,178,56,178,62,178,156,178,102,178,56,178,52
                ,178,63,178,158,178,102,178,112,178,16,178,157,178,43,178,135,178,85
                ,178,16,178,53,178,57,178,135,178,156,178,69,178,158,178,58,178,85
                ,178,123,178,156,178,47,178,102,178,131,178,83,178,56,178,16,178,56
                ,178,156,178,86,178,53,178,61,178,62,178,16,178,62,178,156,178,43
                ,178,102,178,55,178]
scale = [0.6670, 1.0000, 1.0000]

print('input tokens len:', len(input_tokens))
outputs = ort_sess.run(None, {'input': np.expand_dims(input_tokens, 0),
                            'input_lengths': [len(input_tokens)],
                            'scales': scale,
                            'sid': torch.tensor([0]).numpy()})
# Print Result
model_outputs = outputs[0][0][0]

print('outputs.shape:', outputs[0].shape)


wav = None

def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, pipe_out=None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)

def inv_spectrogram(postnet_output, ap, CONFIG):
    if CONFIG.model.lower() in ["tacotron"]:
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_melspectrogram(postnet_output.T)
    return wav
def save_wav_to_file(wav: List[int], path: str, pipe_out=None) -> None:
    """
    Save the waveform as a file.

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
