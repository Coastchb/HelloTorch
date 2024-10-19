import torch
import os
import time
import sys

sys.path.append('/root/coastcao/HelloTorch/TTS')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from TTS.api import TTS
os.environ['TTS_HOME'] = os.curdir

# Example voice cloning with YourTTS in English, French and Portuguese
#tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to('cpu')
#tts.tts_to_file("This is voice cloning.", speaker_wav="~/Downloads/LJ001-0001_16k.wav", language="en", file_path="output.wav")


# Init TTS with the target model name
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False).to('cpu')

# Run TTS
#tts.tts_to_file(speed=1.0, text="Good night, baby, do you miss me, keep going, keep going, baby, it makes me feel so good, fuck", file_path='output_wavs/vits/vits_output_0.wav')
#tts.tts_to_file(speed=0.5, text="Good night, baby, do you miss me, keep going, keep going, baby, it makes me feel so good, fuck", file_path='output_wavs/vits/vits_output_1.wav')
#tts.tts_to_file(text="Here's the dill, sweet boy,We both really wanna suck your cock", file_path='output_wavs/vits/vits_output_2.wav')
#tts.tts_to_file(text="No, so being able to, like, get in different positions for you,Like, specifically doggie, Mmm, I fuck myself.", file_path='output_wavs/vits/vits_output_3.wav')
tts.tts_to_file(text="Oh, baby, you are so sexy, I love it!", file_path='output_wavs/vits/vits_output_4.wav')
# bad case
#tts.tts_to_file(text="Fuck, Fuck, I love your body, baby. Do you love me?", file_path='output_wavs/vits/vits_output_5.wav')
#tts.tts_to_file(text="Oh, fuck, I love your body, baby. Do you love me?", file_path='output_wavs/vits/vits_output_5.wav')
#tts.tts_to_file(text="If eligible, you can receive a share of ads revenue just by posting on X. You must be subscribed to X Premium.", file_path="output_wavs/vits/vits_output_6.wav")


# Run TTS and VC
#t0=time.time()
#tts.tts_with_vc_to_file(
#    "No, so being able to, like, get in different positions for you, Like, specifically doggie, Mmm, I fuck myself.",
    #"No. so being able to, like, get in different positions for you.",
#    speaker_wav="ref_wav/coast.wav",
#    file_path="output_wavs/vits/vits_vc_0.wav"
#)
#t1=time.time()
#print('vc consumed:', t1 - t0, 's')

