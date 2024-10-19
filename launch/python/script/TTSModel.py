import sys
sys.path.append('/root/coastcao/HelloTorch/TTS')

from TTS.tts.models.vits import Vits
from TTS.config import load_config
import torch

config = load_config('configs/config.json')
vits_model = Vits.init_from_config(config)

scripted_model = torch.jit.script(vits_model)
scripted_model.save('../models/scripted_vits.pt')