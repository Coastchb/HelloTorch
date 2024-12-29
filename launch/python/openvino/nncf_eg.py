import nncf
import torch
import os
from dataclasses import dataclass, field
import openvino as ov
from trainer import Trainer, TrainerArgs
import sys
sys.path.append('/root/coastcao/HelloTorch/TTS')

from torch.utils.data import Dataset, DataLoader
from TTS.tts.models.vits import Vits
from TTS.config import load_config
import numpy as np

input_dir = 'f001_text'
# convert text to sequence of token IDs
config_file_path = '../models/G_config.json'
config = load_config(config_file_path)
vits_model = Vits.init_from_config(config)
input_tokens = []
for f in os.listdir(input_dir):
    input_text = open(os.path.join(input_dir, f)).readlines()[0].strip()
    input_tokens.append(vits_model.tokenizer.text_to_ids(input_text))

print('input_tokens num:', len(input_tokens))

class VitsDataset(Dataset):
    def __init__(self):
        self.input_tokens = input_tokens
        self.pad_id = 0
    def __getitem__(self,index):
        sample = {'input': self.input_tokens[index],
                  'input_lengths': len(self.input_tokens[index])}
        return sample
    def __len__(self):
        return len(self.input_tokens)
    def collate_fn(self, batch):
        #print('batch:', batch)
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        input_tokens = batch['input']
        input_lengths = batch['input_lengths']
        
        max_len = max(input_lengths)
        token_padded = torch.LongTensor(B, max_len)
        token_padded = token_padded.zero_() + self.pad_id
        for i in range(B):
            #print('B:', B, ';i:', i)
            token_ids = input_tokens[i]
            token_padded[i, : input_lengths[i]] = torch.LongTensor(token_ids)
        
        return {'input': token_padded,
                'input_lengths': input_lengths,
                'scales': torch.tensor([0.667, 1.0, 1.0],dtype=torch.float32),
                'sid': [0]}
        
vits_dataset = VitsDataset()
vits_dataloader = DataLoader(vits_dataset, batch_size=10, collate_fn=vits_dataset.collate_fn, shuffle=False)


calibration_dataset = nncf.Dataset(vits_dataloader)

raw_model_path = "../models/coqui_vits_G.onnx"
onnx_model = ov.Core().read_model(raw_model_path)

quantized_model = nncf.quantize(onnx_model, calibration_dataset)

# save the model
ov.save_model(quantized_model, "quantized_model.xml")
