import torch
import numpy as np
import librosa

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.var_a = torch.tensor(np.array([[0.1, 0.2, 0.6], [0.3, 0.5, 0.9]]))
        self.var_b = torch.tensor(np.array([[0.4, 0.8, 0.6], [0.3, 0.2, 0.0]]))

    @torch.jit.export
    def func1(self, input):
        print('in func1')
        print('input.shape:', input.shape)
        print('input:', input)
        return self.var_a + self.var_b

    def forward(self, input):
        print('in forward')
        #return self.var_a * self.var_b
        #return input
        print(torch.cat((self.var_a, self.var_b), 0))
        return torch.cat((self.var_a, self.var_b), 0)

    '''
    @torch.jit.export
    def func2(self, input):
        audio_data = '../data/mel001.wav'
        data, sr = librosa.load(audio_data, sr=22050, mono=True,offset=0.0, duration=None, dtype=np.float32, res_type="soxr_hq")

        #y_hat = librosa.resample(data, orig_sr=22050, target_sr=44100)
        duration1 = librosa.get_duration(y=data)
        #duration2 = librosa.get_duration(y=y_hat, sr=44100)
        print('原始时长：', duration1)
        #print('重采样后时长：', duration2)

        return self.var_a
    '''


my_model = MyModel()

'''
with open('./model.pth', 'wb') as fd:
    if hasattr(my_model, "module"):
        model_state = my_model.module.state_dict()
    else:
        model_state = my_model.state_dict()
    torch.save({"model": model_state}, fd)
'''

scripted_model = torch.jit.script(my_model)
scripted_model.save('../models/scripted_mymodel.pt')
