from torchaudio import transforms
import random
import torch, torch.nn as nn

class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=int(win_length*sample_rate),
                                                  hop_length=int(hop_length*sample_rate),
                                                  n_mels=n_mels)
    def forward(self, x):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        return out