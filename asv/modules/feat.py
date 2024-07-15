from torchaudio import transforms
import random
import torch, torch.nn as nn

class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, norm_type='std'):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels)
        self.norm_type = norm_type
    def _norm_speech(self, signal):
        if torch.std(signal) == 0:
            return signal
        if self.norm_type == 'std':
            signal = (signal - torch.mean(signal)) / torch.std(signal)
        else:
            signal = signal / (torch.abs(signal).max()+ 1e-4)
        return signal
    
    def forward(self, x):
        x = self._norm_speech(x)
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        return out