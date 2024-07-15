from scipy.signal import fftconvolve
from python_speech_features import sigproc
from torch.utils.data import Dataset
import scipy.io.wavfile as sciwav
import torch, numpy as np
class WavDataset(Dataset):
    def __init__(self, wav_scp,norm_type):
        self.wav_scp = wav_scp
        self.norm_type = norm_type
    def __len__(self):
        return len(self.wav_scp)
    
    def _load_data(self, filename):
        sr, signal = sciwav.read(filename, mmap=True)
        return signal
    def _norm_speech(self, signal):
        if np.std(signal) == 0:
            return signal
        if self.norm_type == 'std':
            signal = (signal - np.mean(signal)) / np.std(signal)
        else:
            signal = signal / (np.abs(signal).max()+ 1e-4)
        return signal
    def __getitem__(self, idx):
        utt, filename = self.wav_scp[idx]
        signal = self._load_data(filename)
        signal = self._norm_speech(signal)
        signal = sigproc.preemphasis(signal, 0.97)
        signal = torch.from_numpy(signal.astype('float32'))
        return signal, utt
