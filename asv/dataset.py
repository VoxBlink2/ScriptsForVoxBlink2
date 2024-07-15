from scipy.signal import fftconvolve
from python_speech_features import sigproc
from torch.utils.data import Dataset
import scipy.io.wavfile as sciwav
import torch
class WavDataset(Dataset):
    def __init__(self, wav_scp):
        self.wav_scp = wav_scp

    def __len__(self):
        return len(self.wav_scp)
    
    def _load_data(self, filename):
        sr, signal = sciwav.read(filename, mmap=True)
        return signal
    

    def __getitem__(self, idx):
        utt, filename = self.wav_scp[idx]
        signal = self._load_data(filename)
        signal = sigproc.preemphasis(signal, 0.97)
        signal = torch.from_numpy(signal.astype('float32'))
        return signal, utt
