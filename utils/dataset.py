import os
from torch.utils.data import Dataset
from pydub import AudioSegment
import numpy as np


def read_audio(path):
    audio_seg = AudioSegment.from_wav(path)
    audio_samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float64)


def check_args(args):
    raise NotImplementedError()


class MusicDataset(Dataset):
    def __init__(self, args):
        check_args(args)
        self.args = args
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()



if __name__=='__main__':
    main_dir = "D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\ataset\\IRMAS-TrainingData\\cel"
    for path in os.walk():
        read_audio()