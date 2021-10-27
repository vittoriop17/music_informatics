import os
from torch.utils.data import Dataset
from pydub import AudioSegment
import numpy as np
from shutil import copyfile
import torchvision.transforms as transforms
import torch
from sklearn.preprocessing import OneHotEncoder
from scipy.io.wavfile import read
from sklearn.model_selection import StratifiedShuffleSplit
import torchaudio
import torchaudio.transforms
import matplotlib.pyplot as plt


def check_classes(ds_train, ds_test):
    cont_occ_train = np.zeros((1, 11))
    cont_occ_test = np.zeros((1, 11))
    for input_data, input_class in ds_train:
        cont_occ_train += input_class
    for _, input_class in ds_test:
        cont_occ_test += input_class
    print(cont_occ_train)
    print(cont_occ_test)


def read_audio(path):
    rate, audio_array = read(path)
    assert audio_array.shape[1] == 2, f"Audio channels is not 2. {audio_array.shape[1]} channel(s) found"
    return audio_array.reshape(2, -1)


def create_mini_dataset(path_src, path_dest):
    for (root, dirs, files) in os.walk(path_src, topdown=True):
        base_class = os.path.basename(root)
        os.makedirs(os.path.join(path_dest, base_class))
        for idx, file in enumerate(files):
            if idx > 20:
                break
            if not file.endswith(".wav"):
                continue
            copyfile(os.path.join(root, file), os.path.join(path_dest, base_class, file))


class MusicDataset(Dataset):
    def __init__(self, args, skip=False):
        self.check_args(args)
        self.args = args
        if skip:
            self.audio_file_paths, self.classes, self.nominal_classes = list(), list(), None
        else:
            self.audio_file_paths, self.classes, self.nominal_classes = self.get_audio_paths_n_classes()
        if args.train == "cnn":
            self.transform = transforms.Compose([
                ProcessChannels("avg"),
                CropAudio(),
                StdScaler(),
                torchaudio.transforms.Spectrogram(n_fft=1024, win_length=512, normalized=False),
                torchaudio.transforms.AmplitudeToDB(),
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.audio_file_paths)

    def __getitem__(self, index):
        audio_path = self.audio_file_paths[index]
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found! Path: {audio_path}")
        if self.args.train == "cnn":
            waveform, sample_rate = torchaudio.load(audio_path)
            # specgram = torchaudio.transforms.MelSpectrogram()(waveform)
            # print("Shape of spectrogram: {}".format(specgram.size()))
            # plt.figure()
            # plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
            # TODO - Data Normalization!!!! Apply the same normalization of ImageNet
            return self.transform(waveform), self.classes[index].toarray()
        audio_samples = read_audio(audio_path)
        # audio_samples = self.add_padding(audio_samples)
        audio_samples = (audio_samples - np.mean(audio_samples, axis=1, keepdims=True)) / np.std(audio_samples, axis=1, keepdims=True)
        return torch.tensor(audio_samples, dtype=torch.float32), self.classes[index].toarray()

    def add_padding(self, audio_samples):
        mod = audio_samples.shape[1] % self.args.sequence_length
        if mod != 0:
            audio_samples = np.concatenate((audio_samples, np.zeros((2, self.args.sequence_length - mod))), axis=1)
        # setattr(self.args, "input_size", int(len(audio_samples) / self.args.sequence_length))
        return audio_samples

    def check_args(self, args):
        """
        Check the presence of the following arguments:
        -dataset_path
        :param args: Namespace
        """
        assert hasattr(args, "dataset_path"), "Argument 'dataset_path' not found!"
        assert hasattr(args, "sequence_length"), "Argument 'sequence_length' not found!"

    def get_audio_paths_n_classes(self):
        """
        :return: list of strings, list of strings. Respectively: the list of audio file paths; the list of audio classes
        They have the same length
        """
        tot_files = 0
        file_names = list()
        file_classes = list()
        for (root, dirs, files) in os.walk(self.args.dataset_path, topdown=True):
            base_class = os.path.basename(root)
            for file in files:
                if not file.endswith(".wav"):
                    continue
                tot_files += 1
                file_names.append(os.path.join(root, file))
                file_classes.append(base_class)
        print(f"Tot files: {tot_files}")
        self.ohe = OneHotEncoder()
        ohe_classes = self.ohe.fit_transform(X=np.array(file_classes).reshape(-1, 1))
        return file_names, ohe_classes, file_classes


class StdScaler(object):
    def __init__(self):
        pass

    def _normalize(self, x):
        x = (x - torch.mean(x)) / torch.std(x)
        return x

    def __call__(self, x):
        return self._normalize(x)


class ProcessChannels(object):

    def __init__(self, mode):
        self.mode = mode

    def _modify_channels(self, waveform, mode):
        if mode == 'avg':
            new_audio = waveform.mean(axis=0) if waveform.ndim > 1 else waveform
            new_audio = torch.unsqueeze(new_audio, dim=0)
        else:
            new_audio = waveform
        return new_audio

    def __call__(self, tensor):
        return self._modify_channels(tensor, self.mode)


class CropAudio(object):
    def __init__(self, length=3, sample_rate=44100):
        self.length = length
        self.sample_rate = sample_rate

    def _crop_audio(self, waveform):
        num_tot_samples = self.length * self.sample_rate
        audio_len = waveform.shape[1]
        if audio_len > num_tot_samples:
            random_start = np.random.randint(0, audio_len - num_tot_samples)
            end = random_start + num_tot_samples
            waveform = waveform[:, random_start:end]
        return waveform

    def __call__(self, waveform):
        return self._crop_audio(waveform)


def stratified_split(ds: MusicDataset, args, train_size=0.8):
    sss = StratifiedShuffleSplit(1, train_size=train_size, random_state=42)
    ds_train = MusicDataset(args, skip=True)
    ds_test = MusicDataset(args, skip=True)
    for train_idx, test_idx in sss.split(ds.audio_file_paths, ds.nominal_classes):
        ds_train.audio_file_paths = np.array(ds.audio_file_paths)[train_idx]
        ds_train.classes = ds.classes[train_idx, :]
        ds_test.audio_file_paths = np.array(ds.audio_file_paths)[test_idx]
        ds_test.classes = ds.classes[test_idx, :]
    return ds_train, ds_test



if __name__=='__main__':
    print()
    # main_dir = "D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\dataset\\IRMAS-TrainingData"
    # dest_dir = "D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\mini_dataset"
    #
    # # create_mini_dataset(main_dir, dest_dir)
    # cont = 0
    # tot_files = 0
    # classes = list()
    # file_classes = list()
    # for (root, dirs, files) in os.walk(main_dir, topdown=True):
    #     base_class = os.path.basename(root)
    #     for file in files:
    #         if not file.endswith(".wav"):
    #             continue
    #         samples = read_audio(os.path.join(root, file))
    #         tot_files += 1
    #         file_classes.append(base_class)
    #     classes.append(base_class)
    # print(f"Tot files: {tot_files}")
    # print(f"Number of classes: {len(classes)}")
    # print(f"Number of audio x class: {np.unique(file_classes, return_counts=True)}")
