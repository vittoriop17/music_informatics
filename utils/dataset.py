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
from statsmodels.tsa.stattools import adfuller
# from utils import upload_args


class MusicDataset(Dataset):
    def __init__(self, args, skip=False):
        self.args = args
        self.dataset_path = args.dataset_path
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
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ProcessChannels("avg"),
                MinMaxScaler()
            ])

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
        # audio_samples = (audio_samples - np.min(audio_samples, axis=1, keepdims=True)) / np.max(audio_samples, axis=1,
        #                                                                                          keepdims=True)
        return self.transform(audio_samples), self.classes[index].toarray()

    def get_audio_paths_n_classes(self):
        """
        :return: list of strings, list of strings. Respectively: the list of audio file paths; the list of audio classes
        They have the same length
        """
        tot_files = 0
        file_names = list()
        file_classes = list()
        for (root, dirs, files) in os.walk(self.dataset_path, topdown=True):
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

    def copy_files_to(self, dest_path):
        for audio_path in self.audio_file_paths:
            audio_name = os.path.basename(audio_path)
            audio_class = os.path.basename(os.path.dirname(audio_path))
            try:
                os.makedirs(os.path.join(dest_path, audio_class))
            except:
                pass
            copyfile(audio_path, os.path.join(dest_path, audio_class, audio_name))


class TestDataset(MusicDataset):
    def __init__(self, args, root_path=None, single_class=True, classes="all", split_dataset=False):
        self.args = args
        self.single_class = single_class
        self.classes = classes if classes != 'all' else ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru',
                                                         'vio', 'voi']
        if split_dataset:
            self.split_dataset()
        super(TestDataset, self).__init__(args, skip=True)
        self.dataset_path = root_path
        self.audio_file_paths, self.classes, self.nominal_classes = self.get_audio_paths_n_classes()
        self.transform = transforms.Compose([CropAudio(),
                                            self.transform])


    def split_dataset(self):
        n_files_not_found = 0
        one_inst_files, more_inst_files = (list(), list())
        one_inst_class, more_inst_classes = (0, 0)
        for (root, dirs, files) in os.walk(self.dataset_path, topdown=True):
            base_class = os.path.basename(root)
            for file in files:
                if not file.endswith(".txt"):
                    continue
                wav_file_name = str(file).replace(".txt", ".wav")
                wav_file_path = os.path.join(root, wav_file_name)
                if not os.path.exists(wav_file_path):
                    print(f"Audio file not found. No match for current metadata: {str(file)}")
                    n_files_not_found += 1
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r") as fp:
                    lines = fp.readlines()
                n_lines = len(lines)
                lines = list(map(lambda line: line.strip(), lines))
                n_instruments = np.sum([1 for line in lines if line in self.classes])
                if n_lines != n_instruments:
                    print(f"Found invalid class in metadata. Audio file skipped {wav_file_path}")
                    continue
                if len(lines) == 1:
                    base_dir = os.path.join(self.dataset_path, "one_instrument", lines[0])
                    wav_file_path_dest = os.path.join(base_dir, wav_file_name)
                    one_inst_files.append(wav_file_path_dest)
                    one_inst_class += 1
                else:
                    classes = "_".join(lines)
                    base_dir = os.path.join(self.dataset_path, "more_instrument", str(n_instruments), classes)
                    wav_file_path_dest = os.path.join(base_dir, wav_file_name)
                    more_inst_files.append(wav_file_path_dest)
                    more_inst_classes += 1
                try:
                    os.makedirs(base_dir)
                except Exception:
                    pass
                copyfile(wav_file_path, wav_file_path_dest)
                os.remove(wav_file_path)
        print(f"Number files with one instrument: {one_inst_class}")
        print(f"Number files with more instruments: {more_inst_classes}")

    def __getitem__(self, index):
        if self.single_class:
            return super(TestDataset, self).__getitem__(index)
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.single_class:
            return super(TestDataset, self).__len__()
        else:
            raise NotImplementedError()


class StdScaler(object):
    def __init__(self):
        pass

    def _normalize(self, x):
        x = (x - torch.mean(x)) / torch.std(x)
        return x

    def __call__(self, x):
        return self._normalize(x)


class MinMaxScaler(object):
    def __init__(self):
        super(MinMaxScaler, self).__init__()

    def _scaler(self, x: torch.tensor):
        assert x.ndim > 1
        return (x - torch.min(x)) / torch.max(x)

    def __call__(self, x: torch.tensor):
        return self._scaler(x)


class ProcessChannels(object):

    def __init__(self, mode):
        self.mode = mode

    def _modify_channels(self, waveform: torch.tensor, mode):
        if mode == 'avg':
            new_audio = torch.mean(waveform.float(), dim=1) if waveform.ndim > 1 else waveform
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


def check_stationarity(ds: MusicDataset):
    for i in range(20):
        audio, _ = ds[i]
        X = np.array(torch.squeeze(audio).detach().numpy())
        X_left = X[0, :]
        X_right = X[1, :]
        result = adfuller(X_left)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        result = adfuller(X_right)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))


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


if __name__ == '__main__':
    print()
    # main_dir = "D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\dataset\\IRMAS-TrainingData"
    # dataset_path = "D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\mini_dataset"
    # args = upload_args("..\\configuration.json")
    # setattr(args,"dataset_path", dataset_path)
    # ds = MusicDataset(args)
    # check_stationarity(ds)
    # TestDataset(root_path="D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\Dataset\\test")
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
    # args = upload_args(file_path="..\\configuration.json")
    # ds = MusicDataset(args)
    # ds_train, ds_test = stratified_split(ds, train_size=0.8, args=args)
    # # ds_train.copy_files_to(dest_path="..\\dataset\\train")
    # ds_train.copy_files_to(dest_path="..\\dataset\\test")
