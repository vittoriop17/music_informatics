import os
from torch.utils.data import Dataset
from pydub import AudioSegment
import numpy as np
from shutil import copyfile


def read_audio(path):
    audio_seg = AudioSegment.from_wav(path)
    audio_samples = np.array(audio_seg.set_channels(1).get_array_of_samples(), dtype=np.float64)
    return audio_samples

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
    def __init__(self, args):
        self.check_args(args)
        self.args = args
        self.audio_file_paths, self.classes = self.get_audio_paths_n_classes()

    def __len__(self):
        return len(self.audio_file_paths)

    def __getitem__(self, index):
        audio_path = self.audio_file_paths[index]
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found! Path: {audio_path}")
        audio_samples = read_audio(audio_path)
        audio_samples = self.add_padding(audio_samples)
        return audio_samples.reshape(self.args.sequence_length, -1), self.classes[index]

    def add_padding(self, audio_samples):
        mod = len(audio_samples) % self.args.sequence_length
        if mod != 0:
            audio_samples = np.append(audio_samples, np.zeros((self.args.sequence_length - mod,)))
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
        return file_names, file_classes


if __name__=='__main__':
    print()
    # main_dir = "D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\dataset\\IRMAS-TrainingData"
    # dest_dir = "D:\\UNIVERSITA\\KTH\\Semestre 1\\Music Informatics\\Labs\\mini_dataset"
    #
    # create_mini_dataset(main_dir, dest_dir)
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
    #         if not len(samples) == 264598/2:
    #             cont += 1
    #         tot_files += 1
    #         file_classes.append(base_class)
    #     classes.append(base_class)
    # print(f"Tot files: {tot_files}")
    # print(f"Number files with length different from the standard one: {cont}")
    # print(f"Number of classes: {len(classes)}")
    # print(f"Number of audio x class: {np.unique(file_classes, return_counts=True)}")
