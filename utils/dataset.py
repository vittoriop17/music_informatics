from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
