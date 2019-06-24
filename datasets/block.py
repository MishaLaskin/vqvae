import cv2
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):
    """
    Creates block dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')
        data = np.array([cv2.resize(x[0][0][:, :, :3], dsize=(
            32, 32), interpolation=cv2.INTER_CUBIC) for x in data])

        n = data.shape[0]
        cutoff = n//10
        self.data = data[:-cutoff] if train else data[-cutoff:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset 
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading latent block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading latent block data')
        
        self.data = data[:-500] if train else data[-500:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)
