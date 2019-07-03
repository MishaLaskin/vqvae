import cv2
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Creates image dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading data')
        data = np.array(data.item().get('image_observation'))

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


class StateDataset(Dataset):
    """
    Creates state dataset
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading state data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading state data')
        data = np.array(data.item().get('achieved_goal'))

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
