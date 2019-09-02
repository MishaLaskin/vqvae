import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


class ImageDataset(Dataset):
    """
    Creates image dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, 
                file_path, 
                train=True, 
                transform=None, 
                make_temporal=False, 
                include_goals=False,
                path_length=100):
        print('Loading data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading data')
        img_data = np.array(data.item().get('image_observation'))
        self.goals = None
        if include_goals:
            self.goals = np.array(data.item().get('achieved_goal'))
          
        data = img_data
        self.n = data.shape[0]
        self.cutoff = self.n//10
        self.data = data[:-self.cutoff] if train else data[-self.cutoff:]
        self.transform = transform
        self.make_temporal = make_temporal
        self.path_length = path_length
        self.train = train

    def __getitem__(self, index):
        img = self.data[index]
        if self.goals is not None:
            goal = self.goals[index]
        else:
            goal = 0
        if self.transform is not None:
            img = self.transform(img)

        label = 0

        # if we want to keep track of adjacent states
        # e.g.s obs_t and obs_{t-1} we use make_temporal
        if self.train and self.make_temporal:
            if index % self.path_length == self.path_length-1:
                img2 = self.data[index]
            elif index % self.path_length > self.path_length-10:
                img2 = self.data[index+1]
            else:
                img2 = self.data[index+np.random.randint(10)]
            img2 = self.transform(img2)

            return img, img2, goal
        return img, goal

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
