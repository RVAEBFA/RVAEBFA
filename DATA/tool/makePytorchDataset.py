r'''
读取数据到 pytorch 的 dataset 数据库
dataset 包括
train set : train data, train labels

test set : test data, test labels

'''

import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd

class SelfDataLoader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
        data = np.load(data_path)

        labels = data["Y"]
        features = data["X"]
        labels = labels.flatten()
        N, D = features.shape
        normal_data = features[labels==-1]
        normal_labels = labels[labels==-1]
        normal_labels=normal_labels*(-1)

        N_normal = normal_data.shape[0]
        print(N_normal)

        attack_data = features[labels==1]
        attack_labels = labels[labels==1]
        attack_labels=attack_labels*(-1)

        N_attack = attack_data.shape[0]
        print(N_attack)

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        N_normal_train=(N_normal//10)*8

        self.train = attack_data[randIdx[:N_train]]
        self.train_labels = attack_labels[randIdx[:N_train]]

        self.test = attack_data[randIdx[N_train:]]
        self.test_labels = attack_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, normal_data[N_normal_train:]),axis=0)
        self.test_labels = np.concatenate((self.test_labels, normal_labels[N_normal_train:]),axis=0)
        self.train = np.concatenate((self.train, normal_data[:N_normal_train]),axis=0)
        self.train_labels = np.concatenate((self.train_labels, normal_labels[:N_normal_train]),axis=0)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""
    dataset = SelfDataLoader(data_path, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
