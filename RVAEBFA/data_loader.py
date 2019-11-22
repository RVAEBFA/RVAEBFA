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
from sklearn.preprocessing import MinMaxScaler

class SelfDataLoader(object):
    def __init__(self, data_path, mode="train", test_P_N_ratio = 2, train_contaminate_ratio = 0.1):
        '''
        test_P_N_ratio  表示测试集正负样本的数量比例 正样本：normal  负样本 ： anomaly
        '''
        data = np.load(data_path)
        labels = data["Y"]
        features = data["X"]
        minMax = MinMaxScaler()
        features = minMax.fit_transform(features)
        labels = labels.flatten()
        normal_data = features[labels==-1]
        normal_labels = labels[labels==-1]

        N_normal = normal_data.shape[0]
        attack_data = features[labels==1]
        attack_labels = labels[labels==1]

        N_attack = attack_data.shape[0]
        randIdx = np.arange(N_normal)
        np.random.shuffle(randIdx)
        self.mode=mode
        if self.mode == 'train':
            print('!-------train mode--------!')
            train_contaminate_idx = np.random.choice(attack_labels.shape[0], int(train_contaminate_ratio * attack_labels.shape[0]))
            self.train = np.concatenate([attack_data[train_contaminate_idx], normal_data], axis=0)
            self.train_labels = np.concatenate([attack_labels[train_contaminate_idx], normal_labels], axis = 0)
            randIdx = np.arange(self.train.shape[0])
            np.random.shuffle(randIdx)
            self.train = self.train[randIdx]
            self.train_labels = self.train_labels[randIdx]
            print('train dataset normal samples:   {}'.format(sum(self.train_labels == -1)))
            print('train dataset abnormal samples: {}'.format(sum(self.train_labels == 1)))
        else:
            print('!-------test mode--------!')
            randIdx = np.arange(N_normal)
            normal_idx = np.random.choice(randIdx, int(N_attack * test_P_N_ratio))
            test_normal = normal_data[normal_idx]
            test_normal_labels = normal_labels[normal_idx]
            
            self.test = np.concatenate((test_normal, attack_data), axis=0)
            self.test_labels = np.concatenate((test_normal_labels, attack_labels), axis=0)
            print('test normal samples:    ', sum(self.test_labels == -1))
            print(' test abnormal samples: ', sum(self.test_labels==1))
            # shuffle test data
            test_idx = np.arange(self.test_labels.size)
            np.random.shuffle(test_idx)
            self.test = self.test[test_idx]
            self.test_labels = self.test_labels[test_idx]


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            #print(self.train.shape[0])
            return self.train.shape[0]
        else:
            return self.test.shape[0]
    def test_PN_percent(self,):
        return int(sum(self.test_labels == -1) / len(self.test_labels) * 100)



    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

def get_loader(data_path, batch_size, mode='train', test_P_N_ratio = 2, train_contaminate_ratio = 0.1):
    """Build and return data loader."""
    dataset = SelfDataLoader(data_path, mode, test_P_N_ratio, train_contaminate_ratio)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
