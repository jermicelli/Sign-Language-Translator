from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch

import csv

class datasetASL(Dataset):
    def labelMapping():
        mapping = list(range(25))
        mapping.pop(9)
        return mapping
    def readCSV(path: str):
        mapping = datasetASL.labelMapping()
        labels, images = [], [] 
        with open(path) as f:
            _ = next(f) 
            for line in csv.reader(f):
                label = int(line[0])
                labels.append(mapping.index(label))
                images.append(list(map(int, line[1:])))
        return labels, images
    def __init__(self, path: str = "D:/Projects/MachineLearningASL/input/sign_mnist_train/sign_mnist_train.csv", mean: float=0.485, std: float=0.229):
        labels, images = datasetASL.readCSV(path)
        self._images = np.array(images, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))
        self._mean = mean
        self._std = std
    def __len__(self):
        return len(self._labels)
    def __getitem__(self, idx):
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=self._mean, std=self._std)])
        return {
            'image': transform(self._images[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
    }
def dataLoaders(batch_size = 32):
    trainingSet = datasetASL('D:/Projects/MachineLearningASL/input/sign_mnist_train/sign_mnist_train.csv')
    train_loader = torch.utils.data.DataLoader(trainingSet, batch_size=batch_size, shuffle=True)
    testSet = datasetASL('D:/Projects/MachineLearningASL/input/sign_mnist_test/sign_mnist_test.csv')
    test_loader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_loader,test_loader

