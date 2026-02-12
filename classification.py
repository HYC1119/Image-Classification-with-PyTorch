import os
import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable 


# ====== Set Hyperparameters ======
DOWNLOAD_DATASET = True
LR = 0.001
BATCH_SIZE = 10
MODELS_PATH = './models'


# ======  Transform datasets to tensors of nornmalized range [-1, 1] ======
transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), 
                          (0.5, 0.5, 0.5))])


# ====== Load CIFAR10 Dataset ======
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

trainloader = Data.DataLoader(
    trainset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root='./data',                                    
    train=False, 
    download=True, 
    transform=transform
)

testloader = Data.DataLoader(
    testset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer',  'dog', 'frog', 'ship', 'truck')

 
