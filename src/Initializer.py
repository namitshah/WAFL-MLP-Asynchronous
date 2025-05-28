import json
import os
import socket, threading, time, pickle, traceback
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net 
from torch.utils.data.dataset import Subset

import socket, threading, time
n_device = 4
batch_size = 32
# Setting up Train DataSet Loader
trainset = torchvision.datasets.MNIST(
    root="../data/MNIST/",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
indices = torch.load("../data/noniid_filter/filter_r90_s01.pt")
subset = [Subset(trainset, indices[i]) for i in range(n_device)]
trainloader = [
    torch.utils.data.DataLoader(
        subset[i], batch_size=batch_size, shuffle=False, num_workers=1
    )
    for i in range(n_device)
]
with open("../data/pickles/trainloader.pickled", "wb") as file:
    pickle.dump(trainloader, file)