# define and create EEG Class

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data

import random
import os
import matplotlib.pyplot as plt

class EEGData (Dataset):
    # constructor
    def __init__(self):
    
    # size
    def __len__(self):
        return self.length
    
    # get commands