# -*- coding: utf-8 -*-

import numpy as np 
import random
import time
import torch
from natsort import natsorted
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
transforms = transforms.Compose([transforms.ToTensor()])


class Training_Dataset(Dataset):

    def __init__(self,Data_path,State,Size,
                 min_t2m,max_t2m,Transforms=None,
                 geo_path):
    
    #prepare your training dataset here.
    
    

