# -*- coding: utf-8 -*-
# @Description : Make and read EC dataset

import numpy as np 
import random
import time
import torch
from natsort import natsorted
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
transforms = transforms.Compose([transforms.ToTensor()])


class Load_Dataset(Dataset):

    def __init__(self,Data_path,State,min_t2m,max_t2m):
        
        '''
        prepare your own data here, you can use this framework or set it yourself.
        
        '''


    def __getitem__(self,index):
        
       '''
       Read data iteratively according to index
       '''

    def __len__(self):
        
        '''
        return the number of total data
        '''

