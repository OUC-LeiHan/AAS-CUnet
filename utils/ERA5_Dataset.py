# -*- coding: utf-8 -*-
# @Author: ium
# @Date:   2019-12-05 15:08:38
# @Last Modified by:   ium
# @Last Modified time: 2020-01-12 11:00:38
# @Description : Make and read EC dataset

import numpy as np 
import random
import time
import torch
from natsort import natsorted
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
transforms = transforms.Compose([transforms.ToTensor()])


class ERA5_Dataset(Dataset):

    def __init__(self,Data_path,State,Size,
                 min_t2m,max_t2m,Transforms=None,
                 geo_path="/media/Space_ext4/Ckk/MyCode/EC_code/ST_3km/地形数据.npy"):
        '''
        :param Data_path: Path of training or test data
        :param State: "Train" or "Test"
        :param Size: Size of data
        :param min_t2m,max_t2m: Maximum and minimum values of data
        :param geo_path: Path of terrain data
        '''
        if State == "Train":
            self.Paths = natsorted(np.load(Data_path))
        else:
            self.Paths = natsorted(np.load(Data_path))

        #Read terrain data and normalize
        self.geo = np.load(geo_path)
        self.geo = (self.geo-np.min(self.geo))/(np.max(self.geo)-np.min(self.geo))

        if State == "Train":
            self.Total = len(self.Paths)*30
        else:
            self.Total = len(self.Paths)

        self.ele = "t2m"
        self.Data_Size = len(self.Paths)
        self.min_t2m = min_t2m
        self.max_t2m = max_t2m
        self.Transforms = Transforms
        self.State = State
        self.Global_Shape = [96,192]
        self.Size = Size
        self.Data = self.load_2Ram(self.Paths)

    #Load the data path, and then read the training data according to the data path
    def load_2Ram(self,Paths):
        Data_Ram = {"paths":[],
        "input":[],
        "label":[],
        "Ana":[]}
        start = time.time()

        '''
        paths:path of data at the current time
        input:Forecast data at the current time
        label:Actual data of the prediction results at the current time
        Ana:Actual data at the current time
        '''
        for path in Paths:
            data = np.load(path)
            Data_Ram["paths"].append(path)
            Data_Ram["input"].append(data["For_{}".format(self.ele)])
            Data_Ram["label"].append(data["Label_{}".format(self.ele)])
            Data_Ram["Ana"].append(data["Ana_{}".format(self.ele)])

        end = time.time()
        print("test_load_used:{}".format(end-start))
        return Data_Ram

    def __getitem__(self,index):
        img   = None
        label = None
        x_index = np.random.randint(0,self.Global_Shape[0]-self.Size[0])
        y_index = np.random.randint(0,self.Global_Shape[1]-self.Size[1])

        #Read data and complete random clipping
        if self.State == "Train":
            img = self.Data["input"][index%self.Data_Size][np.newaxis,x_index:x_index+self.Size[0],y_index:y_index+self.Size[1]]
            label = self.Data["label"][index%self.Data_Size][np.newaxis,x_index:x_index+self.Size[0],y_index:y_index+self.Size[1]]
            img1 = self.Data["Ana"][index%self.Data_Size][np.newaxis,x_index:x_index+self.Size[0],y_index:y_index+self.Size[1]]
            label = self.Data["label"][index%self.Data_Size][np.newaxis,x_index:x_index+self.Size[0],y_index:y_index+self.Size[1]]
            geo = self.geo[np.newaxis,x_index:x_index+self.Size[0],y_index:y_index+self.Size[1]]
            img = np.concatenate((img,img1),axis=0)
            geo = torch.from_numpy(geo).float()

            img   = torch.from_numpy(img).float()
            label = torch.from_numpy(label).float()
            return img,label,geo
        else:
            img   = self.Data["input"][index][np.newaxis,:,:]
            label = self.Data["label"][index][np.newaxis,:,:]
            img1 = self.Data["Ana"][index][np.newaxis,:,:]
            img = np.concatenate((img,img1),axis=0)
            #geo = self.geo[np.newaxis,0:10,0:10]
            geo = self.Data["paths"][index]
            img   = torch.from_numpy(img).float()
            label = torch.from_numpy(label).float()
            return img,label,geo.split("/")[-1][:-4]

    def __len__(self):
        return self.Total

