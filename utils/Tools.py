# -*- coding: utf-8 -*-
#Function of error calculation

import numpy 
import os

def caculate_Mse(data,label,batch_size):
    batch_Mse = np.square(data-label).sum()/batch_size
    one_Sample_Mse = np.square(data[0,:,:,:]-label[0,:,:,:])
    return batch_Mse,one_Sample_Mse

def mkdir_dir(dir_name):
    if os.path.exists(dir_name):
        pass
    else:
        os.mkdir(dir_name)  

def caculate_Mae(data,label,batch_size):
    batch_Mae = data - label

    return batch_Mae
