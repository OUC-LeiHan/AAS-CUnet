# -*- coding: utf-8 -*-
# @Author: Taurus_Moon
# @Date:   2019-04-15 21:16:53
# @Last Modified by:   Taurus_Moon
# @Last Modified time: 2020-01-12 09:50:01
# @Version : 1.0
# @Description : 这是定义了一些做测试的函数

import numpy as np 
import os
#from utils.plot_fun import *
import time
from utils.log_txt import *
from utils.Tools import *
import datetime

with open("./log/score.txt","w") as f:
	f.write("{0}\t{1:^8}\t{2:^8}\t{3:^8}\t{4:^8}\t{5:^8}\t{6:^8}\t{7:^8}\n".format("iter","training_loss","test_loss","EC_mse","cor_Mae","EC_Mae","Time_Used","Time_now"))

# def eval_net_base(net,dataset,gpu=False):
# 	net.eval()
# 	for i,imgs,label,geo in enumerate(dataset):
# 		if gpu:
# 			imgs = imgs.cuda()
# 			label = label.cuda()
# 		imgs_Correct = net(imgs)


def eval_net_EC(net,dataset,iter_index,
				Scale_Grid=0,min_t2m=-80,
				max_t2m=60,element="T2M",gpu=True):
    net.eval()
	cor_Mse_ave = 0
	EC_Mse_ave  = 0
	cor_Mae_ave = 0
	EC_Mae_ave  = 0
	test_Size = 0
	start = time.time()
	Scale = Scale_Grid

	for i,(imgs,label,geo) in enumerate(dataset):
		test_Size += imgs.size()[0]
		if gpu:
			imgs = imgs.cuda()
			label = label.cuda()

		img_Cor = net(imgs)
		#imgs_Cpu = imgs[:,0,:,:].cpu().detach().numpy()[:,np.newaxis,:,:]*(max_t2m-min_t2m)+min_t2m
		label_Cpu = label.cpu().detach().numpy()*(max_t2m-min_t2m)+min_t2m
		img_Cor_Cpu = img_Cor.cpu().detach().numpy()*(max_t2m-min_t2m)+min_t2m
		#mgs_Cpu = imgs_Cpu
		#label_Cpu = label*(max_t2m-min_t2m)+min_t2m
		#img_Cor_Cpu = img_Cor_Cpu*(max_t2m-min_t2m)+min_t2m

		#np.clip(img_Cor_Cpu,0,100)
		cor_Mae_Batch = img_Cor_Cpu -label_Cpu
		EC_Mae_Batch  = imgs_Cpu - label_Cpu

		cor_Mse_Batch = np.square(cor_Mae_Batch)
		EC_Mse_Batch = np.square(EC_Mae_Batch)

		cor_Mse_ave += cor_Mse_Batch.sum()
		EC_Mse_ave	+= EC_Mse_Batch.sum()
		cor_Mae_ave += np.abs(cor_Mae_Batch).sum()
		EC_Mae_ave  += np.abs(EC_Mae_Batch).sum()

	cor_Mse_ave = cor_Mse_ave/(test_Size*Scale)	
	EC_Mse_ave  = EC_Mse_ave/(test_Size*Scale)
	cor_Mae_ave = cor_Mae_ave/(test_Size*Scale)
	EC_Mae_ave  = EC_Mae_ave/(test_Size*Scale)

	end = time.time()
	return np.sqrt(cor_Mse_ave),cor_Mae_ave
















