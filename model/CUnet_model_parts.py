# -*- coding: utf-8 -*-
# @Description : Build the structure of the model part

import torch
import torch.nn as nn 
import torch.nn.functional as F

#Basic convolution module for encoding and decoding
class double_conv(nn.Module):
	"""(conv=>BN=>ReLu)*2"""
	def __init__(self,in_ch,out_ch):
		super(double_conv,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch,out_ch,3,padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch,out_ch,3,padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU()	
			)

	def forward(self,x):
		x = self.conv(x)
		return x

#Network input convolution module
class inconv(nn.Module):
	def __init__(self,in_ch,out_ch):
		super(inconv,self).__init__()
		self.conv = double_conv(in_ch,out_ch)

	def forward(self,x):
		x = self.conv(x)
		return x

#Network output convolution module
class outconv(nn.Module):
	def __init__(self,in_ch,out_ch):
		super(outconv,self).__init__()
		self.conv = change_channels(in_ch,out_ch)

	def forward(self,x):
		x = self.conv(x)
		return x

#down-sampling module in encoder
class down(nn.Module):
	def __init__(self,in_ch,out_ch):
		super(down,self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool2d(2),
			double_conv(in_ch,out_ch)
			)

	def forward(self,x):
		x = self.mpconv(x)
		return x

#up-sampling module in decoder
#include Linear interpolation、Deconvolution and sub_pixel
class up(nn.Module):
	def __init__(self,in_ch,out_ch,in_ch_x1,mode="sub_pixel",r=2):
		super(up,self).__init__()
		if mode == "bilinear":
			self.up = nn.Upsamples(scale_factor=2,mode="bilinear",align_corners=True)
		elif mode == "ConvTranspose2d":
			self.up = nn.ConvTranspose2d(in_ch//2,in_ch//2,2,stride=2)
		elif mode == "sub_pixel":
			tmp_out_ch = in_ch_x1*r*r
			self.up = nn.Sequential(
				change_channels(in_ch_x1,tmp_out_ch),
				nn.PixelShuffle(r))

		self.conv = double_conv(in_ch,out_ch)

	def forward(self,x1,x2):
		x1 = self.up(x1)

		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1,(diffX//2,diffX-diffX//2,
					   diffY//2,diffY-diffY//2))

		x = torch.cat([x2,x1],dim=1)
		x = self.conv(x)

		return x

#Module used to change channel
class change_channels(nn.Module):
	def __init__(self,in_ch,out_ch):
		super(change_channels,self).__init__()
		self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=1)

	def forward(self,x):
		return self.conv(x)







