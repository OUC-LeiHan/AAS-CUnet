# -*- coding: utf-8 -*-
# @Description : Construction of overall model of Cu-net network

from .CUnet_model_parts import *

class CUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        '''
        :param n_channels: Number of input data channels
        :param n_classes: Number of output data channels
        '''
        super(CUnet, self).__init__()
        #Build encoding module
        self.inc = inconv(n_channels, 64) #48*48
        self.down1 = down(64, 128) #24*24
        self.down2 = down(128, 256) #12*12
        self.down3 = down(256, 512) #6*6
        self.down4 = down(512, 512) #3*3
        # Build decoding module
        self.up1 = up(1024,256,512) #6*6
        self.up2 = up(512, 128,256) #12*12
        self.up3 = up(256, 64,128)  #24*24
        self.up4 = up(128, 64,64)   #48*48
        self.outc = outconv(64, n_classes)

    #Forward propagation of model
    def forward(self, x_raw):

        x1 = self.inc(x_raw)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #print(x_raw[:,0:1,:,:].size())
        return x+x_raw[:,0:1,:,:]

    #Initialize model parameters
    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)

#Set the initialization method of network weight
def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

