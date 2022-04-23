# -*- coding: utf-8 -*-
'''
Time: 2020 7 7
Version:1.01
Description: structure of Unet
'''
from model_parts import *
import torch
from torch import optim


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        #进行下采样
        self.inc = inconv(n_channels, 16)  
        self.down1 = down(16, 32)  
        self.down2 = down(32, 64)  
        self.down3 = down(64, 128)  
        self.down4 = down(128, 128)  

        #进行上采样
        self.up1 = up(256, 64, 128)  
        self.up2 = up(128, 32, 64)  
        self.up3 = up(64, 16, 32)  
        self.up4 = up(32, 16, 16)  
        self.outc = outconv(16, n_classes)

    ##网络前向传播
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

        return x

    ###网络参数初始化，此处使用凯明分布
    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)  # 利用凯明均匀分布来进行初始化
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

if __name__ == '__main__':

    UNet_model = UNet(3, 1)  # 输入数据的channel为3，输出数据的channel为1
    optimizer = optim.Adam(UNet_model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_func = nn.MSELoss(reduction='sum')
    train_loss = 0

    #进行100次模拟训练
    for i in range(100):
        ##模拟训练数据
        img = torch.rand(2, 3, 600, 600)
        ##模拟真实值
        label = torch.rand(2, 1, 600, 600)
        #模型前向传播计算
        img_pred = UNet_model(img)
        loss = loss_func(label, img_pred)

        ##优化器反向传播操作
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss

        #每训练10次便记录一次误差值
        if i % 10 == 0:
            print("iter:{},loss:{}".format(i, train_loss / 10))
            train_loss = 0


