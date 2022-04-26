# -*- coding: utf-8 -*-
# @Description : Training and saving model

import sys
import os
from optparse import OptionParser
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from natsort import natsorted
from utils.Datasets import Load_Dataset
from utils.plot_fun import *
from utils.My_loss import  *
from model.CUnet import CUnet
from evaluate import evaluate_net
from natsort import natsorted
import time
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#Set random seeds to ensure that the experiment is reproducible.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

#def remove_pth(index):
#    root_dir = "./checkpoints"
#    pths = natsorted(os.listdir(root_dir))
#    for i,pth in enumerate(pths):
#        pth_path = os.path.join(root_dir,pth)
#        if i == index: 
#            pass
#        else:
#            os.remove(pth_path)

#training code
def train_net(net,epochs=5,batch_size=1,
              lr=0.1,save_cp=True,gpu=False,
              input_channel=1,output_channel=1,path_For_Train=None,
              path_For_Test=None,test_interval=2500,
              dir_checkpoint="./checkpoints",dir_result="./results",
              plot_Sample=40,w=48,h=48):
    '''
    :param net: Name of model
    :param epochs: Number of training iterations
    :param batch_size: The number of data samples used in a training process
    :param lr: Learning rate
    :param save_cp: Whether to save the intermediate model
    :param gpu: Whether to use GPU
    :param input_channel: Number of input data channels
    :param output_channel: Number of  output data channels
    :param path_For_Train: Training data path
    :param path_For_Test: Test data path
    :param test_interval: Test time node
    :param dir_checkpoint: the dir to save the checkpoints
    :param dir_result: the dir to save the results
    :param plot_Sample:the number of test Samples to be plotted
    :param w,h: Size of data
    :return:
    '''
    train_paths = natsorted(np.load(path_For_Train))
    test_paths  = natsorted(np.load(path_For_Test))
    Grid_NUM    = w*h
    N_train     = len(train_paths)
    test_Loss_List = []
    train_Loss_List= []
    test_mae_List  = []

    print('''Starting training:
        Epoch:{}
        Batch size：{}
        Learning rate:{}
        Training size:{}
        Validation size:{}
        Checkpoints:{}
        CUDA:{}
        input_channel:{}
        output_channel:{}
        path_For_Train:{}
        path_For_Test:{}
        test_interval:{}
    '''.format(epochs, batch_size, lr,
               len(train_paths), len(test_paths),
               str(save_cp), str(gpu), input_channel,
               output_channel, path_For_Train,
               path_For_Test, test_interval))

    #Network optimizer and loss function
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=0.001)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.3,patience=2,verbose=True,threshold=0.00001,threshold_mode="rel",cooldown=0,min_lr=0.000001,eps=1e-08)

    #loading Training data and test data
    train_Data = Load_Dataset(path_For_Train, "Train", [48,48], min_t2m=0, max_t2m=1)
    test_Data = Load_Dataset(path_For_Test, "Test", [48,48], min_t2m=0, max_t2m=1)
    train_Data_Loader = DataLoader(train_Data, batch_size=batch_size, shuffle=True,pin_memory=True)
    test_Data_Loader = DataLoader(test_Data, batch_size=32, shuffle=False,pin_memory=True)
    lr_list = []

    for epoch in range(epochs):
        print("Starting epoch {}/{}.".format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        epoch_index = 0

        for i, (imgs, label,geo) in enumerate(train_Data_Loader):
            if gpu:
                imgs = imgs.cuda()
                label = label.cuda()
                geo = geo.cuda()

            img_pred = net(imgs)
            loss = criterion(img_pred,label)
            epoch_loss += loss.item()
            print("{0:.4f} ---- loss: {1:.6f}".format(i * batch_size / N_train, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_index = i+1
            print(epoch_index)
            
            #Test the effect of the model
            test_loss, test_mae = evaluate_net(net=net, dataset=test_Data_Loader, iter_index=epoch,
                                                  Scale_Grid=Grid_NUM, min_t2m=0, max_t2m=1)
        if (save_cp):
            torch.save(net.state_dict(),os.path.join(dir_checkpoint,"CP_{}.pth".format(epoch)))
            print("Checkpoint {} saved".format(epoch))

        #plot error curve
        test_Loss_List.append(test_loss)
        train_Loss_List.append(epoch_loss)
        test_mae_List.append(test_mae)
        scheduler.step(test_loss)
        lr_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
        plot_Loss(test_Loss_List, "./log/test_loss.eps", "test_loss")
        plot_Loss(train_Loss_List, "./log/train_loss.eps", "train_loss")
        plot_Loss(test_mae_List, "./log/test_mae.eps", "test_mae")
        plot_Loss(lr_list, "./log/lr.eps", "lr")
        np.save("./log/lr.npy",lr_list)

    #min_index = np.argmin(test_Loss_List)
    #remove_pth(min_index)

#配置命令行参数解析，可使用命令行修改训练参数
def get_args():
    parser = OptionParser()
    parser.add_option("-e", "--epoch", dest="epochs", default=50, type="int",
                      help="number of epoches")
    parser.add_option("-b", "--batch-size", dest="batchsize", default=32, type="int",
                      help="batch size")
    parser.add_option("-l", "--learning-rate", dest="lr", default=0.01, type="float",
                      help="learning rate")
    parser.add_option("-g", "--gpu", action="store_true", dest="gpu", default=True,
                      help="use cuda")
    parser.add_option("-c", "--load", dest="load", default="",
                      help="load file model")
    parser.add_option("-i", "--test_interval", dest="test_interval", default=100,
                      help="test interval")
    parser.add_option("-t", "--path_train", dest="path_For_Train", default="/media/Space_ext4/Ckk/MyCode/EC_code/ST_3km/大论文/24h/目录/EC_ERA5_24h所有训练数据目录.npy",
                      help="the address of paths of training Samples")
    parser.add_option("-v", "--path_test", dest="path_For_Test", default="/media/Space_ext4/Ckk/MyCode/EC_code/ST_3km/大论文/24h/目录/EC_ERA5_24h_48*48所有验证数据目录.npy",
                      help="the address of paths of test Samples")
    parser.add_option("-w", "--input_channels", dest="input_channels", default=1,
                      help="the number of channels of input Data")
    parser.add_option("-q", "--output_channels", dest="output_channels", default=1,
                      help="the number of channels of output Data")
    parser.add_option("-n", "--plot_Sample", dest="plot_Sample", default=10,
                      help="the number of test Samples to be plotted")
    parser.add_option("-k", "--dir_checkpoint", dest="dir_checkpoint", default="./checkpoints",
                      help="the dir to save the checkpoints")
    parser.add_option("-x", "--dir_results", dest="dir_results", default="./results",
                      help="the dir to save the results")

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    start = time.time()
    args = get_args()
    net = CUnet(n_channels=2, n_classes=1)

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True
    if args.load:
        models = natsorted(os.listdir(args.load))
        best_model = os.path.join(args.load, models[-1])
        net.load_state_dict(torch.load(best_model))
        print("Model loaded form {}".format(best_model))

    try:
        train_net(net=net,epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,gpu=args.gpu,
                  input_channel=args.input_channels,
                  output_channel=args.output_channels,
                  path_For_Train=args.path_For_Train,
                  path_For_Test=args.path_For_Test,
                  test_interval=args.test_interval,
                  plot_Sample=args.plot_Sample)
        end = time.time()
        print("Train_Finished!,time_Used:{}".format(end - start))
        f = open("./done.txt", "wb")
        f.close()

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        print("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

