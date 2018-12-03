##### NOTE:
##### FOR THE DISCRIMINATOR, MAYBE WE SHOULD PROVIDE PAIRS OF (INPUT,PRED) OR (INPUT,REAL) AS INPUT

from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import torchvision.models as models

import os
import numpy as np
import requests
import argparse
import logging
import time
import pickle
import copy

import torch.backends.cudnn as cudnn
import torch.cuda as cuda

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from Arch import *
from DRLoader import DRLoader

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=40, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--im_H', action='store', default=256, type=int, help='image height (default: 256)')
parser.add_argument('--im_W', action='store', default=256, type=int, help='image wigth (default: 256)')
parser.add_argument('--lambda_pixel', action='store', default=100.0, type=float, help='loss weight (default: 1.0)')
parser.add_argument('--lr_G', '--generator-learning-rate', action='store', default=0.0001, type=float, help='generator learning rate (default: 0.01)')
parser.add_argument('--lr_D', '--discriminator-learning-rate', action='store', default=0.0001, type=float, help='discriminator learning rate (default: 0.01)')
parser.add_argument('--momentum', '--momentum', action='store', default=0.9, type=float, help='learning rate (default: 0.9)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--use_lsgan', action='store_true', default=False, help='Flag to use BCE loss (default: False)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--D_net", default='NLayer', const='Nlayer',nargs='?', choices=['Nlayer', 'Pix'], help="Discriminator  model(default:Nlayer)")
parser.add_argument("--target", default='segmented', const='segmented',nargs='?', choices=['segmented', 'full'], help="Train with segmentation")
arg = parser.parse_args()

def main():

    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
    
    model_path = 'model/'+arg.D_net+'_'+arg.target+'.pt'

    ####################
    ### Setup Logger ###
    ####################
    
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_'+arg.D_net+'_.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Generator Learning Rate: {}".format(arg.lr_G))
    logger.info("Discriminator Learning Rate: {}".format(arg.lr_D))
    logger.info("Discriminator: "+ arg.D_net)
    logger.info("target images:" + arg.target)
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    
    #################
    ### Transform ###
    #################
    
    input_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(arg.im_H, scale=(0.6,1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(arg.im_H),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    target_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(arg.im_H, scale=(0.6,1.0)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(arg.im_H),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    #################
    ### Load Data ###
    #################
    
    root = '/data5/drone_machinelearning/amir/pix3d'
    if arg.target =='segmented':
        train_target_path = root+'/foreground_split/train'
        test_target_path = root+'/foreground_split/test'
    elif arg.target == 'full':
        train_target_path = root+'/img_split/train'
        test_target_path = root+'/img_split/test'
    
    train_path = root+'/render_split/train'
    test_path = root+'/render_split/test'
    
    train_dataset = DRLoader(train_path, train_target_path, in_transform=input_transform['train'] \
                             ,target_transform=target_transform['train'])
    test_dataset = DRLoader(test_path,test_target_path, in_transform=input_transform['test'] \
                            ,target_transform=target_transform['test'])
    
    dataLoader={}
    dataLoader['train'],dataLoader['test'] = [], []
    dataLoader['train'] = DataLoader(train_dataset, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    dataLoader['test'] = DataLoader(test_dataset, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    
    train_size, test_size = train_dataset.__len__(), test_dataset.__len__()
    
    ###################
    ### Load Models ###
    ###################
    
    Generator = UNet(input_nc=1)
    if arg.D_net == 'NLayer':
        Discriminator = NLayerDiscriminator(input_nc=3)
    elif arg.D_net == 'Pix':
        Discriminator = PixelDiscriminator(input_nc=3)
    
    if arg.useGPU_f:
        Discriminator.cuda()
        Generator.cuda()

    optimizer_G = optim.Adam(Generator.parameters(),lr=arg.lr_G)
    optimizer_D = optim.Adam(Discriminator.parameters(), lr = arg.lr_D)
    
    #####################
    ### Loss Function ###
    #####################
    
    if arg.use_lsgan:
        criterion_GAN = nn.BCELoss()
    else:
        criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    
    ################
    ### Training ###
    ################
    
    if arg.useGPU_f:
        temp = Variable(torch.Tensor(arg.batchSize,3,arg.im_H,arg.im_W).fill_(0.1).cuda(),requires_grad=False)
    else:
        temp = Variable(torch.Tensor(arg.batchSize,3,arg.im_H,arg.im_W).fill_(0.1),requires_grad=False)
    temp = Discriminator(temp)
    patch = temp.data.shape
    v = torch.rand(*patch)*0.1
    f = 1-v
    
    print("Start Training")
    logger.info("Start Training")
    epochs = arg.epochs if arg.train_f else 0
    
    min_accuracy = 0
    correct_label, correct_pose , ave_loss = 0, 0, 0
    
    for epoch in xrange(epochs):
        Generator.train()
        Discriminator.train()
        
        for batchIndex,x in enumerate(dataLoader['train']):
            img,target,label,class_ = x
            img,target=img[:,0:3],target[:,0]
            if arg.useGPU_f:
                valid, fake = Variable(v.cuda(), requires_grad=False), Variable(f.cuda(), requires_grad=False)
                img,target, = Variable(img.cuda(),requires_grad=True), Variable(target.cuda(),requires_grad=False)
            else:
                valid, fake = Variable(v, requires_grad=False), Variable(f, requires_grad=False)
                img,target = Variable(img,requires_grad=True),Variable(target,requires_grad=False)
            
            # ---------------
            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            
            gen_img = Generator(img)
            pred_fake = Discriminator(img)
            loss_GAN = criterion_GAN(pred_fake,valid)
            if gen_img.data.shape[1]!=target.data.shape[1]:
                print('L1 ERROR: ',batchIndex,label,class_)
            loss_Pix = criterion_L1(gen_img,target)
            
            loss_G = loss_GAN + loss_Pix*arg.lambda_pixel
            loss_G.backward()
            optimizer_G.step()
            
            # -------------------
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            
            pred_real = Discriminator(target)
            loss_real = criterion_GAN(pred_real,valid)
            
            pred_fake = Discriminator(gen_img.detach())
            loss_fake = criterion_GAN(pred_fake,fake)
            
            loss_D = 0.5*(loss_real+loss_fake)
            loss_D.backward()
            optimizer_D.step()
            
            if batchIndex%10==0:
                logger.info('epoch:{}, index:{}, G_Loss:{}, D_Loss:{}'.format(epoch,batchIndex,loss_G.data[0],loss_D.data[0]))
                
        ##################
        ### Validation ###
        ##################
        
        print("Start Validation")
        logger.info("Start Validation")
        
        Generator.eval()
        Discriminator.eval()
        loss_G, loss_D = 0.0, 0.0
        
        for batchIndex,(img,target,label) in enumerate(dataLoader['test']):
            img=img[:,0:3]
            if arg.useGPU_f:
                valid = Variable(valid.cuda(), requires_grad=False)
                fake = Variable(fake.cuda(), requires_grad=False)
                
                img,target,label = Variable(img.cuda(),requires_grad=True), Variable(target.cuda(),requires_grad=False), \
                Variable(label.cuda(),requires_grad=False)
            else:
                valid = Variable(valid, requires_grad=False)
                fake = Variable(fake, requires_grad=False)

                img,target,label = Variable(img,requires_grad=True),Variable(target,requires_grad=False), \
                Variable(label,requires_grad=False)
            
            optimizer_G.zero_grad()
            
            gen_img = Generator(img)
            pred_fake = Discriminator(img)
            loss_GAN = criterion_GAN(pred_fake,vaild)
            loss_Pix = criterion_L1(gen_img,target)
            
            loss_G += loss_GAN + arg.alpha*loss_Pix

            optimizer_D.zero_grad()
            
            pred_real = Discriminator(target)
            loss_real = criterion_GAN(pred_real,valid)
            
            pred_fake = Discriminator(gen_img.detach())
            loss_fake = criterion_GAN(pred_fake,fake)
            
            loss_D += 0.5*(loss_real+loss_fake)
            
        logger.info('Validation: G_Loss:{}, D_Loss:{}'.format(loss_G.data[0]/test_size,loss_D.data[0]/test_size))
        torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    main()