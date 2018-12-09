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

import scipy.misc

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=40, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--im_H', action='store', default=256, type=int, help='image height (default: 256)')
parser.add_argument('--im_W', action='store', default=256, type=int, help='image wigth (default: 256)')
parser.add_argument('--lambda_pixel', action='store', default=100.0, type=float, help='loss weight (default: 1.0)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--momentum', '--momentum', action='store', default=0.9, type=float, help='learning rate (default: 0.9)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--use_lsgan', action='store_true', default=False, help='Flag to use BCE loss (default: False)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--target", default='segmented', const='segmented',nargs='?', choices=['segmented', 'full'], help="Train with segmentation")
arg = parser.parse_args()

def main():

    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log/UNet'):
        os.makedirs('log/UNet')
    if not os.path.exists('log/UNet/gen_images'):
        os.makedirs('log/UNet/gen_images')
        
    Generator_path = 'model/UNet.pt'

    ####################
    ### Setup Logger ###
    ####################
    
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/UNet/logfile_UNet_.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    
    #################
    ### Transform ###
    #################
    
    input_transform = {
        'train': transforms.Compose([
            transforms.Resize((arg.im_H,arg.im_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((arg.im_H,arg.im_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    target_transform = {
        'train': transforms.Compose([
            transforms.Resize((arg.im_H,arg.im_W)),
            transforms.ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((arg.im_H,arg.im_W)),
            transforms.ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    #################
    ### Load Data ###
    #################
    
    root = '/data5/drone_machinelearning/amir/pix3d'
    back_path = root+'/background_split'
    img_path = root+'/img_split'
    syn_path = root+'/render_split'
    
    if arg.target =='segmented':
        train_foreground = root+'/foreground_split/train'
        test_foreground = root+'/foreground_split/test'
    elif arg.target == 'full':
        train_target_path = root+'/img_split/train'
        test_target_path = root+'/img_split/test'
    
    train_path = root+'/render_split/train'
    test_path = root+'/render_split/test'
    
    train_dataset = DRLoader(syn_path+'/train', train_foreground, in_transform=input_transform['train'] \
                                 ,target_transform=target_transform['train'])
    test_dataset = DRLoader(syn_path+'/test',test_foreground, in_transform=input_transform['test'] \
                                ,target_transform=target_transform['test'])

    dataLoader={}
    dataLoader['train'],dataLoader['test'] = [], []
    dataLoader['train'] = DataLoader(train_dataset, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    dataLoader['test'] = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    train_size, test_size = train_dataset.__len__(), test_dataset.__len__()
    
    ###################
    ### Load Models ###
    ###################
    
    Generator = UNet(input_nc=3)  
    if arg.useGPU_f:
        Generator.cuda()

    optimizer_G = optim.Adam(Generator.parameters(),lr=arg.lr)
    
    #####################
    ### Loss Function ###
    #####################
    
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    
    ################
    ### Training ###
    ################
    
    print("Start Training")
    logger.info("Start Training")
    epochs = arg.epochs if arg.train_f else 0
    
    min_accuracy = 0    
    for epoch in xrange(epochs):
        Generator.train()
        
        
        for batchIndex,(img,target) in enumerate(dataLoader['train']):
            if arg.useGPU_f:
                img,target = Variable(img.cuda(),requires_grad=True), Variable(target.cuda(),requires_grad=False)
            else:
                img,target = Variable(img,requires_grad=True),Variable(target,requires_grad=False)
            
            # ---------------
            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            gen_img = Generator(img)
            loss_G = criterion_L1(gen_img,target)
            loss_G.backward()
            optimizer_G.step()
            
            if batchIndex%10==0:
                logger.info('epoch:{}, index:{}, G_Loss:{}'.format(epoch,batchIndex,loss_G.data[0]))
                
        ##################
        ### Validation ###
        ##################
        
        print("Start Validation")
        logger.info("Start Validation")
        
        Generator.eval()
        loss_G = 0.0
        
        for batchIndex,(img,target) in enumerate(dataLoader['test']):
            if arg.useGPU_f:
                img,target = Variable(img.cuda(),requires_grad=True), Variable(target.cuda(),requires_grad=False)
            else:
                img,target = Variable(img,requires_grad=True),Variable(target,requires_grad=False)
            
            optimizer_G.zero_grad()
            
            gen_img = Generator(img)
            torch.save(Generator.state_dict(), Generator_path)            
            scipy.misc.imsave('log/UNet/gen_images/gen_img_'+str(epoch)+'_'+str(batchIndex)+'.jpg',np.transpose(np.squeeze(gen_img.data.cpu().numpy()),[1,2,0]))
            if batchIndex == 2:
                break
                
    if os.path.isfile(Generator_path):
        Generator.load_state_dict(torch.load(Generator_path, map_location=lambda storage, loc: storage))
    Generator.eval()
    loss_G = 0.0
        
    for batchIndex,(img,target) in enumerate(dataLoader['test']):
        if arg.useGPU_f:
            img,target = Variable(img.cuda(),requires_grad=True), Variable(target.cuda(),requires_grad=False)
        else:
            img,target = Variable(img,requires_grad=True),Variable(target,requires_grad=False)
            
        optimizer_G.zero_grad()
        
        gen_img = Generator(img)
        scipy.misc.imsave('log/UNet/gen_images/gen_img_test.jpg', np.transpose(np.squeeze(gen_img.data.cpu().numpy()),[1,2,0]))
        if batchIndex == 1:
            break


if __name__ == "__main__":
    main()