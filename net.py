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
from DRLoader import DRLoader, Syn2Real

import scipy.misc

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=40, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--im_H', action='store', default=256, type=int, help='image height (default: 256)')
parser.add_argument('--im_W', action='store', default=256, type=int, help='image wigth (default: 256)')
parser.add_argument('--lambda_pixel', action='store', default=100.0, type=float, help='pixed loss weight (default: 100.0)')
parser.add_argument('--lr_G', '--generator-learning-rate', action='store', default=0.001, type=float, help='generator learning rate (default: 0.001)')
parser.add_argument('--lr_D', '--discriminator-learning-rate', action='store', default=0.001, type=float, help='discriminator learning rate (default: 0.001)')
parser.add_argument('--momentum', '--momentum', action='store', default=0.9, type=float, help='learning rate (default: 0.9)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--GAN_Loss", default='MSE', const='MSE',nargs='?', choices=['MSE', 'BCE'], help="GAN Loss (default:MSE)")
parser.add_argument("--D_net", default='NLayer', const='Nlayer',nargs='?', choices=['Nlayer', 'Pix'], help="Discriminator  model(default:Nlayer)")
parser.add_argument("--target", default='segmented', const='segmented',nargs='?', choices=['segmented', 'full'], help="Train with segmentation")
parser.add_argument("--Loader", default='DRLoader', const='DRLoader',nargs='?', choices=['DRLoader', 'Syn2Real'], help="load synth image with/without real background")
arg = parser.parse_args()

def main():
    
    if arg.useGPU_f:
        torch.cuda.set_device(arg.gpu_num)
        torch.cuda.current_device()
    
    if arg.Loader=='Syn2Real':
        log_root = 'log/'+arg.Loader+'_'+arg.D_net+'_'+arg.GAN_Loss
    elif arg.Loader == 'DRLoader':
        log_root = 'log/'+arg.Loader+'_'+arg.D_net+'_'+arg.target

    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if not os.path.exists(log_root+'/gen_images'):
        os.makedirs(log_root+'/gen_images')
    if not os.path.exists(log_root+'/model'):
        os.makedirs(log_root+'/model')

    ####################
    ### Setup Logger ###
    ####################
    
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler(log_root+'/logfile.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Generator Learning Rate: {}".format(arg.lr_G))
    logger.info("Discriminator Learning Rate: {}".format(arg.lr_D))
    logger.info("Discriminator: "+ arg.D_net)
    logger.info("Loader:" + arg.Loader)
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
    
    if arg.Loader == 'DRLoader':
        train_dataset = DRLoader(syn_path+'/train', train_foreground, in_transform=input_transform['train'] \
                                 ,target_transform=target_transform['train'])
        test_dataset = DRLoader(syn_path+'/test',test_foreground, in_transform=input_transform['test'] \
                                ,target_transform=target_transform['test'])
    elif arg.Loader == 'Syn2Real':
        train_dataset = Syn2Real(back_path+'/train',img_path+'/train', syn_path+'/train', in_transform=input_transform['train'] \
                                 ,target_transform=target_transform['train'])
        test_dataset = Syn2Real(back_path+'/train',img_path+'/train', syn_path+'/train', in_transform=input_transform['test'] \
                                ,target_transform=target_transform['test'])

    dataLoader={}
    dataLoader['train'],dataLoader['test'] = [], []
    dataLoader['train'] = DataLoader(train_dataset, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    dataLoader['test'] = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    train_size, test_size = train_dataset.__len__(), test_dataset.__len__()
    
    #####################
    ### Loss Function ###
    #####################
    
    criterion_L1 = nn.L1Loss()
    sigmoid_f=False
    if arg.GAN_Loss == 'BCE':
        criterion_GAN = nn.BCEWithLogitsLoss()
        sigmoid=True
    elif arg.GAN_Loss == 'MSE':
        criterion_GAN = nn.MSELoss()
    
    ###################
    ### Load Models ###
    ###################
    
    Generator = UNet(input_nc=3)
    if arg.D_net == 'NLayer':
        Discriminator = NLayerDiscriminator(input_nc=3,use_sigmoid=sigmoid_f)
    elif arg.D_net == 'Pix':
        Discriminator = PixelDiscriminator(input_nc=3,use_sigmoid=sigmoid_f)
    
    if arg.useGPU_f:
        Discriminator.cuda()
        Generator.cuda()

    optimizer_G = optim.Adam(Generator.parameters(),lr=arg.lr_G)
    optimizer_D = optim.Adam(Discriminator.parameters(), lr = arg.lr_D)
    
    ################
    ### Training ###
    ################
    
    print("Start Training")
    logger.info("Start Training")
    epochs = arg.epochs if arg.train_f else 0
    
    min_accuracy = 0
    correct_label, correct_pose , ave_loss = 0, 0, 0
    
    for epoch in xrange(epochs):
        Discriminator_path = log_root+'/model/'+arg.D_net+'_'+str(epoch)+'.pt'
        Generator_path = log_root+'/model/Generator_'+str(epoch)+'.pt'
        
        Generator.train()
        Discriminator.train()
        
        
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
            pred_fake = Discriminator(img)
            
            patch = pred_fake.shape
            v = torch.rand(*patch)*0.1
            f = 1-v
            if arg.useGPU_f:
                valid, fake = Variable(v.cuda(), requires_grad=False), Variable(f.cuda(), requires_grad=False)
            else:
                valid, fake = Variable(v, requires_grad=False), Variable(f, requires_grad=False)            
            
            loss_GAN = criterion_GAN(pred_fake,valid)
            if gen_img.data.shape[1]!=target.data.shape[1]:
                print('L1 ERROR')
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
                torch.save(Generator.state_dict(), Generator_path)
                torch.save(Discriminator.state_dict(), Discriminator_path)
        ##################
        ### Validation ###
        ##################
        
        print("Start Validation")
        logger.info("Start Validation")
        
        Generator.eval()
        Discriminator.eval()
        loss_G, loss_D = 0.0, 0.0
        
        for batchIndex,(img,target) in enumerate(dataLoader['test']):
            if arg.useGPU_f:
                img,target = Variable(img.cuda(),requires_grad=True), Variable(target.cuda(),requires_grad=False)
            else:
                img,target = Variable(img,requires_grad=True),Variable(target,requires_grad=False)
            
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            
            gen_img = Generator(img)
            
            scipy.misc.imsave(log_root+'/gen_images/gen_img_'+str(epoch)+'_'+str(batchIndex)+'.jpg',\
                              np.transpose(np.squeeze(gen_img.data.cpu().numpy()),[1,2,0]))
            break
                
    best_model = 10
    Generator_path = log_root+'/model/Generator_'+str(best_model)+'.pt'
    if os.path.isfile(Generator_path):
        Generator.load_state_dict(torch.load(Generator_path, map_location=lambda storage, loc: storage))
    Generator.eval()
    Discriminator.eval()
    loss_G, loss_D = 0.0, 0.0
        
    for batchIndex,(img,target) in enumerate(dataLoader['test']):
        if arg.useGPU_f:
            img,target = Variable(img.cuda(),requires_grad=True), Variable(target.cuda(),requires_grad=False)
        else:
            img,target = Variable(img,requires_grad=True),Variable(target,requires_grad=False)
            
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        
        gen_img = Generator(img)
        scipy.misc.imsave(log_root+'/gen_images/gen_img_test.jpg', np.transpose(np.squeeze(gen_img.data.cpu().numpy()),[1,2,0]))
        if batchIndex == 1:
            break


if __name__ == "__main__":
    main()