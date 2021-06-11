"""
source: 
https://github.com/cskarthik7/One-Shot-Learning-PyTorch

"""
import os
import cv2
from PIL import Image
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from dataset import SiameseDataset
from model import SiameseNetwork
from loss import ContrastiveLoss


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def save_model(model, loss=0.0, mode='iter'):
    model_name = mode+'_'+'loss_' + str(round(loss, 4)) + '.pth'
    model_save_path = os.path.join('models', model_name)
    torch.save(model, model_save_path)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_transform = A.Compose(
                            [
                                A.Resize(100, 100),
                                A.RGBShift(),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5), 
                                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                                A.PadIfNeeded(min_height=100, min_width=100, always_apply=True, border_mode=0),
                                A.IAAAdditiveGaussianNoise(p=0.1),
                                A.IAAPerspective(p=0.1),
                                A.RandomBrightnessContrast(p=0.1),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2(),
                            ])
    dataset_path = 'dataset/img_align_celeba'
    dataset = SiameseDataset(dataset_path=dataset_path, 
                            img_transform=img_transform)
    trainloader = DataLoader(dataset, batch_size=512, shuffle=True)

    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    criterion.to(device)
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = [] 
    iteration_number= 0     
    for epoch in range(0, 100):
        for i, data in enumerate(trainloader,0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                cpu_loss = loss_contrastive.item()
                loss_history.append(cpu_loss)
                save_model(net, cpu_loss, 'iter_'+str(epoch))
    
    save_model(net, loss_contrastive.item(), 'final')

    show_plot(counter,loss_history)


if __name__ == '__main__':
    train()