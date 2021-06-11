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
from sklearn.metrics import accuracy_score
from pprint import pprint

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_transform = A.Compose(
                            [
                                A.Resize(100, 100),
                                # A.RGBShift(),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5), 
                                # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                                # A.PadIfNeeded(min_height=100, min_width=100, always_apply=True, border_mode=0),
                                # A.IAAAdditiveGaussianNoise(p=0.1),
                                # A.IAAPerspective(p=0.1),
                                # A.RandomBrightnessContrast(p=0.1),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2(),
                            ])
    dataset_path = 'dataset/img_align_celeba'
    dataset = SiameseDataset(dataset_path=dataset_path, 
                            img_transform=img_transform)
    trainloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # net = SiameseNetwork()
    net = torch.load('models/final_loss_1.0145.pth').to(device)
    net.eval()
    criterion = ContrastiveLoss()
    criterion.to(device)

    counter = []
    loss_history = [] 
    iteration_number= 0     
    print('start')
    losses = []
    labels = []
    distances = []
    for epoch in range(0, 1):
        for i, data in enumerate(trainloader,0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            losses.append(loss_contrastive.item())
            for lab in label.tolist(): 
                labels += lab
            euclidean_distance = F.pairwise_distance(output1, output2)
            distances += euclidean_distance.tolist()
    mean_loss = sum(losses)/len(losses)

    print('mean_loss: ', mean_loss)
    scores = {}
    y_true = [ int(i) for i in labels]
    for treshold in range(40, 110, 10):
        treshold = float(treshold/100.0)
        y_pred = []
        for distance in distances:
            if distance < treshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        scores.update({treshold: accuracy_score(y_true, y_pred)})
    pprint(scores)
        
# dla 'dataset/test_faces' 0.5
# dla 'dataset/img_align_celeba' 0.6


if __name__ == '__main__':
    test()