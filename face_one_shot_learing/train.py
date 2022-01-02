"""
source: 
https://github.com/cskarthik7/One-Shot-Learning-PyTorch

"""
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from dataset_multiplefaces import DatasetMultipleFaces
from model import Siamese


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def save_model(model, loss=0.0, mode='iter'):
    model_name = mode+'_'+'loss_' + str(round(loss, 4)) + '.pth'
    model_save_path = os.path.join('models', model_name)
    torch.save(model, model_save_path)

def train():
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_transform = A.Compose(
                            [
                                A.Resize(100, 100),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5), 
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2(),
                            ])
    dataset_path = 'J:/yt_image_dataset_maker/face_dataset'
    dataset = DatasetMultipleFaces(dataset_path=dataset_path, 
                                    img_transform=img_transform)
    trainloader = DataLoader(dataset, batch_size=256, shuffle=True)

    net = Siamese().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
    optimizer = optim.Adam(net.parameters(),lr = 0.00006 )

    print(f"dataloader len: {len(trainloader)}")
    counter = []
    loss_history = [] 
    iteration_number= 0     
    for epoch in range(0, 100):
        for i, data in enumerate(trainloader,0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output = net(img0, img1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if i %500 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss.item()))
                iteration_number +=10
                counter.append(iteration_number)
                cpu_loss = loss.item()
                loss_history.append(cpu_loss)
                save_model(net, cpu_loss, 'iter_'+str(epoch))
    
    save_model(net, loss.item(), 'final')

    show_plot(counter,loss_history)


if __name__ == '__main__':
    train()