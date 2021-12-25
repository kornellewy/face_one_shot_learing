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

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import load_files_with_given_extension, random_idx_with_exclude

class SiameseDataset(Dataset):
    def __init__(self, dataset_path, img_transform=None):
        self.dataset_path = dataset_path
        self.images_paths = load_files_with_given_extension(dataset_path)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # same = 1, difrent = 0
        class_idx = random.randint(0,1)
        if class_idx == 1:
            image1 = cv2.imread(self.images_paths[idx])
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = image1.copy()
            if self.img_transform:
                image1 = self.img_transform(image=image1)['image']
                image2 = self.img_transform(image=image2)['image']
        elif class_idx == 0:
            image1 = cv2.imread(self.images_paths[idx])
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.imread(self.images_paths[random_idx_with_exclude([idx], [0, len(self.images_paths)-1])])
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            if self.img_transform:
                image1 = self.img_transform(image=image1)['image']
                image2 = self.img_transform(image=image2)['image']
        return image1, image2, torch.Tensor([class_idx])

if __name__=='__main__':
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
    image1, image2, class_idx = dataset[0]
    print('image1.shape: ', image1.shape)
    save_image(image1, 'image1.jpg')
    print('image2.shape: ', image2.shape)
    save_image(image2, 'image2.jpg')
    print('class_idx: ', class_idx)