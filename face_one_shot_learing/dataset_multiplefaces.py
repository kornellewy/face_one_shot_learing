import os
import cv2
from PIL import Image
import numpy as np
from random import randint, choice, sample
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


class DatasetMultipleFaces(Dataset):
    def __init__(self, dataset_path, img_transform=None):
        self.dataset_path = dataset_path
        self.all_images_paths = load_files_with_given_extension(dataset_path)
        self.img_transform = img_transform
        self.classes = [ f.path for f in os.scandir(dataset_path) if f.is_dir() ]
        self.images_by_classes = {class_path: load_files_with_given_extension(class_path) for class_path in self.classes}
        print(self.images_by_classes)

    def __len__(self):
        return len(self.all_images_paths)

    def __getitem__(self, idx):
        # same = 1, difrent = 0
        class_idx = randint(0,1)
        if class_idx == 1:
            class_path = choice(self.classes)
            image1_path = choice(self.images_by_classes[class_path])
            image1 = cv2.imread(image1_path)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2_path = choice(self.images_by_classes[class_path])
            image2 = cv2.imread(image2_path)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            if self.img_transform:
                image1 = self.img_transform(image=image1)['image']
                image2 = self.img_transform(image=image2)['image']
        elif class_idx == 0:
            class_path1, class_path2 = sample(population=self.classes, k=2)
            image1_path = choice(self.images_by_classes[class_path1])
            image1 = cv2.imread(image1_path)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2_path = choice(self.images_by_classes[class_path2])
            image2 = cv2.imread(image2_path)
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
    dataset_path = 'dataset/'
    dataset = DatasetMultipleFaces(dataset_path=dataset_path, 
                            img_transform=img_transform)
    image1, image2, class_idx = dataset[0]
    print('image1.shape: ', image1.shape)
    save_image(image1, 'image1.jpg')
    print('image2.shape: ', image2.shape)
    save_image(image2, 'image2.jpg')
    print('class_idx: ', class_idx)