import torch
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import torch.nn.functional as F


class FaceRecogniction(object):
    def __init__(self, model_path=str(Path(__file__).parent/'models/final_loss_1.0145.pth')):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path).to(self.device)
        self.model.eval()
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.distance_treshhold = 0.4

    def _predict_numpy(self, first_image, second_image):
        first_image_tensor = self.img_transforms(first_image)
        first_image_tensor = first_image_tensor.unsqueeze(0)
        first_image_tensor = first_image_tensor.to(self.device)

        second_image_tensor = self.img_transforms(second_image)
        second_image_tensor = second_image_tensor.unsqueeze(0)
        second_image_tensor = second_image_tensor.to(self.device)

        output1,output2 = self.model(first_image_tensor,second_image_tensor)
        euclidean_distance = F.pairwise_distance(output1, output2).item()
        print(euclidean_distance)
        if euclidean_distance < self.distance_treshhold:
            return True
        else: 
            return False


if __name__ == '__main__':
    import cv2
    kjn = FaceRecogniction()
    img1_path = 'dataset/test_faces/real_00001.jpg'
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_path = 'dataset/test_faces/real_00003.jpg'
    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    print(kjn._predict_numpy(img1, img2))    

