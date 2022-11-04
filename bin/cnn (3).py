import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2


# the model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 6)
        self.fc1 = nn.Linear(16 * 60 * 60, 10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 11)

    def forward(self, x, feature_conv=False):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))
        if feature_conv:
            return x  
        x = x.view(-1, 16 * 60 * 60) 
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x

# Class activation maps  
def returnCAM(feature_conv, weights_fc1, weights_fc2, weights_fc3, class_idx):
    # generate the class activation maps upsample to 512x512
    # feature_conv is of size 16x60x60
    # bs=batch size, nf=number of features, h= height, w=width
    transform = torchvision.transforms.Resize((512, 512))
    bs, nf, h, w = feature_conv.shape
    feature_conv = feature_conv.view(-1, 16 * 60 * 60)
    output_cam = []
    for idx in class_idx:
        x = torch.matmul(weights_fc3[idx], weights_fc2)
        x = torch.matmul(x, weights_fc1)
        x = torch.mul(feature_conv, x)
        x = x.reshape(bs, nf, h, w)
        x = torch.sum(x, 1)
        x = x - torch.min(x)
        x = x / torch.max(x)
        x = 255 * x
        x = transform(x)
        output_cam.append(x)
    return output_cam

def show_CAM(CAMs, width, height, orig_image, class_idx, save_name):
    #heatmap = cv2.applyColorMap(cv2.resize(CAMs,(width, height)), cv2.COLORMAP_JET)
    #result = heatmap * 0.5 + orig_image * 0.5
    return 0
    # put class label text on the result
    #cv2.putText(result, str(int(class_idx[i])), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('CAM', heatmap)
    #cv2.waitKey(0)
    #cv2.imwrite(f"CAM_{save_name}.jpg", result)

#class CustomImageDataset:
class CustomImageDataset:
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()