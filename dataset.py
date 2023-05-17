import glob  
import cv2 
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import pandas as pd 

import torch  
import torchvision
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset 
import random


class Image_Dataset(Dataset):
    def __init__(self,parent_path='',mode=''):
        super().__init__()
        self.parent_path = glob.glob(parent_path + '/**')
        if mode == 'val':
            #### validating on small subset of data
            self.parent_path = self.parent_path[:200]

    def __getitem__(self,index):
        curr_image_path = self.parent_path[index]
        curr_img = Image.open(curr_image_path)      
        curr_img = cv2.resize(np.array(curr_img), (224,224), interpolation=cv2.INTER_AREA)
        if len(curr_img.shape) != 3:
            curr_img = np.stack([curr_img,curr_img,curr_img],2)
        return torch.FloatTensor(np.array(curr_img)).permute(2,0,1)/255.0
    def __len__(self):
        return len(self.parent_path)


################ sanity check for the dataloader

# dataset = Image_Dataset(parent_path='./cars_data/cars_test/cars_test')
# print(len(dataset))
# dataloader = DataLoader(dataset,shuffle=True)


# for x in dataloader:
#     # x = x.squeeze().permute(1,2,0).numpy() * 255.0
#     # cv2.imwrite('img.png',x)
#     print(x.shape)
#     break


