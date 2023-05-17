import sys
import os
import requests
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
from PIL import Image
import cv2

from dataset_cars import *
from model import *
import config


os.environ["CUDA_VISIBLE_DEVICES"]='0'

dataset = Image_Dataset(parent_path='./cars_data/cars_test/cars_test')
dataloader = DataLoader(dataset,shuffle=False,batch_size=1)

# train_ds = Image_Dataset(parent_path='/media/HDD8TB3/sourabh/deep_sight_mae/mae_front_defect_patches/shapes_data/shapes/',mode='val')
# dataloader = DataLoader(train_ds,batch_size=1, num_workers = 4,shuffle=True)


print(len(dataloader))

model = MaskedAutoencoderViT(img_size=224,patch_size=16,in_chans=config.IN_CHANS,embed_dim=512,
                             depth=12,num_heads=config.N_HEADS,decoder_depth=2,
                             decoder_embed_dim=256,decoder_num_heads=config.DECODER_N_HEADS)

print(sum([param.numel() for param in model.parameters()]))

print('MAE model defined')

checkpoint = torch.load('./weights/cars_MAE_expt_adamw_enc_emb_512_dp_12_dec_emb_256_dp_2_mr_75/epoch_2900.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])

print('MAE model loaded')

model.cuda()
model.eval()

write_path = '/media/HDD8TB3/sourabh/deep_sight_mae/mae_front_defect_patches/weights/cars_MAE_expt_adamw_enc_emb_512_dp_12_dec_emb_256_dp_2_mr_75/epoch_2900_results/'

with torch.no_grad():

    for idx,x in tqdm(enumerate(dataloader)):
        x = x.float().cuda()
        loss, y, mask = model(x, mask_ratio=0.75)
 
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', x).detach().cpu()

        # print(x.shape)
        # print(y.shape)
        # print(mask.shape)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # save images 
        x_org = x[0] * 255.0
        x_org = (x_org.numpy()).astype(np.uint8)
        #print(x_org.shape)
        cv2.imwrite(write_path + str(idx+1) + '_1_org.png',cv2.cvtColor(x_org,cv2.COLOR_RGB2BGR))

        im_masked = im_masked[0] * 255.0
        im_masked = (im_masked.numpy()).astype(np.uint8)
        #print(im_masked.shape)
        cv2.imwrite(write_path + str(idx+1) + '_2_masked.png',cv2.cvtColor(im_masked,cv2.COLOR_RGB2BGR))

        # y = y[0] * 255.0
        # y = (y.numpy()).astype(np.uint8)
        # #print(y.shape)
        # cv2.imwrite('3_recon.png',cv2.cvtColor(y,cv2.COLOR_RGB2BGR))

        im_paste = im_paste[0] * 255.0
        im_paste = (im_paste.numpy()).astype(np.uint8)
        #print(im_paste.shape)
        cv2.imwrite(write_path + str(idx+1) + '_3_recon.png',cv2.cvtColor(im_paste,cv2.COLOR_RGB2BGR))


