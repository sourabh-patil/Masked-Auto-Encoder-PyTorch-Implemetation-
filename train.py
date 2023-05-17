import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision

import pandas as pd
import numpy as np

from tqdm import tqdm
import time
import random
from collections import Counter
import os

from dataset import *
from model import * 
from utils import *
import config

import argparse

EXPT_NAME = config.EXPT_NAME

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

writer = SummaryWriter('./runs/' + EXPT_NAME) 

NUM_EPOCHS = config.NUM_EPOCHS

BATCH_SIZE = config.BATCH_SIZE

if not os.path.exists(config.MODEL_SAVE_PATH):
   os.makedirs(config.MODEL_SAVE_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU

print()
print('###'*20)
print('###################### ' + EXPT_NAME)
print('###'*20)
print()

print(f'MIN_LR: {config.MIN_LR} | MAX_LR: {config.MAX_LR}')

model = MaskedAutoencoderViT(img_size=config.IMG_SIZE, patch_size=config.PATCH_SIZE, in_chans=config.IN_CHANS, embed_dim=config.EMBED_DIM,
                             depth=config.DEPTH, num_heads=config.N_HEADS, decoder_depth=config.DECODER_DEPTH,
                             decoder_embed_dim=config.DECODER_EMBED_DIM, decoder_num_heads=config.DECODER_N_HEADS)

print('### MAE model defined')

optimizer = torch.optim.AdamW(model.parameters(), lr=config.MAX_LR, weight_decay=config.WEIGHT_DECAY)


################################################# resume training 
# checkpoint = torch.load('....path_to_saved_model.....')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# for state in optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()
# print('### MAE model weights and optimizer states are imported')
# print('###############################')
# print('Resuming from epoch number: ')
# print(checkpoint['epoch'])
# print('###############################')
##################################################

print()
print(f'Total Number of trainable parameters: {sum([param.numel() for param in model.parameters()])}')
print()

model.cuda()

print('MAE Model pushed to cuda!!!')
print()

print(config.TRAIN_CSV)
print()


###################################################
BATCH_SIZE = config.BATCH_SIZE
        
train_dataset = Image_Dataset(parent_path='./cars_data/cars_train/cars_train')
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=config.TRAIN_NO_OF_WORKERS,shuffle=config.SHUFFLE)
#################################################################

scaler = torch.cuda.amp.GradScaler()

epochs_per_warm_restart = 200
num_iterations_for_each_restart = epochs_per_warm_restart * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = num_iterations_for_each_restart, eta_min=config.MIN_LR)

best_train_loss = 2
iter_count = 0

for epoch in range(config.NUM_EPOCHS):
    
    train_epoch_loss = 0

    epoch_losses = AverageMeter()

    model.train()

    curr_start = time.time()

    with tqdm(total=((len(train_loader)*BATCH_SIZE) - (len(train_loader)*BATCH_SIZE) % BATCH_SIZE), ncols=120) as t:
        
        t.set_description('MAE for cars : {}/{}'.format(epoch+1, config.NUM_EPOCHS))

        for batchidx, x in enumerate(train_loader):

            x = x.float().cuda()

            with torch.cuda.amp.autocast():
                train_loss, _, _ = model(x, mask_ratio=0.75)
            
            epoch_losses.update(train_loss.item(), len(x))
            
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            train_loss = train_loss.detach().cpu().numpy()
            train_epoch_loss += train_loss
            
            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(x))

            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], iter_count)
            iter_count += 1

        writer.add_scalar("Loss/train", train_epoch_loss/len(train_loader), epoch+1)


    writer.flush()

    curr_end = time.time()
        
    print() 
    print(f'Epoch Number {epoch + 1} time taken: {(curr_end - curr_start)//60} min')
    print()   
    print(f'Epoch {epoch+1:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f}')
    print()


    if not os.path.exists('./weights/' + EXPT_NAME):
        os.mkdir('./weights/' + EXPT_NAME)

    curr_train_loss = train_epoch_loss/len(train_loader)

    if curr_train_loss <= best_train_loss:
        ## save model 
        torch.save({'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_epoch_loss/len(train_loader)}, 
                 './weights/' + EXPT_NAME + '/best_epoch_train_loss.pth.tar')

        best_train_loss = curr_train_loss


    if (epoch+1) % 100 == 0:

        torch.save({'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_epoch_loss/len(train_loader)}, 
                 './weights/' + EXPT_NAME + '/epoch_' + str(epoch+1) +'.pth.tar')

        print(EXPT_NAME)


torch.save({'epoch': epoch+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_epoch_loss/len(train_loader)}, 
            './weights/' + EXPT_NAME + '/last_epoch.pth.tar')


writer.close()

print('###' * 20)
print(f'Experiment Name ==> {EXPT_NAME} Done!')
print('###' * 20)












