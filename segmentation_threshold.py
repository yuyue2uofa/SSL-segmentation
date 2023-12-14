#!/usr/bin/env python
# coding: utf-8

# If there's a GPU available...
import torch

if torch.cuda.is_available():
    device = "cuda"
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print('We will use CPU')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage.io import imread
from skimage.transform import resize
import skimage
from skimage import filters

import torchvision
import torchvision.transforms as transforms
from scipy.spatial.distance import directed_hausdorff


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2 as cv

import glob

import pickle
import re
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp

from TransUNet.vit_seg_modeling import *
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg 
from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg


parser = argparse.ArgumentParser(description='TransUNet/UNet')
parser.add_argument('--dataset_path', default = './2D_US/', type=str) 
parser.add_argument('--val_csv', default='val.csv', type=str)
parser.add_argument('--save_dir_finetune', default = './results_finetune_TransUNet/', type=str) 
parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)') 
parser.add_argument('--finetune_option', default = 'all', type=str, help = 'encoder/encoder_decoder/all') 
parser.add_argument('--output_channel', default = 1, type=int, help = 'TransUNet/UNet output channel') 
parser.add_argument('--threshold', default=0.5, type=float,
                        help='segmentation threshold, change it based on validation set results using Otsu’s method')
parser.add_argument('--model_name', default='best_model.pth', type=str,
                        help='best model selection')
parser.add_argument('--vit_name', default = 'R50-ViT-B_16', type=str , metavar='MODEL',
                    help='Name of model to train:R50-ViT-B_16/None; if None, model will be UNet') 




args = parser.parse_args() 
print(args.save_dir_finetune, args.save_prediction)


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print('output channel:',str(args.output_channel), 'model dir:', args.save_dir_finetune)



# evaluation metrics
def evaluation_metrics(y_true, y_pred, smooth = 1):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    cm1 = confusion_matrix(y_true_f, y_pred_f)
    intersection = np.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    jaccard = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)
    mcc = (cm1[1, 1]*cm1[0, 0]-cm1[0, 1]*cm1[1, 0])/(math.sqrt((cm1[1, 1]+cm1[0, 1])*(cm1[1, 1]+cm1[1, 0])*(cm1[0, 0]+cm1[0, 1])*(cm1[0, 0]+cm1[1, 0])+0.00001))
    return(dice, jaccard, mcc)



class SegmentationDataSet(data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, image_transform=None, output_channel = 1):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')
        self.output_path = os.path.join(self.dataset_path, 'Masks/')
        self.df = pd.read_csv(df_input)
        self.images_list = list(self.df['filename'])
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.image_transform = image_transform
        self.output_channel = output_channel
        
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, index: int):
        # Select the sample
        image_filename = self.images_list[index]
        # Load input and target
        image = cv.imread(os.path.join(self.input_path, image_filename),0)
        mask = cv.imread(os.path.join(self.output_path, image_filename),0)
        
        # padding
        width = max(image.shape) - image.shape[1]  # pad 0 on width
        height = max(image.shape) - image.shape[0]  # pad 0 on height
        image = np.pad(image, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        mask = np.pad(mask, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        
        
        if self.output_channel == 4:
            #for 4 channel output
            mask1 = mask.copy()
            mask2 = mask.copy()
            mask3 = mask.copy()

            mask1[mask1 == 1] = 255
            mask1[mask1 == 4] = 255
            mask2[mask2 == 2] = 255
            #mask2[mask2 == 4] = 255
            mask3[mask3 == 3] = 255
            mask[mask == 4] = 255
            mask = np.dstack((mask1, mask2, mask3, mask))
        
            mask = cv.resize(mask, (224,224)) #256, 256, 3
            mask[mask<128] = 0
            mask[mask>=128] = 1
            mask = np.transpose(mask, (2,0,1))
        else: #1 channel
            #for 1 channel output
            mask[mask >= 1] = 1        
            mask = cv.resize(mask, (224,224)) #256, 256, 3
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
            mask = np.expand_dims(mask, axis=0)
            #mask = np.transpose(mask, (2,0,1))


        # add: 3 channel
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
        # add transform
        if self.image_transform:
            image = self.image_transform(np.uint8(image))
              
        # Typecasting
        mask = torch.from_numpy(mask).type(self.targets_dtype)




        return image, mask, image_filename


# In[95]:

if args.vit_name:
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes= args.output_channel 

    model = ViT_seg(config_vit, img_size=224, num_classes=args.output_channel)
else:
    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=3,                  
        classes=args.output_channel,                      
        activation = "sigmoid"
    )


############## Notice: remember to add updated weights to model

PATH = args.save_dir_finetune + args.model_name 
#PATH = ""
if os.path.exists(PATH): #if path exist, evaluate model performance
    model.load_state_dict(torch.load(PATH))
    print('evaluate model performance on test set')
else:
    print('warning: no trained model found')



# Evaluation function
def eval_model(dataloader, model):
    model.eval()
    thresholds = []
    with torch.no_grad():
        i = 0
        for images, masks, image_filename in validation_dataloader:
            i += 1
            images = images.cuda()
            preds = model(images)
            preds = preds.cuda()

            masks = masks.cpu()
            preds = preds.cpu()
            gray_image = skimage.color.rgb2gray(preds[0][0])
            thresholds.append(filters.threshold_otsu(gray_image))

    return sum(thresholds)/len(thresholds)



train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Create validation(test) dataset
validation_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.val_csv,
                                        image_transform=train_transform, output_channel = args.output_channel)

# Initialization

validation_dataloader = data.DataLoader(dataset=validation_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = False)


model = model.cuda()
threshold= eval_model(validation_dataloader, model)
print("eval threshold: " + str(threshold))