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

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dataset_path', default = './2D_US/', type=str) 
parser.add_argument('--train_csv', default='train_SSL_2D_finetune1.csv', type=str)
parser.add_argument('--val_csv', default='val.csv', type=str)
parser.add_argument('--save_dir', default = './results_pretrain_TransUNet/', type=str, help = 'folder for pretrained model') 
parser.add_argument('--save_dir_finetune', default = './results_finetune_TransUNet/', type=str) 
parser.add_argument('--vit_name', default = 'R50-ViT-B_16', type=str , metavar='MODEL',
                    help='Name of model to train: R50-ViT-B_16/None; if None, model will be UNet') 
parser.add_argument('--model_name', default = 'latest_model.pth', type=str, help = 'Pretrained model name') 
parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)') 
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
parser.add_argument('--lr', type=float, default=0.002, 
                        help='learning rate') 
parser.add_argument('--finetune_option', default = 'encoder_decoder', type=str, help = 'encoder/encoder_decoder/all') 
parser.add_argument('--output_channel', default = 1, type=int, help = 'output prediction: 1/4') 


args = parser.parse_args() #change



print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
if os.path.exists(args.save_dir_finetune):
    pass
else:
    os.mkdir(args.save_dir_finetune)


from segmentation_models_pytorch.losses.dice import DiceLoss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, threshold = 0.5):
        super(DiceBCELoss, self).__init__()
        self.thredshold = threshold
        self.diceloss = DiceLoss('binary', from_logits = False)
    def forward(self, inputs, targets, smooth=1): 
        dice_loss = self.diceloss(inputs, targets) 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE
    
# evaluation metrics, we only evaluate dice and jaccard during training/validation phase
def evaluation_metrics(y_true, y_pred, smooth = 1):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    cm1 = confusion_matrix(y_true_f, y_pred_f)
    intersection = np.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    jaccard = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)
    return(dice, jaccard)       


class SegmentationDataSet(data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, image_transform=None, output_channel = 4):
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
            #for 1 channel output: bony region/background
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

        return image, mask 
if args.vit_name: #TransUNet
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes= args.output_channel #add
    model = ViT_seg(config_vit, img_size=224, num_classes=args.output_channel)
else:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  
        classes=args.output_channel,                      
        activation = "sigmoid"
    )

############## Notice: remember to add updated weights to model

PATH = args.save_dir_finetune + 'latest_model.pth'
PATH_pretrained = args.save_dir + args.model_name 
#PATH = ""
if os.path.exists(PATH): #if path exist, continue training(finetuning)
    model.load_state_dict(torch.load(PATH))
    print('continue finetuning')
elif os.path.exists(PATH_pretrained):
    pretrained_dict = torch.load(PATH_pretrained)
    pretrained_dict_keys = list(pretrained_dict.keys())
    
    if args.finetune_option == 'all':
        pass
    elif args.finetune_option == 'encoder_decoder':
        for i in range(len(pretrained_dict_keys)):
            if 'segmentation_head' in pretrained_dict_keys[i]:
                del pretrained_dict[pretrained_dict_keys[i]]
        print('finetune the model: encoder-decoder')
    elif args.finetune_option == 'encoder':
        for i in range(len(pretrained_dict_keys)):
            if 'segmentation_head' in pretrained_dict_keys[i] or 'decoder' in pretrained_dict_keys[i]:
                del pretrained_dict[pretrained_dict_keys[i]]
        print('finetune the model: encoder')
    else:
        print('finetune the model: encoder-decoder-segmentation head')
                
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

else:
    print('model training: from imagenet weights')



# Training function
def train_model(dataloader, optimizer, model):
    train_loss = 0
    model.train()
    dices = []
    jaccards = []

    for images, masks in dataloader:
        images = images.cuda()
        masks = masks.cuda()
        preds = model(images)
        preds = preds.cuda()
        #for debug
        #print(preds.shape, masks.shape, images.shape)
        #print(type(preds), type(masks))
        #print('preds:', preds.max(), preds.min(), ', masks:', masks.max(), masks.min())
        loss = DiceBCELoss().forward(preds, masks) # Calculate Loss
        #debug
        #print(loss)
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        train_loss =+ loss.item()
        masks = masks.cpu()
        preds = preds.cpu()
        preds[preds>=0.5] = 1 #change threshold
        preds[preds<0.5] = 0
        dice, jaccard = evaluation_metrics(masks, preds)     
        dices.append(dice)
        jaccards.append(jaccard)

        
    return train_loss, model, sum(dices)/len(dices), sum(jaccards)/len(jaccards)

# Validation function
def eval_model(dataloader, model):
    eval_loss = 0
    dices = []
    jaccards = []
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.cuda()
            masks = masks.cuda()
            preds = model(images)
            preds = preds.cuda()
            #print(preds.shape, masks.shape, images.shape)
            loss = DiceBCELoss().forward(preds, masks)
            eval_loss =+ loss.item()
            masks = masks.cpu()
            preds = preds.cpu()
            preds[preds>=0.5] = 1 
            preds[preds<0.5] = 0
            dice, jaccard  = evaluation_metrics(masks, preds)
            dices.append(dice)
            jaccards.append(jaccard)


    return eval_loss, sum(dices)/len(dices), sum(jaccards)/len(jaccards)

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Create training dataset
training_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.train_csv, 
                                       image_transform=train_transform, output_channel = args.output_channel)
# Create validation dataset
validation_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.val_csv,
                                        image_transform=train_transform, output_channel = args.output_channel)

# Initialization

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = True)

validation_dataloader = data.DataLoader(dataset=validation_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = False)     

# Run training and evaluation cycles
print('------------------------------------------------------------------------')
print('Epochs:', args.epochs)
print('Batch size:', args.batch_size)
print('Learning rate:', args.lr)
print('')

with open(args.save_dir_finetune + 'training_result.txt', 'a') as f:
    f.write('Batch size:'+str(args.batch_size)+'Learning rate:'+str(args.lr))


model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay = args.weight_decay)
torch.set_grad_enabled(True)

result_path = args.save_dir_finetune + "training_result.pickle"
if not os.path.exists(result_path): #if results do not exit
    results = {'train_loss': [], 'eval_loss': [], 'train_dice': [], 'eval_dice': [],
               'train_jaccard': [], 'eval_jaccard': []}
else:
    results_ = open(result_path,'rb')
    results = pickle.load(results_)
    results_.close()
print(results)

if len(results['eval_loss']) == 0:
    best_val_loss = np.inf
else:
    best_val_loss = min(results['eval_loss'])
print('best eval loss:', best_val_loss)


for epoch in range(args.start_epoch, args.epochs):
    train_loss, model, dice_train, jaccard_train = train_model(training_dataloader, optimizer, model)
    eval_loss, dice_eval, jaccard_eval = eval_model(validation_dataloader, model)
    # save model
    if epoch < 100:
        if eval_loss<best_val_loss:
            best_val_loss = eval_loss # save on val loss
            torch.save(model.state_dict(),args.save_dir_finetune + 'best_model_loss100.pth') #change
    else:
        if eval_loss<best_val_loss:
            best_val_loss = eval_loss # save on val loss
            torch.save(model.state_dict(),args.save_dir_finetune + 'best_model_loss200.pth') #change

    if epoch % 1 == 0:
        print("(epoch "+str(epoch)+")", 
              "\t"+"train loss: "+str(train_loss)+
              "\t"+"eval loss: "+str(eval_loss)+
              "\t"+"training dice: "+str(dice_train)+
              "\t"+"eval dice: "+str(dice_eval)+
              "\t"+"training jaccard: "+str(jaccard_train)+
              "\t"+"eval jaccard: "+str(jaccard_eval)+
              "\t"+"best eval loss: "+str(best_val_loss))
        torch.save(model.state_dict(),args.save_dir_finetune + 'latest_model.pth') #change
        with open(args.save_dir_finetune + 'training_result.txt', 'a') as f:
            f.write("(epoch "+str(epoch)+")"+ 
                    "\t"+"train loss: "+str(train_loss)+
                    "\t"+"eval loss: "+str(eval_loss)+
                    "\t"+"training dice: "+str(dice_train)+
                    "\t"+"eval dice: "+str(dice_eval)+
                    "\t"+"training jaccard: "+str(jaccard_train)+
                    "\t"+"eval jaccard: "+str(jaccard_eval)+
                    "\t"+"best eval loss: "+str(best_val_loss)+"\n")
    results['train_loss'].append(train_loss)
    results['eval_loss'].append(eval_loss)
    results['train_dice'].append(dice_train)
    results['eval_dice'].append(dice_eval)
    results['train_jaccard'].append(jaccard_train)
    results['eval_jaccard'].append(jaccard_eval)

    pickle.dump(results, open(args.save_dir_finetune + "training_result.pickle", "wb")) #change

     