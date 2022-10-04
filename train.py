import numpy as np 
import pandas as pd 
import os, sys, random
import numpy as np
import pandas as pd
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _utils

from random import choice

from skimage import io
from PIL import Image, ImageOps

import glob

#from torchsummary import summary
import logging

import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.utils import shuffle
# from apex import amp

import random

import time

from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter

from albumentations.augmentations.transforms import Lambda, ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, RandomResizedCrop
from albumentations.pytorch import ToTensor
from albumentations import Compose, OneOrOther

import albumentations

import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import warnings


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch_xla.distributed.parallel_loader as pl
import time


warnings.filterwarnings("ignore")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


norm_mean = [0.143] #0.458971
norm_std = [0.144] #0.225609

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio = (0.5, 2), p = 0.8)

def randomErase(image, **kwargs):
    return RandomErasing(image)

def sample_normalize(image, **kwargs):
    image = image/255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis = 0), image.reshape((-1, channel)).std(axis = 0)
    return (image-mean)/(std + 1e-3)

transform_train = Compose([
    # RandomBrightnessContrast(p = 0.8),
    RandomResizedCrop(512, 512, (0.5, 1.0), p = 0.5),
    ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, border_mode = cv2.BORDER_CONSTANT, value = 0.0, p = 0.8),
    # HorizontalFlip(p = 0.5),
    
    # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
    HorizontalFlip(p = 0.5),
    RandomBrightnessContrast(p = 0.8, contrast_limit=(-0.3, 0.2)),                             
    Lambda(image = sample_normalize),
    ToTensor(),
    Lambda(image = randomErase) 
    
])

transform_val = Compose([                                   
    Lambda(image = sample_normalize),
    ToTensor(),
])

transform_test = Compose([                                   
    Lambda(image = sample_normalize),
    ToTensor(),
])


def read_image(path, image_size = 512):
    img = Image.open(path)
    w, h = img.size
    long = max(w, h)
    w, h = int(w/long*image_size), int(h/long*image_size)
    img = img.resize((w, h), Image.ANTIALIAS)
    delta_w, delta_h = image_size - w, image_size - h
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return np.array(ImageOps.expand(img, padding).convert("RGB"))


class BAATrainDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            #nomalize boneage distribution
            df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean)/boneage_div )
            #change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        return (transform_train(image = read_image(f"{self.file_path}/{num//1000}/{num}.png"))['image'], Tensor([row['male']])), row['zscore']
    
    def __len__(self):
        return len(self.df)
class BAAValDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            #change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df
        self.df = preprocess_df(df)
        self.file_path = file_path
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (transform_val(image = read_image(f"{self.file_path}/{int(row['id'])}.png"))['image'], Tensor([row['male']])), row['boneage']
    
    def __len__(self):
        return len(self.df)
        
class BAATestDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            #change the type of gender, change bool variable to float32
            df['male'] = (df['Sex'] == 'M').astype('float32')
            df['boneage'] = df['Ground truth bone age (months)'].astype('float32')
            df['id'] = df['Case ID'].astype('int32')
            return df
        self.df = preprocess_df(df)
        print(self.df.head())
        self.file_path = file_path
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (transform_test(image = read_image(f"{self.file_path}/{int(row['id'])}.png"))['image'], Tensor([row['male']])), row['boneage']
    
    def __len__(self):
        return len(self.df) 

def create_data_loader(train_df, val_df, test_df, train_root, val_root, test_root):
    return BAATrainDataset(train_df, train_root), BAAValDataset(val_df, val_root), BAATestDataset(test_df, test_root)


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average = False)
    loss = 0
    for param in net.fc.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha*loss


def train_fn(net, train_loader, loss_fn, epoch, optimizer, device):
    '''
    checkpoint is a dict
    '''
    global total_size 
    global training_loss 

    net.fine_tune()
    if xm.is_master_ordinal():
        train_pbar = tqdm(train_loader)
        train_pbar.desc = f'Epoch {epoch + 1}'
    else:
        train_pbar = train_loader
    for batch_idx, data in enumerate(train_pbar):
        # #put data to GPU
        size = len(data[1])
        
        image, gender = data[0]
        image, gender= image.to(device), gender.to(device)

        label = data[1].to(device)

        batch_size = len(data[1])
        label = data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        #forward
        y_pred = net(image, gender)
        y_pred = y_pred.squeeze()

        # print(y_pred, label)
        loss = loss_fn(y_pred, label)
        #backward,calculate gradients
        total_loss = loss + L1_penalty(net, 1e-5)
        total_loss.backward()
        #backward,update parameter
        xm.optimizer_step(optimizer)

        #the learning rate should be update after optimizer's update 
        #change the learning rate, because using One cycle pollicy,the learning rate should be update per mini-batch
        # scheduler.step()

        batch_loss = loss.item()

        training_loss += batch_loss
        total_size += batch_size
        if xm.is_master_ordinal():
            train_pbar.set_postfix({'loss': batch_loss/batch_size})
        # print('loss:', batch_loss/batch_size)
        # print(f'xla:{xm.get_ordinal()}, batch is{batch_idx}, loss is {mse_loss/total_size}, {size}')
    return training_loss/total_size 

def evaluate_fn(net, val_loader, device):
    net.fine_tune(False)
    
    global mae_loss 
    global val_total_size 
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image, gender= image.to(device), gender.to(device)

            label = data[1].to(device)

            y_pred = net(image, gender)*boneage_div+boneage_mean
            # y_pred = net(image, gender)
            y_pred = y_pred.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            # print(batch_loss/len(data[1]))
            mae_loss += batch_loss
    return mae_loss


def test_fn(net, test_loader, device):
    net.train(False)
    
    global test_mae_loss 
    global test_total_size 
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            test_total_size += len(data[1])

            image, gender = data[0]
            image, gender= image.to(device), gender.to(device)

            label = data[1].to(device)

            y_pred = net(image, gender)*boneage_div+boneage_mean
            # y_pred = net(image, gender)
            y_pred = y_pred.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            # print(batch_loss/len(data[1]))
            test_mae_loss += batch_loss
    return mae_loss


def reduce_fn(vals):
    return sum(vals)
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch_xla.distributed.parallel_loader as pl
import time



def map_fn(index, flags):

  ## Setup
  root = '/content/drive/My Drive/BAA'
  model_name = 'rsa50_4.48' 
  path = f'{root}/{model_name}'

  if xm.is_master_ordinal():
    if not os.path.exists(path):
        os.mkdir(path)
        
  # Sets a common random seed - both for initialization and ensuring graph is the same
  seed_everything(seed=flags['seed'])

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = xm.xla_device()


#   mymodel = BAA_base(32)
  mymodel = BAA_New(32, *get_My_resnet50())
#   mymodel.load_state_dict(torch.load('/content/drive/My Drive/BAA/resnet50_pr_2/best_resnet50_pr_2.bin'))
  mymodel = mymodel.to(device)
  
  # Creates the (distributed) train sampler, which let this process only access
  # its portion of the training dataset.
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True)
  
  val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_set,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=False)
  
  test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_set,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=False)
  
  # Creates dataloaders, which load data in batches
  # Note: test loader is not shuffled or sampled
  train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=flags['batch_size'],
      sampler=train_sampler,
      num_workers=flags['num_workers'],
      drop_last=True)

  val_loader = torch.utils.data.DataLoader(
      val_set,
      batch_size=flags['batch_size'],
      sampler=val_sampler,
      shuffle=False,
      num_workers=flags['num_workers'])
  
  test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=flags['batch_size'],
      sampler=test_sampler,
      shuffle=False,
      num_workers=flags['num_workers'])  

  ## Network, optimizer, and loss function creation

  # Creates AlexNet for 10 classes
  # Note: each process has its own identical copy of the model
  #  Even though each model is created independently, they're also
  #  created in the same way.
  net = mymodel.train()

  global best_loss 
  best_loss = float('inf')
#   loss_fn =  nn.MSELoss(reduction = 'sum')
  loss_fn = nn.L1Loss(reduction = 'sum')
  lr = flags['lr']

  wd = 0
    
  optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
#   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
  scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

  ## Trains
  train_start = time.time()
  for epoch in range(flags['num_epochs']):
    global training_loss 
    training_loss = torch.tensor([0], dtype = torch.float32)
    global total_size 
    total_size = torch.tensor([0], dtype = torch.float32)

    global mae_loss 
    mae_loss = torch.tensor([0], dtype = torch.float32)
    global val_total_size 
    val_total_size = torch.tensor([0], dtype = torch.float32)

    global test_mae_loss 
    test_mae_loss = torch.tensor([0], dtype = torch.float32)
    global test_total_size 
    test_total_size = torch.tensor([0], dtype = torch.float32)
    # xm.rendezvous("initialization")

    start_time = time.time()
    para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    train_fn(net, para_train_loader, loss_fn, epoch, optimizer, device)
    
    ## Evaluation
    # Sets net to eval and no grad context
    para_val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
    evaluate_fn(net, para_val_loader, device)

    para_test_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
    test_fn(net, para_test_loader, device)

    scheduler.step()
    
    xm.save(net.state_dict(), '/'.join([path, f'{model_name}.bin']))  
    training_loss = xm.mesh_reduce('training_loss',training_loss,reduce_fn)
    total_size = xm.mesh_reduce('total_size_reduce',total_size,reduce_fn)
    mae_loss = xm.mesh_reduce('mae_loss_reduce',mae_loss,reduce_fn)
    val_total_size = xm.mesh_reduce('val_total_size_reduce',val_total_size,reduce_fn)
    test_mae_loss = xm.mesh_reduce('test_mae_loss_reduce',test_mae_loss,reduce_fn)
    test_total_size = xm.mesh_reduce('test_total_size_reduce',test_total_size,reduce_fn)

    if xm.is_master_ordinal():
        print(test_total_size)
        train_loss, val_mae, test_mae = training_loss/total_size, mae_loss/val_total_size, test_mae_loss/test_total_size
        print(f'training loss is {train_loss}, val loss is {val_mae}, test loss is {test_mae}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')


    if xm.is_master_ordinal() and best_loss >= test_mae:
        best_loss = test_mae
        shutil.copy(f'{path}/{model_name}.bin', \
                    f'{path}/best_{model_name}.bin')



def map_ensemble_fn(index, flags):

  ## Setup
  root = '/content/drive/My Drive/BAA'
  model_name = 'final_ensemble_3.88' 
  path = f'{root}/{model_name}'

  if xm.is_master_ordinal():
    if not os.path.exists(path):
        os.mkdir(path)
        
  # Sets a common random seed - both for initialization and ensuring graph is the same
  seed_everything(seed=flags['seed'])

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = xm.xla_device()


#   mymodel = BAA_base(32)
  net = Ensemble(new_model)
#   mymodel.load_state_dict(torch.load('/content/drive/My Drive/BAA/resnet50_pr_2/best_resnet50_pr_2.bin'))
  net = net.to(device)
  
  # Creates the (distributed) train sampler, which let this process only access
  # its portion of the training dataset.
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True)
  
  val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_set,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=False)
  
  test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_set,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=False)
  
  # Creates dataloaders, which load data in batches
  # Note: test loader is not shuffled or sampled
  train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=flags['batch_size'],
      sampler=train_sampler,
      num_workers=flags['num_workers'],
      drop_last=True)

  val_loader = torch.utils.data.DataLoader(
      val_set,
      batch_size=flags['batch_size'],
      sampler=val_sampler,
      shuffle=False,
      num_workers=flags['num_workers'])
  
  test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=flags['batch_size'],
      sampler=test_sampler,
      shuffle=False,
      num_workers=flags['num_workers'])  

  ## Network, optimizer, and loss function creation

  # Creates AlexNet for 10 classes
  # Note: each process has its own identical copy of the model
  #  Even though each model is created independently, they're also
  #  created in the same way.
  net.fine_tune()

  global best_loss 
  best_loss = float('inf')
#   loss_fn =  nn.MSELoss(reduction = 'sum')
  loss_fn = nn.L1Loss(reduction = 'sum')
  lr = flags['lr']

  wd = 0
    
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
#   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
  scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

  ## Trains
  train_start = time.time()
  for epoch in range(flags['num_epochs']):
    global training_loss 
    training_loss = torch.tensor([0], dtype = torch.float32)
    global total_size 
    total_size = torch.tensor([0], dtype = torch.float32)

    global mae_loss 
    mae_loss = torch.tensor([0], dtype = torch.float32)
    global val_total_size 
    val_total_size = torch.tensor([0], dtype = torch.float32)

    global test_mae_loss 
    test_mae_loss = torch.tensor([0], dtype = torch.float32)
    global test_total_size 
    test_total_size = torch.tensor([0], dtype = torch.float32)
    # xm.rendezvous("initialization")

    start_time = time.time()
    para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    train_fn(net, para_train_loader, loss_fn, epoch, optimizer, device)
    
    ## Evaluation
    # Sets net to eval and no grad context
    para_val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
    evaluate_fn(net, para_val_loader, device)

    para_test_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
    test_fn(net, para_test_loader, device)

    scheduler.step()
    
    xm.save(net.state_dict(), '/'.join([path, f'{model_name}.bin']))  
    training_loss = xm.mesh_reduce('training_loss',training_loss,reduce_fn)
    total_size = xm.mesh_reduce('total_size_reduce',total_size,reduce_fn)
    mae_loss = xm.mesh_reduce('mae_loss_reduce',mae_loss,reduce_fn)
    val_total_size = xm.mesh_reduce('val_total_size_reduce',val_total_size,reduce_fn)
    test_mae_loss = xm.mesh_reduce('test_mae_loss_reduce',test_mae_loss,reduce_fn)
    test_total_size = xm.mesh_reduce('test_total_size_reduce',test_total_size,reduce_fn)

    if xm.is_master_ordinal():
        print(test_total_size)
        train_loss, val_mae, test_mae = training_loss/total_size, mae_loss/val_total_size, test_mae_loss/test_total_size
        print(f'training loss is {train_loss}, val loss is {val_mae}, test loss is {test_mae}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')


    if xm.is_master_ordinal() and best_loss >= test_mae:
        best_loss = test_mae
        shutil.copy(f'{path}/{model_name}.bin', \
                    f'{path}/best_{model_name}.bin')



if __name__ == "__main__":
    from model import Ensemble, Graph_BAA, BAA_New, get_My_resnet50, BAA_Base
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type')
    parser.add_argument('lr', type=float)
    parser.add_argument('batch_size', type = int)
    parser.add_argument('num_epochs', type = int)
    parser.add_argument('seed', type = int)
    args = parser.parse_args()

    if args.model_type == 'ensemble':
        model = BAA_New(32, *get_My_resnet50())
        model.load_state_dict(torch.load('/content/drive/MyDrive/BAA/MRSA_50++_4.03/best_MRSA_50++_4.03.bin'))
        new_model = Graph_BAA(model)
        ensemble = Ensemble(new_model)
    else:
        model = BAA_New(32, *get_My_resnet50())

    flags = {}
    flags['lr'] = args.lr
    flags['batch_size'] = args.batch_size
    flags['num_workers'] = 2
    flags['num_epochs'] = args.num_epochs
    flags['seed'] = args.seed

    train_df = pd.read_csv(f'/content/drive/My Drive/BAA/train.csv')
    val_df = pd.read_csv(f'/content/drive/My Drive/BAA/Validation Dataset.csv')
    test_df = pd.read_excel('/content/drive/My Drive/BAA/Bone age ground truth.xlsx')
    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()
    train_set, val_set, test_set = create_data_loader(train_df, val_df, test_df, '/content/drive/My Drive/BAA/boneage-training-dataset', '/content/drive/My Drive/BAA/boneage-validation-dataset', '/content/drive/My Drive/BAA/Test Set Images')
    torch.set_default_tensor_type('torch.FloatTensor')
    if args.model_type == 'ensemble':
        xmp.spawn(map_ensemble_fn, args=(flags,), nprocs=8, start_method='fork')
    else:
        xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')