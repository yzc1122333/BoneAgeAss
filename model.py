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

warnings.filterwarnings("ignore")
from pretrainedmodels import se_resnext101_32x4d, se_resnet152, xception, inceptionv4, inceptionresnetv2
from torchvision.models import resnet34, resnet50
def get_My_resnet34():
    model = resnet34(pretrained = True)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels

def get_My_resnet50():
    model = resnet50(pretrained = True)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels

def get_My_se_resnet152():
    model = se_resnet152(pretrained = None)
    output_channels = model.last_linear.in_features
    model = nn.Sequential(*list(model.children())[:-2])
    return model, output_channels

def get_My_se_resnext101_32x4d():
    model = se_resnext101_32x4d()
    output_channels = model.last_linear.in_features
    model = nn.Sequential(*list(model.children())[:-2])
    return model, output_channels

def get_My_inceptionv4():
    model = inceptionv4()
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-2]
    
    model = nn.Sequential(*model)
    return model, output_channels

def get_My_inceptionresnetv2():
    model = inceptionresnetv2()
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-2]
    
    model = nn.Sequential(*model)
    return model, output_channels

def get_My_resnet34():
    model = resnet34(pretrained = True)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels

def get_My_resnet50():
    model = resnet50(pretrained = True)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels

def get_My_se_resnet152():
    model = se_resnet152(pretrained = None)
    output_channels = model.last_linear.in_features
    model = nn.Sequential(*list(model.children())[:-2])
    return model, output_channels

def get_My_se_resnext101_32x4d():
    model = se_resnext101_32x4d()
    output_channels = model.last_linear.in_features
    model = nn.Sequential(*list(model.children())[:-2])
    return model, output_channels

def get_My_inceptionv4():
    model = inceptionv4()
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-2]
    
    model = nn.Sequential(*model)
    return model, output_channels

def get_My_inceptionresnetv2():
    model = inceptionresnetv2()
    output_channels = model.last_linear.in_features
    model = list(model.children())[:-2]

class Pooling_attention(nn.Module):
  def __init__(self, input_channels, kernel_size = 1):
    super(Pooling_attention, self).__init__()
    self.pooling_attention = nn.Sequential(
        nn.Conv2d(input_channels, 1, kernel_size = kernel_size, padding = kernel_size//2),
        nn.ReLU()
    )
  def forward(self, x):
    return self.pooling_attention(x) 




class Part_Relation(nn.Module):
  def __init__(self, input_channels, reduction = [16], level = 1):
    super(Part_Relation, self).__init__()
    
    modules = []
    for i in range(level):
        output_channels = input_channels//reduction[i]
        modules.append(nn.Conv2d(input_channels, output_channels, kernel_size = 1))
        modules.append(nn.BatchNorm2d(output_channels))
        modules.append(nn.ReLU())
        input_channels = output_channels

    self.pooling_attention_0 = nn.Sequential(*modules)
    self.pooling_attention_1 = Pooling_attention(input_channels, 1)
    self.pooling_attention_3 = Pooling_attention(input_channels, 3)
    self.pooling_attention_5 = Pooling_attention(input_channels, 5)

    self.last_conv =  nn.Sequential(
        nn.Conv2d(3, 1, kernel_size = 1),
        nn.Sigmoid()
    )
  def forward(self, x):
    input = x
    x = self.pooling_attention_0(x)
    x = torch.cat([self.pooling_attention_1(x), self.pooling_attention_3(x), self.pooling_attention_5(x)], dim = 1)
    x = self.last_conv(x)
    return input - input*x
    
    model = nn.Sequential(*model)
    return model, output_channels

class BAA_New(nn.Module):
    def __init__(self, gender_encode_length, backbone, out_channels):
        super(BAA_New, self).__init__()
        self.backbone0 = nn.Sequential(*backbone[0:5])
        self.part_relation0 = Part_Relation(256)
        self.out_channels = out_channels
        self.backbone1 = backbone[5]
        self.part_relation1 = Part_Relation(512, [4, 8], 2)
        self.backbone2 = backbone[6]
        self.part_relation2 = Part_Relation(1024, [8, 8], 2)
        self.backbone3 = backbone[7]
        self.part_relation3 = Part_Relation(2048, [8, 16], 2)

        #3.788
        # self.part_relation0 = Part_Relation(256)
        # self.part_relation1 = Part_Relation(512, 32)
        # self.part_relation2 = Part_Relation(1024, 8, 2)
        # self.part_relation3 = Part_Relation(2048, 8, 2)

        
        
        
        

        self.gender_encoder = nn.Linear(1, gender_encode_length)
        self.gender_bn = nn.BatchNorm1d(gender_encode_length)

        self.fc0 = nn.Linear(out_channels + gender_encode_length, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.output = nn.Linear(512, 1)

    def forward(self, image, gender):
        x = self.part_relation0(self.backbone0(image))
        # x  = self.backbone0(image)
        x = self.part_relation1(self.backbone1(x))
        # x = self.backbone1(x)
        x = self.part_relation2(self.backbone2(x))
        # x = self.backbone2(x)
        x = self.part_relation3(self.backbone3(x))
        # x = self.backbone3(x)
        feature_map = x
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)
        image_feature = x

        gender_encode = self.gender_bn(self.gender_encoder(gender))
        gender_encode = F.relu(gender_encode)

        x = torch.cat([x,  gender_encode], dim = 1)

        x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn1(self.fc1(x)))

        x = self.output(x)

        return feature_map, gender_encode, image_feature, x

    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)



class BAA_Base(nn.Module):
    def __init__(self, gender_encode_length, backbone, out_channels):
        super(BAA_New, self).__init__()
        self.backbone0 = nn.Sequential(*backbone[0:5])
        self.part_relation0 = Part_Relation(256)
        self.out_channels = out_channels
        self.backbone1 = backbone[5]
        self.part_relation1 = Part_Relation(512, [4, 8], 2)
        self.backbone2 = backbone[6]
        self.part_relation2 = Part_Relation(1024, [8, 8], 2)
        self.backbone3 = backbone[7]
        self.part_relation3 = Part_Relation(2048, [8, 16], 2)

        #3.788
        # self.part_relation0 = Part_Relation(256)
        # self.part_relation1 = Part_Relation(512, 32)
        # self.part_relation2 = Part_Relation(1024, 8, 2)
        # self.part_relation3 = Part_Relation(2048, 8, 2)

        
        
        
        

        self.gender_encoder = nn.Linear(1, gender_encode_length)
        self.gender_bn = nn.BatchNorm1d(gender_encode_length)

        self.fc0 = nn.Linear(out_channels + gender_encode_length, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.output = nn.Linear(512, 1)

    def forward(self, image, gender):
        x = self.part_relation0(self.backbone0(image))
        # x  = self.backbone0(image)
        x = self.part_relation1(self.backbone1(x))
        # x = self.backbone1(x)
        x = self.part_relation2(self.backbone2(x))
        # x = self.backbone2(x)
        x = self.part_relation3(self.backbone3(x))
        # x = self.backbone3(x)
        feature_map = x
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)
        image_feature = x

        gender_encode = self.gender_bn(self.gender_encoder(gender))
        gender_encode = F.relu(gender_encode)

        x = torch.cat([x,  gender_encode], dim = 1)

        x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn1(self.fc1(x)))

        x = self.output(x)

        return  x

    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)



class Self_Attention_Adj(nn.Module):
    def __init__(self, feature_size, attention_size):
        super(Self_Attention_Adj, self).__init__()
        self.queue = nn.Parameter(torch.empty(feature_size, attention_size))
        nn.init.kaiming_uniform_(self.queue)

        self.key = nn.Parameter(torch.empty(feature_size, attention_size))
        nn.init.kaiming_uniform_(self.key)

        self.leak_relu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        Q = self.leak_relu(torch.matmul(x, self.queue))
        K = self.leak_relu(torch.matmul(x, self.key))

        return self.softmax(torch.matmul(Q, K.transpose(1, 2)))



class Graph_GCN(nn.Module):
    def __init__(self, node_size, feature_size, output_size):
        super(Graph_GCN, self).__init__()
        self.node_size = node_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(feature_size, output_size))
        nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, x, A):
        x = torch.matmul(A, x.transpose(1, 2))
        return (torch.matmul(x, self.weight)).transpose(1, 2)


class Graph_BAA(nn.Module):
    def __init__(self, backbone):
        super(Graph_BAA, self).__init__()
        self.backbone = backbone
        #freeze image backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adj_learning = Self_Attention_Adj(2048, 256)
        self.gconv = Graph_GCN(16*16, 2048, 1024)

        self.fc0 = nn.Linear(1024 + 32, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)

        self.output = nn.Linear(1024, 1)

    def forward(self, image, gender):
        #input image to backbone, 16*16*2048
        feature_map, gender, image_feature, cnn_result = self.backbone(image, gender)
        node_feature = feature_map.view(-1, 2048, 16*16)
        A = self.adj_learning(node_feature)
        x = F.leaky_relu(self.gconv(node_feature, A))
        x = torch.squeeze(F.adaptive_avg_pool1d(x, 1))
        graph_feature = x
        x = torch.cat([x, gender], dim = 1)

        x = F.relu(self.bn0(self.fc0(x)))
        # x = F.relu(self.bn1(self.fc1(x)))

        return image_feature,  graph_feature, gender, (self.output(x), cnn_result)

    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)
        self.backbone.eval()


class Ensemble(nn.Module):
    def __init__(self, model):
        super(Ensemble, self).__init__()
        self.model = model
        #freeze image backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # self.image_encoder = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU()
        # )

        self.fc = nn.Sequential(
            nn.Linear(1024 + 2048 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1)
        )


    def forward(self, image, gender):
        image_feature,  graph_feature, gender, result = self.model(image, gender)
        # image_feature = self.image_encoder(image_feature)
        if self.training:
            return (self.fc(torch.cat([image_feature, graph_feature, gender], dim = 1)) + result[0])/2
        else:
            return (self.fc(torch.cat([image_feature, graph_feature, gender], dim = 1)) + result[0])/2

        
    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)
        self.model.eval()