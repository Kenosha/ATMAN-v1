# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:19:29 2023

@author: gianl
"""


import math
import random
import numpy as np

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, k):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, k)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ToyModel(nn.Module):
    
    def __init__(self, in_dim = 10, share_balls = 0.1):
        super(ToyModel, self).__init__()
        n1, k1 = 64, 5
        n2, k2 = 16, 5
        k_avg = in_dim - k1 + 1 - k2 + 1
        
        ### convolutional layers
        self.conv1 = ConvBlock(3, n1, k1)
        self.conv2 = ConvBlock(n1, n2, k2)
        
        ### Average pooling
        self.global_avgpool = nn.AvgPool2d(k_avg)
        
        ### convolutional implementation of fully connected layer
        self.conv_fc_out = nn.Conv2d(n2, 1, 1) 
        prior_bias = - np.log(1 - share_balls) + np.log(share_balls)
        self.conv_fc_out.bias = nn.Parameter(torch.Tensor(np.array([prior_bias])))
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, verbose = False):
        if verbose: print(f'Shape of Input:             {x.shape}')
        
        y1 = self.conv1(x)
        if verbose: print(f'  ... after conv1:          {y1.shape}')
        
        y2 = self.conv2(y1)
        if verbose: print(f'  ... after conv2:          {y2.shape}')
            
        ypool = self.global_avgpool(y2)
        if verbose: print(f'  ... after global avgpool: {ypool.shape}')
        
        z = self.conv_fc_out(ypool)
        if verbose: print(f'  ... of Output:            {z.shape}')
        
        p = self.sigmoid(z)
        
        return p
    
class DynamicModel(nn.Module):
    def __init__(self, conv_filters, conv_kernels, in_dim = 10, share_balls = 0.1):
        super(DynamicModel, self).__init__()
        self.convs = nn.ModuleList()
        prev_n = 3
        for n, k in zip(conv_filters, conv_kernels):
            currBlock = ConvBlock(prev_n, n, k)
            self.convs.append(currBlock)
            prev_n = n
            
        k_avg = in_dim - sum(conv_kernels) + len(conv_kernels)
        
        ### Average pooling
        self.global_avgpool = nn.AvgPool2d(k_avg)
        
        ### convolutional implementation of fully connected layer
        self.conv_fc_out = nn.Conv2d(n, 1, 1) ## note that n here is the last number of conv_filters n 
        prior_bias = - np.log(1 - share_balls) + np.log(share_balls)
        self.conv_fc_out.bias = nn.Parameter(torch.Tensor(np.array([prior_bias])))
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, verbose = False):
        if verbose: print(f'Shape of Input:             {x.shape}')
        
        y = x
        for idx, convBlock in enumerate(self.convs): ### is it faster to compose the convolutions beforehand?
            y = convBlock(y)
            if verbose: print(f'  ... after conv{idx}:          {y.shape}')
        
        ypool = self.global_avgpool(y)
        if verbose: print(f'  ... after global avgpool: {ypool.shape}')
        
        z = self.conv_fc_out(ypool)
        if verbose: print(f'  ... of Output:            {z.shape}')
        
        p = self.sigmoid(z)
        
        return p
        
        
        
        
        
        