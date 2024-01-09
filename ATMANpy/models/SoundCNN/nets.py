# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:14:51 2023

@author: gianl
"""


import math
import random
import numpy as np

import torch
import torch.nn as nn

from functools import partial

def calc_dims(in_dim, kernel = 3, padding = 0, dilation = 1, stride = 1):
    num = in_dim + 2 * padding - dilation * (kernel - 1) - 1
    den = stride
    return int(num / den + 1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size, pool_kernel_size):
        super(ConvBlock, self).__init__()
        
        conv_stride, conv_padding, conv_dilation = 1, 0, 1
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size = conv_kernel_size, 
                              stride = conv_stride, 
                              padding = conv_padding,
                              dilation = conv_dilation,
                              bias = False)  ### we are doing batchnorm right after -> bias not needed!
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        pool_stride, pool_padding, pool_dilation = pool_kernel_size, 0, 1
        self.mp = nn.MaxPool2d(kernel_size = pool_kernel_size,
                               stride = pool_stride,
                               padding = pool_padding,
                               dilation = pool_dilation)
        
        self.relu = nn.ReLU()
        
        ### functions for calculating the reduction in height and width
        self.dimFunc_conv = partial(calc_dims, 
                             kernel = conv_kernel_size, 
                             padding = conv_padding, 
                             dilation = conv_dilation,
                             stride = conv_stride)
        
        self.dimFunc_mp = partial(calc_dims, 
                             kernel = pool_kernel_size, 
                             padding = pool_padding, 
                             dilation = pool_dilation,
                             stride = pool_stride)
    
    def calculate_out_dimensions(self, curr_dims):
        h, w = curr_dims
        h_out = self.dimFunc_mp(self.dimFunc_conv(h))
        w_out = self.dimFunc_mp(self.dimFunc_conv(w))
        return h_out, w_out
    
    def forward(self, x):
        return self.relu(self.mp(self.bn(self.conv(x))))
    
class SoundModel_avgpool(nn.Module):
    def __init__(self, conv_filters, conv_kernels, pool_kernels, frequencies = 512, hops = 81, share_shots = 0.1):
        super(SoundModel_avgpool, self).__init__()
        
        ### Populate Convolutional Blocks
        self.conv_blocks = nn.ModuleList()
        prev_n = 1
        curr_dims = frequencies, hops
        for n, c_k, p_k in zip(conv_filters, conv_kernels, pool_kernels):
            ### populate conv blocks
            currBlock = ConvBlock(prev_n, n, c_k, p_k)
            self.conv_blocks.append(currBlock)
            
            ### calculate dimensions
            curr_dims = currBlock.calculate_out_dimensions(curr_dims)
            
            ### output channels become next input channels
            prev_n = n
            
        ### Global Average Pooling
        self.global_avgpool = nn.AvgPool2d(curr_dims)
        
        ### convolutional implementation of fully connected layer
        self.conv_fc_out = nn.Conv2d(n, 1, 1) ## note that n here is the last number of conv_filters n 
        prior_bias = - np.log(1 - share_shots) + np.log(share_shots)
        self.conv_fc_out.bias = nn.Parameter(torch.Tensor(np.array([prior_bias])))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = x
        for convBlock in self.conv_blocks:
            y = convBlock(y)
        
        ypool = self.global_avgpool(y)
        z = self.conv_fc_out(ypool)
        p = self.sigmoid(z)
        return p
            
class SoundModel_FC(nn.Module):
    ### fully connected layers implemented convolutionally
    def __init__(self, conv_filters, conv_kernels, pool_kernels, fc_neurons,
                 frequencies = 512, hops = 81, share_shots = 0.1):
        super(SoundModel_FC, self).__init__()
        
        ### Populate Convolutional Blocks
        self.conv_blocks = nn.ModuleList()
        prev_n = 1
        curr_dims = frequencies, hops
        for n, c_k, p_k in zip(conv_filters, conv_kernels, pool_kernels):
            ### populate conv blocks
            currBlock = ConvBlock(prev_n, n, c_k, p_k)
            self.conv_blocks.append(currBlock)
            
            ### calculate dimensions
            curr_dims = currBlock.calculate_out_dimensions(curr_dims)
            
            ### output channels become next input channels
            prev_n = n
            
        ### Fully Connected Layer In (implemented convolutionally)
        self.conv_fc_in = nn.Conv2d(in_channels = n, out_channels = fc_neurons, kernel_size = curr_dims) ## note that n here is the last number of conv_filters n 
        
        ### Fully Connected Layer Out (implemented convolutionally)
        self.conv_fc_out = nn.Conv2d(in_channels = fc_neurons, out_channels = 1, kernel_size = 1) 
        
        ### Reset Bias Terms to match prior proportion of shots heard
        prior_bias = - np.log(1 - share_shots) + np.log(share_shots)
        self.conv_fc_out.bias = nn.Parameter(torch.Tensor(np.array([prior_bias])))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = x
        for convBlock in self.conv_blocks:
            y = convBlock(y)
        
        y_fc = self.conv_fc_in(y)
        logits = self.conv_fc_out(y_fc)
        probs = self.sigmoid(logits)
        return probs
            

class SoundModel_MN(nn.Module):
    ### Multinomial implementation of the soundmodel, without final sigmoid.
    def __init__(self, conv_filters, conv_kernels, pool_kernels, fc_neurons,
                 frequencies = 512, hops = 81, initial_output_shares = [0.1]):
        super(SoundModel_MN, self).__init__()
        
        out_channels = len(initial_output_shares)
        initial_output_shares = torch.Tensor(initial_output_shares)
        
        ### Populate Convolutional Blocks
        self.conv_blocks = nn.ModuleList()
        prev_n = 1
        curr_dims = frequencies, hops
        for n, c_k, p_k in zip(conv_filters, conv_kernels, pool_kernels):
            ### populate conv blocks
            currBlock = ConvBlock(prev_n, n, c_k, p_k)
            self.conv_blocks.append(currBlock)
            
            ### calculate dimensions
            curr_dims = currBlock.calculate_out_dimensions(curr_dims)
            
            ### output channels become next input channels
            prev_n = n
            
        ### Fully Connected Layer In (implemented convolutionally)
        self.conv_fc_in = nn.Conv2d(in_channels = n, out_channels = fc_neurons, kernel_size = curr_dims) ## note that n here is the last number of conv_filters n 
        
        ### Fully Connected Layer Out (implemented convolutionally)
        self.conv_fc_out = nn.Conv2d(in_channels = fc_neurons, out_channels = out_channels, kernel_size = 1)  
        
        ### Reset Bias Terms to match prior proportion of shots heard / bounces heard
        # since shots and bounces are orthogonal in this model 
        # (the probability of a shot being heard does not diminish the probability of a bounce being heard),
        # we can initialise the biases like we would in binary cross entropy, for each output channel
        prior_biases = - torch.log(1 - initial_output_shares) + torch.log(initial_output_shares)
        self.conv_fc_out.bias = nn.Parameter(prior_biases)
    
    def forward(self, x):
        y = x
        for convBlock in self.conv_blocks:
            y = convBlock(y)
        
        y_fc = self.conv_fc_in(y)
        logits = self.conv_fc_out(y_fc)
        return logits
    