# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:49:37 2023

@author: gianl
"""

import math
import random
import numpy as np

import torch
import torch.nn as nn

class TwoPlusOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, t, 
                 stride_d = 1, stride_t = 1):
        super(TwoPlusOneBlock, self).__init__()

        ### Calculate the number of temporal kernels M, using 
            # temporal length of kernel t, 
            # height and width of kernel d
            # input channels N_in
            # spatial output channels N_out 
        # (see tran et al 2018)
        numerator = t * d ** 2 * in_channels * out_channels
        denominator = (d ** 2 * in_channels + t * out_channels)
        M_i = numerator // denominator
        
        ### Calculate padding
        pad_d = (d - 1) // 2  ### note that since d and t are odd, 
        pad_t = (t - 1) // 2  ### floor division only enforces an integer pad
        
        self.conv_spatial = nn.Conv3d(in_channels, M_i, (1, d, d), 
                                      stride = (1, stride_d, stride_d),
                                      padding = (0, pad_d, pad_d),
                                      bias = False) ### we are doing batchnorm right after -> bias not needed!
        self.bn1 = nn.BatchNorm3d(M_i)
        self.relu = nn.ReLU()
        self.conv_temporal = nn.Conv3d(M_i, out_channels, (t, 1, 1),
                                       stride = (stride_t, 1, 1),
                                       padding = (pad_t, 0, 0),
                                       bias = False) ### we are doing batchnorm right after -> bias not needed!
        self.bn2 = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        y = self.relu(self.bn1(self.conv_spatial(x)))
        return self.relu(self.bn2(self.conv_temporal(y)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, t, 
                 stride_d = 1, stride_t = 1):
        super(ResidualBlock, self).__init__()
        self.twoplusone1 = TwoPlusOneBlock(in_channels, out_channels, d, t,
                                           stride_d = stride_d, stride_t = stride_t) 
        self.twoplusone2 = TwoPlusOneBlock(out_channels, out_channels, d, t,
                                           stride_d = 1, stride_t = 1) 
        if stride_d == stride_t == 1:
            self.downsample_id = nn.Identity()
        else:
            self.downsample_id = nn.Conv3d(in_channels, out_channels, 1, 
                                           stride = (stride_t, stride_d, stride_d))

    def forward(self, x):
        f_x = self.twoplusone2(self.twoplusone1(x))
        down_x = self.downsample_id(x)
        return down_x + f_x
    
# class FullConvModel(nn.Module):
    
#     def __init__(self, *conv_args,
#                  num_labels = 14, T = 32, D = 112, share_other = 0.9):
#         ### conv_args are a list of dictionaries which have the following keys:
#             ### channels, kernel, stride_d, stride_t
#         super(FullConvModel, self).__init__()
        
#         self.num_labels = num_labels
#         self.T = T
#         self.D = D
#         self.share_other = share_other
        
#         conv_args[0] = TwoPlusOneBlock(...)
#         conv_args[1, 2, 3, ...] = ResidualBlock(...)
    
    
class SmallFullConvModel(nn.Module):
    
    def __init__(self, num_labels = 14, T = 32, D = 112,
                 stride_d1 = 2, stride_d2 = 1, stride_d3 = 2, stride_d4 = 2,
                 stride_t1 = 1, stride_t2 = 1, stride_t3 = 2, stride_t4 = 2,
                 share_other = 0.9):
        super(SmallFullConvModel, self).__init__()
        
        ### Write Init Params
        self.num_labels = num_labels
        self.T = T
        self.D = D
        self.share_other = share_other
        
        self.stride_d1 = stride_d1
        self.stride_d2 = stride_d2
        self.stride_d3 = stride_d3
        self.stride_d4 = stride_d4
        
        self.stride_t1 = stride_t1
        self.stride_t2 = stride_t2
        self.stride_t3 = stride_t3
        self.stride_t4 = stride_t4
        
        self.stride_d = stride_d1 * stride_d2 * stride_d3 * stride_d4
        self.stride_t = stride_t1 * stride_t2 * stride_t3 * stride_t4
        
        ### Write Building Blocks
        self.conv1 = TwoPlusOneBlock(3, 64, 3, 3, 
                                     stride_d = stride_d1, 
                                     stride_t = stride_t1)
        
        self.conv2_1 = ResidualBlock(64, 64, 3, 3, 
                                     stride_d = stride_d2, 
                                     stride_t = stride_t2)
        self.conv2_2 = ResidualBlock(64, 64, 3, 3,
                                     stride_d = 1, 
                                     stride_t = 1)
        
        self.conv3_1 = ResidualBlock(64, 128, 3, 3, 
                                     stride_d = stride_d3, 
                                     stride_t = stride_t3)
        self.conv3_2 = ResidualBlock(128, 128, 3, 3,
                                     stride_d = 1, 
                                     stride_t = 1)
        
        self.conv4_1 = ResidualBlock(128, 256, 3, 3, 
                                     stride_d = stride_d4, 
                                     stride_t = stride_t4)
        self.conv4_2 = ResidualBlock(256, 256, 3, 3,
                                     stride_d = 1, 
                                     stride_t = 1)
        
        self.avgpool_d = math.ceil(D / self.stride_d) ### 3 times stride_d == 2
        self.avgpool_t = math.ceil(T / self.stride_t) ### 2 times stride_t == 2
        
        ### now, as opposed to the models with fully connected layers
        ### we add a stride of 1 to the global average pooling 
        ### (default is kernel size (= (4, 7, 7)))
        ### with a stride of 1, we can apply average pooling in an overlapping manner.
        self.global_avgpool = nn.AvgPool3d((self.avgpool_t, self.avgpool_d, self.avgpool_d),
                                           stride = 1) 
        
        ### also, we replace the flatten and fc layer with num_labels 1x1x1 3D convolutions
        ### each convolution spans all 256 input channels,
        ### and so implements a linear function with 256 weights and a single bias.
        ### num_labels of these therefore act like a fully connected layer,
        ### turning 256 input channels into num_labels output activations,
        ### shaped like batch_size x num_labels x 1 x 1 x 1.
        ### Note that 1 x 1 x 1 kernel is only because we average pool over the last conv layer conv4_2.
        ### Note also now that if we have an input larger than the training images,
        ### we will get as output something shaped batch_size x num_labels x a x b x c
        ### where a, b, c can be larger than 1. 
        ### In each of these channels, we then find the logits for a different part of the input image!
        self.conv_fc_out = nn.Conv3d(256, num_labels, kernel_size = (1, 1, 1))
        
        ### re-initialize conv_fc_out biases
        # idea: initially, the output of the (convolutionally implemented) FC layer with Softmax has to (roughly) match
        # the empirical share of hits and serves and others, etc.
        # to do this, we set the first bias (for "other") to zero.
        # all other biases are the same -> topspin backhand groundstroke same as serve
        # given share_other, what do the other biases need to equal?
        bias_event = np.log(1 - share_other) - np.log(share_other) - np.log(num_labels - 1)
        bias_other = 0
        biases = [bias_other, *[bias_event] * (num_labels - 1)]
        biases = nn.Parameter(torch.Tensor(biases))
        self.conv_fc_out.bias = biases
        
        # self.softmax = nn.Softmax(dim = 1) ### dim = 1 --> the label dimension
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        
    def forward(self, x, verbose = False):
        if verbose: print(f'Shape of Input:             {x.shape}')
        
        y1 = self.conv1(x)
        if verbose: print(f'  ... after conv1:          {y1.shape}')
        
        y2 = self.conv2_2(self.conv2_1(y1))
        if verbose: print(f'  ... after conv2:          {y2.shape}')
            
        y3 = self.conv3_2(self.conv3_1(y2))
        if verbose: print(f'  ... after conv3:          {y3.shape}')
            
        y4 = self.conv4_2(self.conv4_1(y3))
        if verbose: print(f'  ... after conv4:          {y4.shape}')
        
        ypool = self.global_avgpool(y4)
        if verbose: print(f'  ... after global avgpool: {ypool.shape}')
        
        z = self.conv_fc_out(ypool)
        if verbose: print(f'  ... of Output:            {z.shape}')
        
        return z


class SmallModel(nn.Module):
    
    def __init__(self, num_labels = 14, T = 32, D = 112, share_other = 0.9):
        super(SmallModel, self).__init__()
       
        self.conv1 = TwoPlusOneBlock(3, 64, 3, 3, 
                                     stride_d = 2, stride_t = 1)
        
        self.conv2_1 = ResidualBlock(64, 64, 3, 3, 
                                     stride_d = 1, stride_t = 1)
        self.conv2_2 = ResidualBlock(64, 64, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv3_1 = ResidualBlock(64, 128, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv3_2 = ResidualBlock(128, 128, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv4_1 = ResidualBlock(128, 256, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv4_2 = ResidualBlock(256, 256, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        avgpool_t = math.ceil(T / (2 ** 2)) ### 2 times stride_t == 2
        avgpool_d = math.ceil(D / (2 ** 3)) ### 3 times stride_d == 2
    
        self.global_avgpool = nn.AvgPool3d((avgpool_t, avgpool_d, avgpool_d))
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(256, num_labels)
        
        ### re-initialize fc_out biases
        # idea: initially, the output of the FC layer with Softmax has to (roughly) match
        # the empirical share of hits and serves and others, etc.
        # to do this, we set the first bias (for "other") to zero.
        # all other biases are the same -> topspin backhand groundstroke same as serve
        # given share_other, what do the other biases need to equal?
        bias_event = np.log(1 - share_other) - np.log(share_other) - np.log(num_labels - 1)
        bias_other = 0
        biases = [bias_other, *[bias_event] * (num_labels - 1)]
        biases = nn.Parameter(torch.Tensor(biases))
        self.fc_out.bias = biases
        
        # self.softmax = nn.Softmax(dim = 1) ### dim = 1 --> the label dimension
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        
    def forward(self, x, verbose = False):
        if verbose: print(f'Shape of Input:             {x.shape}')
        
        y1 = self.conv1(x)
        if verbose: print(f'  ... after conv1:          {y1.shape}')
        
        y2 = self.conv2_2(self.conv2_1(y1))
        if verbose: print(f'  ... after conv2:          {y2.shape}')
            
        y3 = self.conv3_2(self.conv3_1(y2))
        if verbose: print(f'  ... after conv3:          {y3.shape}')
            
        y4 = self.conv4_2(self.conv4_1(y3))
        if verbose: print(f'  ... after conv4:          {y4.shape}')
        
        ypool = self.global_avgpool(y4)
        if verbose: print(f'  ... after global avgpool: {ypool.shape}')
        
        # z = self.softmax(self.fc_out(self.flatten(ypool)))
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        z = self.fc_out(self.flatten(ypool))
        if verbose: print(f'  ... of Output:            {z.shape}')
        
        return z
    

class OrigModel(nn.Module):
    
    def __init__(self, num_labels = 14, T = 32, D = 112):
        super(OrigModel, self).__init__()
        
        self.conv1 = TwoPlusOneBlock(3, 64, 7, 3, 
                                     stride_d = 2, stride_t = 1)
        
        self.conv2_1 = ResidualBlock(64, 64, 3, 3, 
                                     stride_d = 1, stride_t = 1)
        self.conv2_2 = ResidualBlock(64, 64, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv3_1 = ResidualBlock(64, 128, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv3_2 = ResidualBlock(128, 128, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv4_1 = ResidualBlock(128, 256, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv4_2 = ResidualBlock(256, 256, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv5_1 = ResidualBlock(256, 512, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv5_2 = ResidualBlock(512, 512, 3, 3,
                                     stride_d = 1, stride_t = 1)
  
        avgpool_t = math.ceil(T / (2 ** 3)) ### 3 times stride_t == 2
        avgpool_d = math.ceil(D / (2 ** 4)) ### 4 times stride_d == 2
    
        self.global_avgpool = nn.AvgPool3d((avgpool_t, avgpool_d, avgpool_d))
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(512, num_labels)
        # self.softmax = nn.Softmax(dim = 1) ### dim = 1 --> the label dimension
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        
    def forward(self, x, verbose = False):
        y1 = self.conv1(x)
        y2 = self.conv2_2(self.conv2_1(y1))
        y3 = self.conv3_2(self.conv3_1(y2))
        y4 = self.conv4_2(self.conv4_1(y3))
        y5 = self.conv5_2(self.conv5_1(y4))
        
        ypool = self.global_avgpool(y5)
        # z = self.softmax(self.fc_out(self.flatten(ypool)))
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        z = self.fc_out(self.flatten(ypool))
        
        if verbose:
            print(f'Shape of Input:             {x.shape}')
            print(f'  ... after conv1:          {y1.shape}')
            print(f'  ... after conv2:          {y2.shape}')
            print(f'  ... after conv3:          {y3.shape}')
            print(f'  ... after conv4:          {y4.shape}')
            print(f'  ... after conv5:          {y5.shape}')
            print(f'  ... after global avgpool: {ypool.shape}')
            print(f'  ... of Output:            {z.shape}')
        
        return z
    