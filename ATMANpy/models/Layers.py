# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 20:41:54 2023

@author: gianl
"""

import numpy as np
from scipy import signal
from skimage.measure import block_reduce

class Layer:
    def __init__(self):
        self.input_array = None
        self.output_array = None
    
    def forward(self, input_array):
        pass
    
    def backward(self, output_gradient, learning_rate):
        pass
        
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input_array):
        self.input = input_array
        return self.activation(input_array)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)
          
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
    
        super().__init__(sigmoid, sigmoid_prime)
    
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_shape = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        
        input_depth, input_height, input_width = input_shape
        output_dim = input_height - kernel_size + 1
        
        self.input_depth = input_depth
        self.output_shape = (depth, output_dim, output_dim)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        
        ### Random Initialization
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        
    def forward(self, input_array):
        self.input_array = input_array
        self.output_array = np.copy(self.biases)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output_array[i] += signal.correlate2d(self.input_array[j], self.kernels[i, j], 'valid')
        return self.output_array
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input_array[j], output_gradient[i], 'valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], 'full') 
                
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        
        return input_gradient
    
class MaxPool(Layer):
    def __init__(self, kernel_size):
        ### kernel size ought to be an integer!
        self.kernel_size = kernel_size
        self.kernel_shape = (1, kernel_size, kernel_size) # don't pool over channels.
        
    def forward(self, input_array):
        self.input_array = input_array
        self.input_shape = input_array.shape
        self.output_array = block_reduce(input_array, block_size = self.kernel_shape, func = np.max)
        self.output_shape = self.output_array.shape
        return self.output_array
        
    def backward(self, output_gradient, learning_rate):
        ### First, calculate expanded matrix with maximum values, but input shape
        C, H, W = self.output_shape
        expanded_output_array = np.zeros(self.input_shape)
        expanded_gradient_array = np.zeros(self.input_shape)
        for c_idx in range(C):
            for h_idx in range(H):
                exp_h_idx = h_idx * self.kernel_size
                for w_idx in range(W):
                    exp_w_idx = w_idx * self.kernel_size
                    
                    val = self.output_array[c_idx, h_idx, w_idx]
                    grd = output_gradient[c_idx, h_idx, w_idx]
                    
                    for h_offset in range(self.kernel_size):
                        for w_offset in range(self.kernel_size):
                            coords = c_idx, exp_h_idx + h_offset, exp_w_idx + w_offset
                            
                            try:
                                expanded_output_array[coords] = val
                                expanded_gradient_array[coords] = grd
                            except IndexError:
                                pass
                        
        ### Then, match input_array and expanded_output_array, elementwise,
        ### if there is a match, the gradient that is passed in, is passed backwards.
        return (self.input_array == expanded_output_array) * expanded_gradient_array
    
class Flatten(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def forward(self, input_array):
        return input_array.flatten()
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
        
        
        
    
    