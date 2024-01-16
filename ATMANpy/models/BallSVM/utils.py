# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:39:57 2024

@author: gianl
"""

import os
from ATMANpy.funcs.util_funcs import get_folder_struct
import pickle

def load_svm(svm_name):
    ### Load SVM for Inference ###
    svmFolder = get_folder_struct('ball_svm')['svm_brest']    
    svmPath = os.path.join(svmFolder, 'weights', svm_name + '.pickle')

    with open(svmPath, 'rb') as f:
        svmDict = pickle.load(f)
        
    return svmDict
        