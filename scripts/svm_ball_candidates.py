# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:15:10 2022

@author: gianl
"""

import ATMANpy.utils as atu

import pandas as pd
import numpy as np

### Path to labelled data
VIDEO_ID = 'pqxAmUWIoKc'

import os
burnDict = atu.get_folder_struct('ball_svm')
readPath = os.path.join(burnDict['main'], VIDEO_ID)

frmList = []
for labelled_csv in os.listdir(readPath):
    ballFrm = pd.read_csv(os.path.join(readPath, labelled_csv), 
                          index_col = 0, dtype = {'idx_within_image': np.int64,
                                                  'index': np.int64,
                                                  'frameNum': np.int64})
    frmList.append(ballFrm)
    
rawFrm = pd.concat(frmList, ignore_index = True)
rawFrm['idx_within_image'] = rawFrm['idx_within_image'].fillna(rawFrm['index'])
rawFrm = rawFrm.drop('index', axis = 1)

balls_identified = rawFrm.isBall.sum()
non_balls = len(rawFrm) - balls_identified
print(f'Identified {balls_identified} balls and {non_balls} non-balls')