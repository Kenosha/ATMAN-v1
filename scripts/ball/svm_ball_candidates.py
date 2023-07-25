# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:15:10 2022

@author: gianl
"""

import os

import pandas as pd
import numpy as np

import pickle
import datetime as dt

import ATMANpy.funcs.util_funcs as ut

### Path to labelled data
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'

ballDict = ut.get_folder_struct('ball_svm')
readPath  = os.path.join(ballDict['main'], VIDEO_ID, 'raw_data')
writePath = os.path.join(ballDict['main'], VIDEO_ID, 'fitted_svms')


try:
    os.mkdir(writePath)
except FileExistsError:
    print('   Adding to existing ball_svm data')


frmList = []
for labelled_csv in os.listdir(readPath):
    ballFrm = pd.read_csv(os.path.join(readPath, labelled_csv), index_col = 0)
    frmList.append(ballFrm)
    
rawFrm = pd.concat(frmList, ignore_index = True)
index_name = 'idx1'

# rawFrm['idx_within_image'] = rawFrm['idx_within_image'].fillna(rawFrm['index'])
# rawFrm = rawFrm.drop(['index', 'frameNum'], axis = 1)
rawFrm = rawFrm.drop(['frameNum'], axis = 1)

typeDict = {index_name: np.int64}
for col, tp in typeDict.items():
    rawFrm[col] = rawFrm[col].astype(tp)

balls_identified = rawFrm.isBall.sum()
non_balls = len(rawFrm) - balls_identified
print(f'Identified {balls_identified} balls and {non_balls} non-balls')


##### Setting Up SVM #######

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA

### Deal with missing data
nanIdxs = rawFrm.isna().any(axis = 1)
print(f'Deleting {nanIdxs.sum()} out of {len(nanIdxs)} observations, due to missing values.')
df = rawFrm[~nanIdxs]

### Downsampling
# skipped. Not necessary with small dataset

### Prepare Dep and Indep Variables
features = ['Alpha', 'SemiMaj', 'SemiMin', 'Cx', 'Cy', 'H', 'S', 'V']
depVar = 'isBall'

X = df[features]
y = df[depVar]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 17)

### extract means and std, to be able to normalize ball shapes later.
mean, std = X_train.mean(), X_train.std()
### scale
X_train = scale(X_train)
X_test = scale(X_test)

N = len(X_train)

###### Training SVC ###### 

param_grid = [{'C': [0.5, 1, 10, 100],
               'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
               'kernel': ['rbf']}]

# 1) Out of the Box
svm = SVC(random_state = 17)
svm.fit(X_train, y_train)

plot_confusion_matrix(svm, X_test, y_test, normalize = None, 
                      display_labels = ['Not Ball', 'Ball'])

plot_confusion_matrix(svm, X_test, y_test, normalize = 'true', 
                      display_labels = ['Not Ball', 'Ball'])

# 1.5) Optimised Out of the Box

optimal_params = GridSearchCV(SVC(class_weight = {1:5}), param_grid, 
                              cv = 5, scoring = 'accuracy')

optimal_params.fit(X_train, y_train)
print(f'Out of the Box SVM, best params: {optimal_params.best_params_}')

plot_confusion_matrix(optimal_params, X_test, y_test, normalize = None, 
                      display_labels = ['Not Ball', 'Ball'])

plot_confusion_matrix(optimal_params, X_test, y_test, normalize = 'true', 
                      display_labels = ['Not Ball', 'Ball'])


# 2) With more emphasis on correctly identifying balls
BALL_FACTOR = 5 

svm_weighted = SVC(random_state = 17, class_weight = {1: BALL_FACTOR})
svm_weighted.fit(X_train, y_train)

plot_confusion_matrix(svm_weighted, X_test, y_test, normalize = None, 
                      display_labels = ['Not Ball', 'Ball'])

plot_confusion_matrix(svm_weighted, X_test, y_test, normalize = 'true', 
                      display_labels = ['Not Ball', 'Ball'])


# 1.5) Optimised Weighted

optimal_params_weighted = GridSearchCV(SVC(class_weight = {1: BALL_FACTOR}), param_grid, 
                              cv = 5, scoring = 'accuracy')

optimal_params_weighted.fit(X_train, y_train)
print(f'Weighted SVM, best params: {optimal_params_weighted.best_params_}')

plot_confusion_matrix(optimal_params_weighted, X_test, y_test, normalize = None, 
                      display_labels = ['Not Ball', 'Ball'])

plot_confusion_matrix(optimal_params_weighted, X_test, y_test, normalize = 'true', 
                      display_labels = ['Not Ball', 'Ball'])


# 1.6a) Optimised Balance-Weighted

optimal_params_weighted_a = GridSearchCV(SVC(class_weight = 'balanced'), param_grid, 
                              cv = 5, scoring = 'accuracy')

optimal_params_weighted_a.fit(X_train, y_train)
print(f'Weighted SVM, best params: {optimal_params_weighted.best_params_}')

plot_confusion_matrix(optimal_params_weighted_a, X_test, y_test, normalize = None, 
                      display_labels = ['Not Ball', 'Ball'])

plot_confusion_matrix(optimal_params_weighted_a, X_test, y_test, normalize = 'true', 
                      display_labels = ['Not Ball', 'Ball'])


# 1.6a) Optimised Balance-Weighted with computed balanced weights

weightArr = N / (2 * np.bincount(y_train))
weightDict = dict(zip([0, 1], weightArr))

optimal_params_weighted_b = GridSearchCV(SVC(class_weight = weightDict), param_grid, 
                              cv = 5, scoring = 'accuracy')

optimal_params_weighted_b.fit(X_train, y_train)
print(f'Weighted SVM, best params: {optimal_params_weighted.best_params_}')

plot_confusion_matrix(optimal_params_weighted_b, X_test, y_test, normalize = None, 
                      display_labels = ['Not Ball', 'Ball'])

plot_confusion_matrix(optimal_params_weighted_b, X_test, y_test, normalize = 'true', 
                      display_labels = ['Not Ball', 'Ball'])

# optimal_params_weighted.predict([[1, 1, 1, 2, 3, 2, 1, 1]])

#### Pickle the best model ########

timestamp = dt.datetime.now().strftime('%Y-%m-%d')

### without probability prediction...
best_class_weight = optimal_params_weighted_b.estimator.class_weight
best_params = optimal_params_weighted_b.best_params_

best_SVC = SVC(class_weight = best_class_weight, **best_params)
best_SVC.fit(X_train, y_train)

fileName = f'svm_dict_{timestamp}.pickle'
savePath = os.path.join(writePath, fileName)

pickleDict = {'features': features,
              'mean': mean,
              'std': std,
              'svm': best_SVC}

with open(savePath, 'wb') as f:
    pickle.dump(pickleDict, f, pickle.HIGHEST_PROTOCOL)

### with probability prediction...

best_SVC = SVC(class_weight = best_class_weight, **best_params, probability = True)
best_SVC.fit(X_train, y_train)

fileName = f'svm_proba_dict_{timestamp}.pickle'
savePath = os.path.join(writePath, fileName)

pickleDict = {'features': features,
              'mean': mean,
              'std': std,
              'svm': best_SVC}

with open(savePath, 'wb') as f:
    pickle.dump(pickleDict, f, pickle.HIGHEST_PROTOCOL)


# FOR READ-IN
# with open(savePath, 'rb') as f:
#     clf = pickle.load(f)

# ##### PCA to understand classification ####

# pca = PCA()
# X_train_pca = pca.fit_transform(X_train)
# per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)

# per_var[:2].sum() ### not great... only 41% of variance is explained by two PCs.



