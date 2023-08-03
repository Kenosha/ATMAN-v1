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
import ATMANpy.funcs.extract_funcs as ex

from ATMANpy.types.Features import StreamFeatures
from ATMANpy.types.Streams import NCStream, MulStream

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA


if __name__ == '__main__':
    
    ####### READ IN LABELLED DATA ###########
    conf = ut.read_config('brest')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    cntSF = StreamFeatures.from_store(vid, sto, featnameList = ['svm_labelled_ballCnts', 'svm_labelled_nonballCnts'])
    
    greyStm = NCStream.from_SF(cntSF, cap, target_color_space = 'grey')
    hsvStm = NCStream.from_SF(cntSF, cap, target_color_space = 'hsv')
    
    mulStm_balls = MulStream([greyStm, hsvStm]).attach(cntSF['svm_labelled_ballCnts'].rename('movCnts'))
    mulStm_other = MulStream([greyStm, hsvStm]).attach(cntSF['svm_labelled_nonballCnts'].rename('movCnts'))
    
    params = {'zoom_factor': round(10 / conf['scale_factor']), 
              'pixel_margin': max(round(3 * conf['scale_factor']), 1),
              'ksize': 3}
    frmSF_balls = mulStm_balls.extract_mp(ex.get_feature_frames, nwo, **params)   
    frmSF_other = mulStm_other.extract_mp(ex.get_feature_frames, nwo, **params)  
    
    ballsDF = frmSF_balls.to_frame()
    otherDF = frmSF_other.to_frame()
    
    ballsDF['isBall'] = 1
    otherDF['isBall'] = 0
    
    print(f'Identified {len(ballsDF)} balls and {len(otherDF)} non-balls')
    
    rawDF = pd.concat([ballsDF.reset_index(drop = True),
                       otherDF.reset_index(drop = True)])
    
    ###########################################
    
    ###### SORT OUT WRITE PATH ################
    ballDict = ut.get_folder_struct('ball_svm')
    writePath = os.path.join(ballDict['main'], vid, 'fitted_svms')
    
    try:
        os.mkdir(writePath)
    except FileExistsError:
        print('   Adding to existing ball_svm data')
    
    ###########################################
    
    ####### SET UP TRAIN AND TEST ARRAYS ######
    
    ### Prepare Dep and Indep Variables
    features = ['Alpha', 'SemiMaj', 'SemiMin', 'Cx', 'Cy', 'H', 'S', 'V']
    depVar = 'isBall'
    
    X = rawDF[features]
    y = rawDF[depVar]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 17)
    
    ### extract means and std, to be able to normalize ball shapes later.
    mean, std = X_train.mean(), X_train.std()
    ### scale
    X_train = scale(X_train)
    X_test = scale(X_test)
    
    N = len(X_train)
    
    ########### TRAIN SVC ############## 
    
    def cv_scorer(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
    
        try:
            precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) ### true positives to all positives
            recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) ### true positives to true balls
        except:
            return 0
        
        lowest_acceptable_precision = 0.35 ### what precision is "acceptable"
        lowest_acceptable_recall = 0.8     ### what recall is equally "acceptable"
        
        xs_precision = max(precision - lowest_acceptable_precision, 0)
        xs_recall = max(recall - lowest_acceptable_recall, 0)
        
        xs_precision_rate = xs_precision / (1 - lowest_acceptable_precision)
        xs_recall_rate = xs_recall / (1 - lowest_acceptable_recall)
        
        return xs_precision_rate * xs_recall_rate
        
    ### a) Fixed Params ###
    params = {'class_weight': {1: 8},
              'C': 10,
              'gamma': 'scale',
              'kernel': 'rbf'}
    
    svc = SVC(random_state = 17, **params)
    svc.fit(X_train, y_train)
    
    plot_confusion_matrix(svc, X_test, y_test, normalize = None, 
                          display_labels = ['Not Ball', 'Ball'])
    
    plot_confusion_matrix(svc, X_test, y_test, normalize = 'true', 
                          display_labels = ['Not Ball', 'Ball'])
    
    
    #### b) Grid Search #####
    param_grid = [{'class_weight': [{1: 1}, {1: 4}, {1: 8}],
                   'C': [0.5, 1, 5, 10, 100],
                   'gamma': ['scale'],
                   'kernel': ['rbf']}]
    
    optimal_params = GridSearchCV(SVC(random_state = 17), param_grid, 
                                  cv = 5, scoring = cv_scorer)
    
    optimal_params.fit(X_train, y_train)
    print(f'Out of the Box SVM, best params: {optimal_params.best_params_}')
    
    plot_confusion_matrix(optimal_params, X_test, y_test, normalize = None, 
                          display_labels = ['Not Ball', 'Ball'])
    
    plot_confusion_matrix(optimal_params, X_test, y_test, normalize = 'true', 
                          display_labels = ['Not Ball', 'Ball'])
    
    
    #### Pickle the best model ########
    
    timestamp = dt.datetime.now().strftime('%Y-%m-%d')
    
    ### without probability prediction...
    best_class_weight = optimal_params.estimator.class_weight
    best_params = optimal_params.best_params_
    
    best_SVC = SVC(**best_params)
    best_SVC.fit(X_train, y_train)
    
    fileName = f'svm_dict_{timestamp}.pickle'
    savePath = os.path.join(writePath, fileName)
    
    pickleDict = {'feature_params': params, ### feature params, so we can use the same params for training and inference!
                  'features': features,
                  'mean': mean,
                  'std': std,
                  'svm': best_SVC}
    
    with open(savePath, 'wb') as f:
        pickle.dump(pickleDict, f, pickle.HIGHEST_PROTOCOL)
    
    ### with probability prediction...
    
    best_SVC = SVC(**best_params, probability = True)
    best_SVC.fit(X_train, y_train)
    
    fileName = f'svm_proba_dict_{timestamp}.pickle'
    savePath = os.path.join(writePath, fileName)
    
    pickleDict = {'feature_params': params,
                  'features': features,
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
    
    
    
