# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:49:11 2023

TODO:
    - How to do split (should be deterministic) into training and testing
    - Transform label_frame into a certainty vector 
    - Transform clips (normalize at 0.5)

@author: gianl
"""

from ATMANpy.funcs.util_funcs import get_folder_struct, split_into_contiguous
from ATMANpy.models.TemporalCNN import clip_to_stream, label_frame_to_series
from ATMANpy.models.TemporalCNN import ClipDataset, SmallModel, SmallFullConvModel

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill as pickle

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

from itertools import product
from datetime import datetime


from torchmetrics import ConfusionMatrix
import sklearn.metrics
from ATMANpy.models.TemporalCNN import plot_confusion_matrix


### TODO: Data augmentation -> how?
### TODO: Change model weights for output of non-"other" categories (hits and serves)?
### TODO: Experiment: remove background from clips, keep only foreground shapes -> should make training much easier
### TODO: Experiment: change colors in clips more or less drastically depending on "color distance" to skin color.
            # --> skin color should be the only invariant; everything else is clothing, background, etc.

### TODO: Model weights or oversample or undersample the non-event clips?

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

DATA_PATH = get_folder_struct('video_files')['labelled_data']
DATASET_FOLDER = 'bot_player_labels_foremotion32'
DATASET_FOLDER_DENSE = 'bot_player_labels_foremotion32_highoverlap'

LOG_PATH = get_folder_struct('cnn')['runs']
RES_PATH = get_folder_struct('cnn')['weights']
PRM_PATH = get_folder_struct('cnn')['params']

MODEL_RUN_IDENTIFIER = 'small_fullconv_model_foremotion'

LOG_NAME = DATASET_FOLDER + '_' + MODEL_RUN_IDENTIFIER + '_' + timestamp
RES_NAME = DATASET_FOLDER + '_' + MODEL_RUN_IDENTIFIER + '_' + timestamp
PRM_NAME = DATASET_FOLDER + '_' + MODEL_RUN_IDENTIFIER + '_' + timestamp + '.pkl'

log_save_where = os.path.join(LOG_PATH, LOG_NAME)
res_save_where = os.path.join(RES_PATH, RES_NAME)
prm_save_where = os.path.join(PRM_PATH, PRM_NAME)


VID_INDICES = None #[0, 1, 2] ## None for all data!
D_CROP_SIZE = 112
T_CROP_SIZE = 16
D_SCALE_SIZE = 56
FLIP_PROB = 0.2
SHARE_TRAIN = 0.8
NUM_WORKERS = 4
BATCH_SIZE = 32

DETERMINISTIC = False
DEVICE = 'cuda'

EPOCHS = 40
LEARNING_RATE = 0.0003
MOMENTUM = 0.9

REPORT_FREQ = 100

TRAIN_DATA_INDEPENDENT_BASELINE = False
DO_TEST_PROXFUNC = False
DO_TENSORBOARD = True
DO_SHOW_CLIP = False
DO_TEST_COLOR_JITTER = False
DO_TEST_FORWARD = False
DO_TEST_BACKWARD = False
DO_CHECK_TIMING = False
DO_PLOT_CONF_MATRICES = False
DO_ANALYTICS = False

# MODEL_TO_READ = f'bot_player_labels_foremotion32_small_model_foremotion_20231101_132442'
# MODEL_TO_READ = f'bot_player_labels_foremotion32_small_model_foremotion_20231101_230538'
MODEL_TO_READ = None

SHARE_OTHER = None # 0.92 ### if you want a more precise estimate, set to None.
                            # if you want speed, set it to somewhere between 0.9 and 0.95, 
                            # which is good enough

def PROX_FUNC(impact_index, central_index):
    LO = 0.05
    HI = 0.95
    
    out = 1 - ((impact_index - central_index) / central_index) ** 2
    
    if out < LO:
        return 0
    elif LO <= out < HI:
        return out
    else:
        return 1
    
# def PROX_FUNC(impact_index, central_index):
#     ### this function is nice and sharp at the peak --> this way, we get good timing identification.
    
#     absdiff_floor = int(np.abs(impact_index - central_index)) ### absolute distance, rounded down 
    
#     return max(1 - np.log(absdiff_floor + 1) / 2, 0) 
    
def get_share_other(train_data):
    ### Find out how often there are hits and such in data, to be able
    # to initialize biases of the last fc layer in agreement with sample averages
    train_labels = []
    for vid, trainDF in train_data.label_frames.items():
        clip_indices = train_data.clip_indices[vid]
        relevantDF = trainDF.loc[clip_indices]
        vid_train_labels = relevantDF.groupby('clip').apply(label_frame_to_series, data)
        train_labels.append(vid_train_labels)
    label_means = pd.concat(train_labels).mean()
    share_other = label_means.loc[0]
    return share_other


#%% Setup

### Randomness (or the lack of it)
if DETERMINISTIC:
    torch.use_deterministic_algorithms(True, warn_only = True)
    ### CUDA Backends
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ### Main Process Seeds
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    ### DataLoader Subprocess Randomness
    g_worker = torch.Generator()
    g_worker.manual_seed(0)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
else:    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    seed_worker = None
    g_worker = None
        
if __name__ == '__main__':
    ### Prepare the Data     
    data = ClipDataset(DATA_PATH, DATASET_FOLDER, 
                       vid_indices = VID_INDICES, ### should be None for whole dataset
                       d_crop_size = D_CROP_SIZE, 
                       t_crop_size = T_CROP_SIZE,
                       d_scale_size = D_SCALE_SIZE,
                       flip_prob = FLIP_PROB)
    data.set_proximity_function(PROX_FUNC)    
    data.set_relabel_maps(hit_spin_map = {'topspin': 'topspin', 'slice': 'slice_or_flat', 'flat': 'slice_or_flat'},
                          serve_side_map = {'left': 'any', 'right': 'any'}) 

    train_data, test_data = data.train_test_split(share_train = SHARE_TRAIN)
    if TRAIN_DATA_INDEPENDENT_BASELINE:
        train_data.zero_out_clips = True ### set to true if you want to train a data independent baseline
        test_data.zero_out_clips = True
    
    ### Show a clip of the data, along with its label
    if DO_SHOW_CLIP:
        idx = 200 ### 101 has a hit, if you want to see that
        clip, label = data[idx]
        clipStm = clip_to_stream(clip)
        clipStm.show(ms_between_frames = 100)
        for lab, val in zip(data.get_all_labels(), label.numpy()):
            print(lab, val)
    
    ### Create Objects for Training Loop
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True,
                              num_workers = NUM_WORKERS, pin_memory = True,
                              worker_init_fn = seed_worker, generator = g_worker)
    
    if not test_data.vid_indices == []:
        test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False,
                                 num_workers = NUM_WORKERS, pin_memory = True,
                                 worker_init_fn = seed_worker, generator = g_worker)
    else:
        test_loader = None
        print('No test data: test data loader cannot be constructed!')
        
    ### Initialize model
    share_other = get_share_other(train_data) if SHARE_OTHER is None else SHARE_OTHER
    model = SmallFullConvModel(num_labels = len(data.get_all_labels()), 
                               T = T_CROP_SIZE, D = D_SCALE_SIZE,
                               share_other = share_other)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), 
    #                             lr = LEARNING_RATE, 
    #                             momentum = MOMENTUM)

    if DO_TEST_PROXFUNC:
        from matplotlib.ticker import MaxNLocator
        lab_frm_idxs = list(data.label_frames.values())[0].loc[0].index
        central_idx = np.mean(lab_frm_idxs)
        prox_vals = []
        for impact_idx in lab_frm_idxs:
            prox_val = data.prox_func(impact_idx, central_idx)
            prox_vals.append(prox_val)
        
        fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 5))
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        plotSer = pd.Series(prox_vals, index = lab_frm_idxs)
        plotSer.plot(ax = axes[0])
        plotSer[plotSer > 0].plot(ax = axes[1])
        
        
    ### Test forward pass
    if DO_TEST_FORWARD:
        with torch.inference_mode():
            clips, labels = next(iter(train_loader))
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(clips, verbose = True)
            print('Output Means: ')
            print(torch.mean(out, axis = 0))
            probs = nn.Softmax(dim = 1)(out)
            print('Probability Means: ')
            print(torch.mean(probs, axis = 0))
            loss = criterion(out[:, :, 0, 0, 0], labels)

    if DO_TENSORBOARD:
        ### Prepare TensorBoard
        writer = SummaryWriter(log_save_where)
        
        tb = program.TensorBoard()
        tb.configure(argv = [None, '--logdir', log_save_where])
        url = tb.launch()
        print(f'TensorBoard launched on {url}')
        
        clips, labels = next(iter(train_loader))
        clips = clips.to(DEVICE)
        writer.add_graph(model, clips)
        writer.flush()
        
        clips_large = tf.Resize(D_CROP_SIZE, antialias = True)(clips)
        clips_swap = torch.swapaxes(clips_large, 1, 2) ### convolutions need (N,C,T,H,W), add_video needs (N,T,C,H,W)
        writer.add_video('sample clip', clips_swap)
        writer.flush()
        
        if DO_TEST_COLOR_JITTER:
            idx = 200
            clip, label = data[idx]
            
            clip_swap = torch.swapaxes(clip, 0, 1) ### need T, C, H, W, not C, T, H, W
            
            jitFunc = tf.ColorJitter(brightness = 0.5, contrast = 0, saturation = 0, hue = 0)
            jittered_clips = [clip_swap]
            for i in range(31):
                jittered_clips.append(jitFunc(clip_swap))
                
            jittered_clips = torch.stack(jittered_clips)
            jittered_clips = tf.Resize(D_CROP_SIZE, antialias = True)(jittered_clips)
            
            clip_to_stream(torch.swapaxes(jittered_clips[15], 0, 1)).show(ms_between_frames = 100)
            
            writer.add_video('sample jitter', jittered_clips)
            writer.flush()
    
    #%%
    ### Training Loop
    
    if MODEL_TO_READ is None:
        
        ###### SAVE RELEVANT PARAMETERS IN A DICTIONARY ######
        params = {'VID_INDICES': VID_INDICES,
                  'D_CROP_SIZE': D_CROP_SIZE,
                  'T_CROP_SIZE': T_CROP_SIZE,
                  'D_SCALE_SIZE': D_SCALE_SIZE,
                  'FLIP_PROB': FLIP_PROB,
                  'SHARE_TRAIN': SHARE_TRAIN,
                  'NUM_WORKERS': NUM_WORKERS,
                  'BATCH_SIZE': BATCH_SIZE,
                  'DETERMINISTIC': DETERMINISTIC,
                  'DEVICE': DEVICE,
                  'EPOCHS': EPOCHS,
                  'LEARNING_RATE': LEARNING_RATE,
                  'MOMENTUM': MOMENTUM,
                  'REPORT_FREQ': REPORT_FREQ,
                  'SHARE_OTHER': SHARE_OTHER,
                  'PROX_FUNC': PROX_FUNC,
                  'LABELS': data.get_all_labels(),
                  }
        with open(prm_save_where, 'wb') as f:
            pickle.dump(params, f)
            print('Parameters saved to file.')
 
        
        def train_one_epoch(epoch_index, tb_writer):
            running_loss = 0
            last_loss = 0
            
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad(set_to_none = True)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
  
                running_loss += loss.item()
                
                if i % REPORT_FREQ == REPORT_FREQ - 1:
                    last_loss = running_loss / REPORT_FREQ
                    print(f'   batch {i + 1} --> loss: {last_loss}')
                    tb_x = (epoch_index - 1) * len(train_loader) + i + 1
                    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                    running_loss = 0
                    
            return last_loss
    
        best_vloss = 10 ** 6
        
        for epoch in range(1, EPOCHS + 1):
            print(f'EPOCH {epoch}')
            
            model.train(True)
            avg_loss = train_one_epoch(epoch, writer)
            model.train(False)
            
            if test_loader is None:
                writer.add_scalar('Training Loss', avg_loss, epoch)
                writer.flush()
                torch.save(model.state_dict(), res_save_where)
            else:
                with torch.inference_mode():
                    running_vloss = 0
                    vlabel_list = []
                    vprob_list = []
                    for i, vdata in enumerate(test_loader):
                        vinputs, vlabels = vdata
                        vinputs = vinputs.to(DEVICE)
                        vlabels = vlabels.to(DEVICE)
                        voutputs = model(vinputs).squeeze()
                        vloss = criterion(voutputs, vlabels)
                        running_vloss += vloss
                    
                        ### for confusion matrix
                        vprobs = nn.Softmax(dim = 1)(voutputs)
                        vprob_list.append(vprobs)
                        vlabel_list.append(vlabels)
            
                    avg_vloss = running_vloss / (i + 1)
                    
                    ### for confusion matrix
                    vprobs = torch.cat(vprob_list).to('cpu')
                    vlabels = torch.cat(vlabel_list).to('cpu')
                        
                    categories = [l[0] for l in test_data.get_all_labels()]
                    unique_categories = sorted(set(categories))
                    agg_probs = []
                    agg_labels = []
                    for cat in unique_categories:
                        cat_idxs = [idx for idx, c in enumerate(categories) if c == cat]
                        agg_prob = vprobs[:, cat_idxs].sum(axis = 1)
                        agg_label = vlabels[:, cat_idxs].sum(axis = 1)
                        
                        agg_probs.append(agg_prob)
                        agg_labels.append(agg_label)
                        
                    agg_probs = torch.stack(agg_probs, -1)
                    agg_labels = torch.stack(agg_labels, -1)
                    
                    chosen_label = torch.max(agg_probs, 1).indices
                    true_label = torch.argmax(agg_labels, 1)
                    
                    conf_func = ConfusionMatrix(task="multiclass", num_classes = 4)
                    confmat_agg = conf_func(chosen_label, true_label).numpy()
                    
                    confplot = plot_confusion_matrix(confmat_agg, unique_categories)
                    
                    confPath = os.path.join(log_save_where, 'confmats')
                    try:
                        os.mkdir(confPath)
                    except FileExistsError:
                        pass
                    finally:
                        confplot.savefig(os.path.join(confPath, f'confmat_{epoch}'))
                    
                writer.add_figure('Confusion Matrix', confplot, epoch)
                writer.flush()
                
                writer.add_scalars('Training vs. Validation Loss',
                                    {'Training': avg_loss, 'Validation': avg_vloss},
                                    epoch)
                writer.flush()
                
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    torch.save(model.state_dict(), res_save_where)
        
    else:
        res_read_where = os.path.join(RES_PATH, MODEL_TO_READ)
        model.load_state_dict(torch.load(res_read_where))
        model.train(False)

#%% 

    def aggregate_probs_and_labels(vprobs, vlabels, label_names, category_names):
        ### get a more aggregate confusion matrix -> other, forehand, backhand, serve
        unique_categories = sorted(set(category_names))
        agg_probs = []
        agg_labels = []
        for cat in unique_categories:
            cat_idxs = [idx for idx, c in enumerate(category_names) if c == cat]
            agg_prob = vprobs[:, cat_idxs].sum(axis = 1)
            agg_label = vlabels[:, cat_idxs].sum(axis = 1)
            
            agg_probs.append(agg_prob)
            agg_labels.append(agg_label)
            
        agg_probs = torch.stack(agg_probs, -1)
        agg_labels = torch.stack(agg_labels, -1)
        return unique_categories, agg_probs, agg_labels

    assert not model.training
        
    if DATASET_FOLDER_DENSE is not None:
        
        ### Prepare the Dense Data --> it has high overlap, so we can assess the quality of timing.     
        dense_data = ClipDataset(DATA_PATH, DATASET_FOLDER_DENSE, 
                                 vid_indices = VID_INDICES, ### should be None for whole dataset
                                 d_crop_size = D_CROP_SIZE, 
                                 t_crop_size = T_CROP_SIZE,
                                 d_scale_size = D_SCALE_SIZE,
                                 flip_prob = 0)
        dense_data.set_proximity_function(PROX_FUNC)    
        dense_data.set_relabel_maps(hit_spin_map = {'topspin': 'topspin', 'slice': 'slice_or_flat', 'flat': 'slice_or_flat'},
                                    serve_side_map = {'left': 'any', 'right': 'any'}) 
    
        my_data_loader = DataLoader(dense_data, batch_size = BATCH_SIZE, shuffle = False,
                                    num_workers = NUM_WORKERS, pin_memory = True,
                                    worker_init_fn = seed_worker, generator = g_worker)
    else:
        my_data_loader = test_loader

    if MODEL_TO_READ is not None:
        
        labels = test_data.get_all_labels()
        
        with torch.inference_mode():
            vlabel_list = []
            vprob_list = []
            for i, vdata in enumerate(my_data_loader):
                if i % 100 == 0: print(i)
                vinputs, vlabels = vdata
                vinputs = vinputs.to(DEVICE)
                vlabels = vlabels.to(DEVICE)
                voutputs = model(vinputs)
                vprobs = nn.Softmax(dim = 1)(voutputs)
                
                vprob_list.append(vprobs)
                vlabel_list.append(vlabels)
            
        vprobs = torch.cat(vprob_list).to('cpu')
        vlabels = torch.cat(vlabel_list).to('cpu')
        
        chosen_label = torch.max(vprobs, 1).indices
        true_label = torch.argmax(vlabels, 1)
        

    if DO_PLOT_CONF_MATRICES:
        ### get an overall confusion matrix
        conf_func = ConfusionMatrix(task="multiclass", num_classes = len(labels))
        confmat = conf_func(chosen_label, true_label).numpy()
        confFrm = pd.DataFrame(confmat, 
                               index = labels, 
                               columns = labels)
    
        
        ### get a more aggregate confusion matrix -> other, forehand, backhand, serve
        categories = [l[0] for l in labels]
        unique_categories, agg_probs, agg_labels = aggregate_probs_and_labels(vprobs, vlabels, labels, categories)
            
        chosen_label = torch.max(agg_probs, 1).indices
        true_label = torch.argmax(agg_labels, 1)
        
        conf_func = ConfusionMatrix(task="multiclass", num_classes = len(unique_categories))
        confmat_agg = conf_func(chosen_label, true_label).numpy()
        
        a = plot_confusion_matrix(confmat_agg, unique_categories)
            
        ### get a left-right hand confusion matrix
        categories = [('other',), *[(l[0], l[1]) for l in labels[1:]]]
        unique_categories, agg_probs, agg_labels = aggregate_probs_and_labels(vprobs, vlabels, labels, categories)
        
        chosen_label = torch.max(agg_probs, 1).indices
        true_label = torch.argmax(agg_labels, 1)
        
        conf_func = ConfusionMatrix(task="multiclass", num_classes = len(unique_categories))
        confmat_agg = conf_func(chosen_label, true_label).numpy()
        
        b = plot_confusion_matrix(confmat_agg, unique_categories)
           
    if DO_CHECK_TIMING:
        start, stop = 380, 420
        start, stop = 300, 700
        start, stop = 0, 1000
        # start, stop = 300, 350
        
        not_other_probs = 1 - vprobs[start:stop, 0].numpy()
        not_other_labels = 1 - vlabels[start:stop, 0].numpy()
        not_other = np.vstack([not_other_probs, not_other_labels]).T
        
        plot_frame = pd.DataFrame(not_other, columns = ['probs', 'labels'], index = range(start, stop))
        plot_frame.plot(figsize = (10, 10), style = {'probs': 'gx', 'labels': 'bo'})
        
        clip_to_stream(dense_data[325][0]).show(ms_between_frames = 100)
        
        ### 1) Check timing of serves.
        categories = [l[0] for l in labels]
        unique_categories, agg_probs, agg_labels = aggregate_probs_and_labels(vprobs, vlabels, labels, categories)
        
        start, stop = 7950, 8100
        start, stop = 5000, 10000
        
        serve_probs = agg_probs[start:stop, 0].numpy()
        serve_labels = agg_labels[start:stop, 0].numpy()
        serve = np.vstack([serve_probs, serve_labels]).T
        
        plot_frame = pd.DataFrame(serve, columns = ['probs', 'labels'], index = range(start, stop))
        plot_frame.plot(figsize = (10, 10), style = {'probs': 'gx', 'labels': 'bo'})
        
        ### serve segmentation with a simple threshold filter
        start, stop = None, None
        serve_probs = agg_probs[start:stop, 0].numpy()
        serve_labels = agg_labels[start:stop, 0].numpy()
        serve = np.vstack([serve_probs, serve_labels]).T
        
        serve_frame = pd.DataFrame(serve, columns = ['probs', 'labels'], index = range(len(serve)))
        
        serve_impact = serve_frame['labels'] > 0.9
        num_serves = len(split_into_contiguous((serve_frame[serve_impact]).index.tolist()))
        
        thresholds = np.linspace(0, 1, 101)
        true_positives = []
        false_positives = []
        false_negatives = []
        
        for t in thresholds:
            t_frame = serve_frame[serve_frame > t].dropna(how = 'all')
            serve_groups = split_into_contiguous(t_frame.index.tolist())
            
            true_positive = false_positive = false_negative = 0
            for g in serve_groups:
                g_frame = t_frame.loc[g]
                g_frame = g_frame.dropna(axis = 1, how = 'all')
                
                pred = 'probs' in g_frame
                true = 'labels' in g_frame
                
                true_positive += true and pred
                false_positive += not true and pred
                false_negative += true and not pred
                
            true_positives.append(true_positive)
            false_positives.append(false_positive)
            false_negatives.append(false_negative)
            
        fp = pd.Series(false_positives, name = 'false positives', index = thresholds)
        fn = pd.Series(false_negatives, name = 'false negatives', index = thresholds)
        tp = pd.Series(true_positives, name = 'true positives', index = thresholds) 
        plot_frame = pd.concat([fp, fn, tp], axis = 1)
        plot_frame.plot(color = {'true positives': 'forestgreen', 'false positives': 'magenta', 'false negatives': 'firebrick'})
        
        optimal = (tp == num_serves) & (fp == 0) & (fn == 0)
        optimal_thresholds = optimal[optimal].index.tolist()
        mn, mx = min(optimal_thresholds), max(optimal_thresholds)
        print(f'Optimal segmentation from t = {mn} to {mx}')
        
    if DO_ANALYTICS:
        ### Understand if the predicted probabilities of an event correlate with real "degrees"    
        other_y = torch.stack([vlabels[:, 0], vprobs[:, 0]], dim = 1)
        other_frame = pd.DataFrame(other_y.numpy(), columns = ['labels', 'probs'])
        corr = other_frame.corr().loc['labels', 'probs']
        print(f'Correlation between labels and predicted probabilities: {round(corr, 3)}')
        
        other_frame['del'] = np.abs(other_frame['labels'] - other_frame['probs'])
        print(other_frame.sort_values('del', ascending = False)[:15])
        
        ### the first six misses are complete hits that the model completely fails to recognize
        #3968 --> hit on the run near the net
        #2633 --> hit while moving
        #1743 --> hit with leg moving strange
        #2181 --> open stance hit with legs slightly loose
            # ... should be ok but is not well recognized 
            # --> only 4.4% chance that this is indeed a groundstroke...
        #86 --> forehand hit mid-stride
        #4395 --> half hearted slice after point is over already
        #3100 --> open stance backhand with leg movement
        #224 --> virtually identical to 3100
        #1346 --> half-hearted drop shot, model gives 1% chance of topspin groundstroke
        
        # 3817 --> not a hit, but looks a bit like a running forehand: model indeed says 87%.
        # 4240 --> not a hit, but looks like a backhand: model says either backhand slice or groundstroke
        
        idx = 1532
        clip, label = test_data[idx]
        clip_to_stream(clip, zoom = 4).show(ms_between_frames = 100)
        
        pd.Series(vprobs[idx].numpy(), index = test_data.get_all_labels()).sort_values(ascending = False)
        
        ### Check if the model recognizes serves well
        serve_y = torch.stack([vlabels[:, 13], vprobs[:, 13]], dim = 1)
        serve_frame = pd.DataFrame(serve_y.numpy(), columns = ['labels', 'probs'])
        serve_frame['del'] = np.abs(serve_frame['labels'] - serve_frame['probs'])
        print(serve_frame.sort_values('del', ascending = False)[:15])
        
        # 2324, 2325 --> the model sees a serve because 
            # the hits are very near at baseline, slightly off center! 
            # --> where player serves!
        idx = 2251
        clip, label = test_data[idx]
        clip_to_stream(clip, zoom = 4).show(ms_between_frames = 100)
        
        pd.Series(vprobs[idx].numpy(), index = test_data.get_all_labels()).sort_values(ascending = False)
        
        ### does the model get timing of serves right?
        serve_frame[serve_frame['labels'] == 1]
    
    ### Understand if the model manages to get the timing of hits right.
    
#%%
    # ### In Karpathy "recipe": Try to do "use backprop to chart dependencies".
    # # Question: How to get gradient with respect to input after backward pass?
    # if DO_TEST_BACKWARD:
    #     def dummy_criterion(outputs, labels):
    #         return outputs[0].sum()
    
        
    #     vdata = next(iter(test_loader))
    #     vinputs, vlabels = vdata
    #     vinputs.requires_grad_(True)
    #     vinputs = vinputs.to(DEVICE)
    #     vlabels = vlabels.to(DEVICE)
        
    #     voutputs = model(vinputs)
        
    #     loss = dummy_criterion(voutputs, vlabels)
    #     loss.backward()
        
    #     assert (vinputs.grad[0] != 0).any()
    #     assert (vinputs.grad[1:] == 0).all()
    
    # with torch.inference_mode():
    #     tdata = next(iter(train_loader))
    #     tinputs, tlabels = tdata
        
    #     tinputs = tinputs.to(DEVICE)
    #     tlabels = tlabels.to(DEVICE)
        
    #     toutputs = model(tinputs)
    #     tprobs = nn.Softmax(dim = 1)(toutputs)
        
    # clip_to_stream(tinputs[0]).show(ms_between_frames = 100)
    # clip_to_stream(tinputs[1]).show(ms_between_frames = 100)

    
    # ### How to calculate accuracy, precision, recall, F1?
    # # accuracy: (true p + true n) / (true p + true n + false p + false n)
    # ## in our case, a negative -> other! a positive: some event
    
    
    






