# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:49:11 2023

TODO:
    - How to do split (should be deterministic) into training and testing
    - Transform label_frame into a certainty vector 
    - Transform clips (normalize at 0.5)

@author: gianl
"""

from ATMANpy.funcs.util_funcs import get_folder_struct
from ATMANpy.models.TemporalCNN import clip_to_stream, label_frame_to_series
from ATMANpy.models.TemporalCNN import ClipDataset, SmallModel

import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

from itertools import product
from datetime import datetime

### TODO: Data augmentation -> how?
### TODO: Change model weights for output of non-"other" categories (hits and serves)?
### TODO: Experiment: remove background from clips, keep only foreground shapes -> should make training much easier
### TODO: Experiment: change colors in clips more or less drastically depending on "color distance" to skin color.
            # --> skin color should be the only invariant; everything else is clothing, background, etc.

### TODO: Overfit one batch, see if we can make train loss -> 0
### TODO: Model weights or oversample or undersample the non-event clips?

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

DATA_PATH = get_folder_struct('video_files')['labelled_data']
DATASET_FOLDER = 'bot_player_labels_sequential'

LOG_PATH = get_folder_struct('cnn')['runs']
RES_PATH = get_folder_struct('cnn')['weights']

MODEL_RUN_IDENTIFIER = 'small_model'

LOG_NAME = DATASET_FOLDER + '_' + MODEL_RUN_IDENTIFIER + '_' + timestamp
RES_NAME = DATASET_FOLDER + '_' + MODEL_RUN_IDENTIFIER + '_' + timestamp

log_save_where = os.path.join(LOG_PATH, LOG_NAME)
res_save_where = os.path.join(RES_PATH, RES_NAME)


VID_INDICES = None #[0, 1, 2] ## None for all data!
D_CROP_SIZE = 112
T_CROP_SIZE = 16
D_SCALE_SIZE = 56
SHARE_TRAIN = 0.8
NUM_WORKERS = 0
BATCH_SIZE = 32

DETERMINISTIC = False
DEVICE = 'cpu'

EPOCHS = 50
LEARNING_RATE = 0.0003
MOMENTUM = 0.9

REPORT_FREQ = 100

TRAIN_DATA_INDEPENDENT_BASELINE = False
DO_SHOW_CLIP = False
DO_TEST_FORWARD = False
DO_TEST_BACKWARD = False

MODEL_TO_READ = f'bot_player_labels_sequential_small_model_20231024_162630'
# MODEL_TO_READ = None

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
                       d_scale_size = D_SCALE_SIZE)
    data.set_proximity_function(PROX_FUNC)    
    data.set_relabel_maps(hit_spin_map = {'topspin': 'topspin', 'slice': 'slice_or_flat', 'flat': 'slice_or_flat'},
                          serve_side_map = {'left': 'any', 'right': 'any'}) 

    train_data, test_data = data.train_test_split(share_train = SHARE_TRAIN)
    if TRAIN_DATA_INDEPENDENT_BASELINE:
        train_data.zero_out_clips = True ### set to true if you want to train a data independent baseline
        test_data.zero_out_clips = True
    
    ### Show a clip of the data, along with its label
    if DO_SHOW_CLIP:
        idx = 101 ### 101 has a hit, if you want to see that
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
        test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True,
                                 num_workers = NUM_WORKERS, pin_memory = True,
                                 worker_init_fn = seed_worker, generator = g_worker)
    else:
        test_loader = None
        print('No test data: test data loader cannot be constructed!')
        
    ### Initialize model
    num_labels = len(data.get_all_labels())    
    share_other = get_share_other(train_data) if SHARE_OTHER is None else SHARE_OTHER
    model = SmallModel(num_labels = num_labels, 
                       T = T_CROP_SIZE, D = D_SCALE_SIZE,
                       share_other = share_other)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), 
    #                             lr = LEARNING_RATE, 
    #                             momentum = MOMENTUM)

    ### Test forward pass
    if DO_TEST_FORWARD:
        with torch.inference_mode():
            clips, labels = next(iter(train_loader))
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(clips, verbose = True)
            print('Output Means: ')
            print(torch.mean(out, axis = 0))

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
    
    clips_swap = torch.swapaxes(clips, 1, 2) ### convolutions need (N,C,T,H,W), add_video needs (N,T,C,H,W)
    writer.add_video('sample clip', clips_swap)
    
    
    #%%
    ### Training Loop
    
    if MODEL_TO_READ is None:
        def train_one_epoch(epoch_index, tb_writer):
            running_loss = 0
            last_loss = 0
            
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad(set_to_none = True)
                outputs = model(inputs)
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
                    for i, vdata in enumerate(test_loader):
                        vinputs, vlabels = vdata
                        vinputs = vinputs.to(DEVICE)
                        vlabels = vlabels.to(DEVICE)
                        voutputs = model(vinputs)
                        vloss = criterion(voutputs, vlabels)
                        running_vloss += vloss
                    
                    avg_vloss = running_vloss / (i + 1)
              
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
    from torchmetrics import ConfusionMatrix
    import sklearn.metrics
    
    from ATMANpy.models.TemporalCNN import plot_confusion_matrix
    
    assert not model.training
    
    with torch.inference_mode():
        vlabel_list = []
        vprob_list = []
        for i, vdata in enumerate(test_loader):
            print(i)
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
    
    ### get an overall confusion matrix
    conf_func = ConfusionMatrix(task="multiclass", num_classes = 14)
    confmat = conf_func(chosen_label, true_label).numpy()
    confFrm = pd.DataFrame(confmat.numpy(), 
                           index = test_data.get_all_labels(), 
                           columns = test_data.get_all_labels())

    
    ### get a more aggregate confusion matrix -> other, forehand, backhand, serve
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
    
    a = plot_confusion_matrix(confmat_agg, unique_categories)
    
    ### Understand if the predicted probabilities of an event correlate with real "degrees"
    vdata[1][2]
    
    for inp, lab in test_data:
        if lab[0] < 1:
            break
    test_data[5]
    
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
    
    
    






