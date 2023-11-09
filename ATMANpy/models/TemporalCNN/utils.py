# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:49:37 2023

@author: gianl
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn

from ATMANpy.funcs.util_funcs import split_into_contiguous
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import Stream, DecapStream
            

from itertools import product

def clip_to_stream(clip, zoom = 1):
    clipArr = clip.numpy()
    if not clipArr.dtype == 'uint8':
        clipArr = (clipArr * 255).astype('uint8')
    clipArr = np.moveaxis(clipArr, 0, -1) ### right now, shape is C, T, H, W --> we want T, H, W, C
    clipArr = np.flip(clipArr, -1) ### additionally, flip RGB to BGR
    clipFrms = BGRFrames(clipArr).zoom(zoom)
    clipStm = DecapStream(clipFrms, start_frame = 0, stop_frame = len(clipArr))
    return clipStm

def frames_to_stream(myFrms):
    return DecapStream(myFrms, start_frame = 0, stop_frame = myFrms.length())

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def infer_with_tbatch(model, loader, device = 'cpu', report_frequency = 10):
    ### Here, the dataloader yields 
        # an input array for the model (batch x label x T x H x W), where the batch dimension is redundant (batch_size = 1)
        # an array of central frame indices (batch x T)
        
    softmax = nn.Softmax(dim = 1)
    ### Keep batch size at 1! 
    ### Have the smallest possible batch size to make frames_in_sample as large as possible at constant memory requirements.
    prbArrs = []
    labArrs = []
    with torch.inference_mode():
        for idx, batchObj in enumerate(loader):
            if idx % report_frequency == 0: print(f'Inferred batch: {idx}')
            batchArr, labArr = batchObj
            
            batchArr = batchArr.to(device)
            outArr = model(batchArr)
            prbArr = softmax(outArr) ### batch x label x T x H x W
            prbArr = torch.swapaxes(prbArr, 0, 1) ### shape: label x batch x T x H x W
            prbArr = prbArr.flatten(1, 2)   ### merge dimensions batch and T
            prbArrs.append(prbArr)          ### --> label x T x H x W
            
            ### labArr 
            labArr = labArr.flatten(0, 1) ### merge dimensions batch and T
            labArrs.append(labArr)
            
    prbArr = torch.cat(prbArrs, dim = 1).cpu() ### concatenate on time dimension
    labArr = torch.cat(labArrs, dim = 0).cpu() ### append along time dimension
    return prbArr, labArr


def infer_with_index(model, loader, device = 'cpu', report_frequency = 10):
    ### Here, the dataloader yields 
        # an input array for the model (batch x label x T x H x W)
        # plus an index of frames (batch x T). e.g: idxArr[0] = [152, 154, 156, ..., 168] (if stride = 2) 
    
    softmax = nn.Softmax(dim = 1)
    ### Keep batch size at 1! 
    ### Have the smallest possible batch size to make frames_in_sample as large as possible at constant memory requirements.
    prbArrs = []
    labArrs = []
    with torch.inference_mode():
        for idx, batchObj in enumerate(loader):
            if idx % report_frequency == 0: print(f'Inferred batch: {idx}')
            batchArr, labArr = batchObj
      
            batchArr = batchArr.to(device)
            outArr = model(batchArr)
            prbArr = softmax(outArr) 
            prbArrs.append(prbArr)
            
            labArrs.append(labArr) 
            
    prbArr = torch.cat(prbArrs, dim = 0).cpu() ### concatenate on batch dimension
    if labArr is not None:
        labArr = torch.cat(labArrs, dim = 0).cpu() ### concatenate on batch dimension
    else:
        labArr = None
    
    return prbArr, labArr

def make_cat_frame(category_names, prbArr, index):
    ### Input shape: Labels x Samples x ... 
    npArr = prbArr.numpy()
    dims = len(npArr.shape)
    
    ### Gather unique categories to be aggregated
    unique_categories = sorted(set(category_names))
    
    ### some labelling functions
    replFunc_str = lambda myStr: 'serve' if myStr == 'any' else myStr
    def replFunc_tup(catTup):
        if len(catTup) == 1:
            l1 = catTup[0]
            return (replFunc_str(l1), )
        elif len(catTup) == 2:
            l1, l2 = catTup
            return (replFunc_str(l1), l2)

    ### if we have height and or width also, find lowest non-other (= event) probability
    if dims > 2: ### in this case, we have more than just labels and points in time
        other_idx = [idx for idx, c in enumerate(category_names) if c[0] == 'other' or c == 'other']
        assert len(other_idx) == 1
        other_idx = other_idx[0]
        othArr = npArr[other_idx]

        # do argmin over flattened othArr
        min_idx = othArr.reshape(othArr.shape[0], -1).argmin(1)    
        # Get unravel indices corresponding to original shape of A
        min_pos = np.column_stack(np.unravel_index(min_idx, othArr[0,:,:].shape))
            
    catSers = []
    for cat in unique_categories:
        cat_idxs = [idx for idx, c in enumerate(category_names) if c == cat]
        catArr = npArr[cat_idxs]
        catArr = catArr.sum(axis = 0) ### sum over contained labels to aggregate
        
        if dims > 2: ### in this case, we have more than just labels and points in time
            catArr_max = np.array([sample[idx[0], idx[1]] for sample, idx in zip(catArr, min_pos)]) ### terrible...
        else:
            catArr_max = catArr
    
        if isinstance(cat, tuple):
            replFunc = replFunc_tup
        else:
            replFunc = replFunc_str
            
        catSer = pd.Series(catArr_max, index = index, name = replFunc(cat))
        catSers.append(catSer)
    return pd.concat(catSers, axis = 1)

def plot_timing(at_frames, df, name, true_event_dict, grp_save_where,
                shape_x = 3, shape_y = 3, show_margin = 10,
                colors = {'serve': 'darkblue', 'forehand': 'firebrick', 'backhand': 'orange'},
                line_styles = {'left': 'dashed', 'right': 'solid'},
                marker_styles = {'left': '$L$', 'right': '$R$'},
                threshold = 0.2):
    serves_in_graph = shape_x * shape_y
    num_plots_done = 0
    
    fig, axes = plt.subplots(shape_x, shape_y, sharey = True, figsize = (10, 10))
    for idx, frm in enumerate(at_frames):
        lo, hi = frm - show_margin, frm + show_margin
        myFilter = (lo < df.index) & (df.index < hi)
        # frames_to_show = list(range(frm - show_margin, frm + show_margin))
        # frames_to_show = [f for f in frames_to_show if f in df.index]
        plotFrm = df[myFilter]
            
        num_plots_done, plot_modulus  = idx // serves_in_graph, idx % serves_in_graph
        if plot_modulus == 0 and idx > 0:
            fig.savefig(os.path.join(grp_save_where, f'{name}_{num_plots_done}.png'))
            fig, axes = plt.subplots(shape_x, shape_y, sharey = True, figsize = (10, 10))
        
        plot_idx = idx - serves_in_graph * num_plots_done
        ax_x, ax_y = plot_idx // shape_x, plot_idx % shape_y
        make_legend = ax_x == 0 and ax_y == 0
        
        my_ax = axes[ax_x, ax_y]
        for col in plotFrm:
            if isinstance(col, tuple):
                l1, l2 = col
                color = colors[l1]
                ls = line_styles[l2]    
                plotFrm[col].plot(ax = my_ax, color = color, ls = ls, legend = False)
            else:
                color = colors[col]
                plotFrm[col].plot(ax = my_ax, color = color, ls = 'solid', legend = False)
                
        my_ax.axhline(threshold, c = 'forestgreen', ls = 'dotted')
        
        ### scatter point for true label
        for lab, frms in true_event_dict.items():
            if isinstance(lab, tuple):
                l1, l2 = lab
                color = colors[l1]
                ms = marker_styles[l2]    
                my_ax.scatter(frms, [1 for _ in frms], c = color, marker = ms)
            else:
                color = colors[lab]
                my_ax.scatter(frms, [1 for _ in frms], c = color)
        
        ### limits x-axis
        my_ax.set_xlim(frm - show_margin, frm + show_margin)
        
        ### legend
        if make_legend:
            custom_labels = []
            custom_lines = []
            for evt, color in colors.items():
                custom_labels.append(evt)
                custom_lines.append(Line2D([0], [0], color=color, lw=4))
            
            my_ax.legend(custom_lines, custom_labels, loc = 4)
    
    fig.savefig(os.path.join(grp_save_where, f'{name}_{num_plots_done + 1}.png'))
    plt.close()

def split_and_group(inList, offset = 1):
    outList = split_into_contiguous(inList, offset = offset)
    outList = [int(np.median(evt)) for evt in outList]
    return outList

    



