# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:36:06 2024

@author: gianl
"""

import os
import inspect

import pandas as pd
import numpy as np
 
import networkx as nx
import matplotlib.pyplot as plt

import ATMANpy.funcs.util_funcs as ut


def rhs_parser(rhs):
    obj_class = rhs.split('.')[0]
    if obj_class not in ['ff']:
        in_brackets = rhs.replace(f'{obj_class}.from_store(', '').replace(')', '')
    else:
        in_brackets = rhs.replace(f'{obj_class}.read_feature_frame(', '').replace(')', '')
    
    if obj_class == 'StreamList':
        argStr = in_brackets.split(',')[0]
    
    elif 'Features' in obj_class:
        argStr = in_brackets.split(',', 2)[2].replace('featnameList=','')
        argStr = argStr.replace('[', '').replace(']', '')
        
    elif 'Feature' in obj_class:
        argStr = in_brackets.split(',')[2].replace('featname=', '')

    elif obj_class == 'ff':
        argStr = in_brackets.split(',')[0]

    argStr = argStr.replace('\'', '')
    argStrs = argStr.split(',')
    return argStrs

def read_load_arguments(someFunc):
    src, num_lines = inspect.getsourcelines(someFunc)
    src = [ln.strip(' ').rstrip('\n') for ln in src] ### strip whitespace and linebreaks
    src = [ln.split('#')[0] for ln in src if not ln.split('#')[0] == ''] ### strip comments
    
    ret = [ln for ln in src if ln.startswith('return ')]
    if len(ret) == 0:
        return []
    else:
        assert len(ret) == 1
    
    ret = ret[0]
    
    out_names = []
    ret_strings = ret.replace('return', '').replace(' ', '').split(',')
    for ret_str in ret_strings:
        def_line = [l for l in src if l.replace(' ', '').startswith(ret_str + '=')]
        assert len(def_line) == 1
        def_line = def_line[0]
        def_line = def_line.replace(' ', '')
    
        rhs = def_line.split('=', 1)[1]
        arg_names = rhs_parser(rhs)
        out_names.extend(arg_names)
        
    return out_names

def read_save_arguments(someFunc):
    src, num_lines = inspect.getsourcelines(someFunc)
    src = [ln.strip(' ').rstrip('\n') for ln in src] ### strip whitespace and linebreaks
    src = [ln.split('#')[0] for ln in src] ### strip comments

    src_body = src[1:]
    
    
    save_lines = [l for l in src_body if 'save' in l]
    save_lines = [l.replace(' ', '') for l in save_lines]
    # assert len(src_body) == len(save_lines)

    out_names = []
    for l in save_lines:
        if 'rename' in l:
            nameStr = l.split('.')[1].replace('rename(', '').replace(')', '')
        elif 'save_feature_frame' in l:
            in_brackets = l.split('(')[1].replace(')', '')
            nameStr = in_brackets.split(',')[1]
        elif 'ConstFeature' in l:
            callStr = l.split('.', 1)[0]
            in_brackets = callStr.replace('ConstFeature(', '').replace(')', '')
            nameStr = in_brackets.split(',', 1)[0]
            
        nameStr = nameStr.replace('\'', '')
        out_names.append(nameStr)
        
    return out_names

def get_load_and_save_dicts(ppl):
    loadDict = {}
    saveDict = {}    
    for script in ppl._scripts:
        script_name = script[:-3]
        loadFunc = getattr(getattr(ppl, script_name), 'load')
        saveFunc = getattr(getattr(ppl, script_name), 'save')
        loadDict[script_name] = read_load_arguments(loadFunc)
        saveDict[script_name] = read_save_arguments(saveFunc)
    return loadDict, saveDict
    
def get_dependency_dict(loadDict, saveDict):    
    # a) first, invert the saveDict!
    inv_saveDict = {}
    for script_name, saved_features in saveDict.items():
        for feat in saved_features:
            inv_saveDict[feat] = script_name
    
    # b) replace loaded features with dependencies
    depDict = {}
    for script_name, loaded_features in loadDict.items():
        depDict[script_name] = [inv_saveDict[feat] for feat in loaded_features]

    return depDict

def get_sequential_ordering(depDict):
    orderDict = {} ### script: seq_order
    
    while True:
        add_orderDict = {}
        current_lat_order = 0
        for script_name, dependencies in depDict.items():
            if script_name in orderDict:
                continue
            
            try:
                pred_ords = [orderDict[dep] for dep in dependencies]
            except:
                continue
            else:
                if pred_ords == []:
                    seq_order = 0
                else:
                    seq_order = max(pred_ords) + 1
                    current_lat_order += 1
                add_orderDict[script_name] = seq_order
    
        if add_orderDict == {}:
            break
        else:
            orderDict = {**orderDict, **add_orderDict}
    
    return orderDict

def get_lateral_ordering(orderDict, depDict):
   
    lateralDict = {} ### script: lat_order
    
    seqDicts = [{script: order for script, order in orderDict.items() if order == curr_order} for curr_order in sorted(set(orderDict.values()))]
    for seqDict in seqDicts:
        mean_lat_val_dict = {}
        for script in seqDict:
            dependencies = depDict[script]
            if dependencies == []:
                mean_lat_val = 0
            else:
                deps_lat_val = [lateralDict[dep] for dep in dependencies]
                mean_lat_val = sum(deps_lat_val) / len(deps_lat_val)
            
            mean_lat_val_dict[script] = mean_lat_val
        
        ### normalize lateral values -> be between -1 and 1, equidistant
        same_lat_points = len(mean_lat_val_dict)
        
        if same_lat_points == 1:
            mean_lat_val_ser = pd.Series(mean_lat_val_dict)
            rank_lat_val_ser = pd.Series({s: 0 for s in mean_lat_val_dict})
        else:
            norm_indices = np.linspace(-1, 1, num = same_lat_points, endpoint = True)
            
            mean_lat_val_ser = pd.Series(mean_lat_val_dict).sort_values(ascending = True)
            rank_lat_val_ser = pd.Series(norm_indices, index = mean_lat_val_ser.index)
        
        norm_lat_val_ser = mean_lat_val_ser * 0.5 + rank_lat_val_ser * 0.5
        norm_lat_val_dict = norm_lat_val_ser.to_dict()
        
        lateralDict = {**lateralDict, **norm_lat_val_dict}
           
    return lateralDict
      
def make_digraph(depDict, posDict, saveName = None):
    
    color_map = {'court': 'forestgreen',
                 'ball': 'gold',
                 'player': 'salmon',
                 'segmentation': 'cornflowerblue',
                 'hits': 'violet',
                 'manual': 'crimson',
                 'visualization': 'turquoise',
                 'default': 'grey'}
    
    saveDir = ut.get_folder_struct('graphs')['pipeline_dependencies']
    if saveName is None:
        saveName = 'pipeline'
    savePath = os.path.join(saveDir, saveName + '.png')
    
    G = nx.DiGraph()
    
    for script_name in depDict:
        G.add_node(script_name)
    
    for script_name, dependencies in depDict.items():
        for dep_name in dependencies:
            G.add_edge(dep_name, script_name)
        
    ### figure out colors
    colors = []
    for script in list(G):
        prefix = script.split('_')[0]
        try:
            script_color = color_map[prefix]
        except KeyError:
            script_color = color_map['default']
        colors.append(script_color)
    
    ### figure out labels
    labels = {lab: lab.replace('_', '\n') for lab in list(G)}
        
    fig, ax = plt.subplots(1, 1, figsize = (50, 50))
    fig.tight_layout()
    ax.set_facecolor('black')
    fig.set_facecolor('black')
    
    nx.draw(G, pos = posDict, ax = ax, with_labels = True, 
            node_size = 24000, node_color = colors,
            labels = labels, font_weight = 'bold', font_size = 18)
    fig.savefig(savePath)
    plt.close()

def can_run_has_run(loadDict, saveDict, vid, sto):
    ### if someDict = loadDict -> we are checking if we can run the scripts
    ### if someDict = saveDict -> we are checking if we have run the scripts

    storePath = os.path.join(ut.get_folder_struct('features_stored')[vid], sto)
    available_features = os.listdir(storePath)
    
    can_run = {script: all([necessary_feature in available_features for necessary_feature in loadDict[script]]) for script in loadDict}
    has_run = {script: all([necessary_feature in available_features for necessary_feature in saveDict[script]]) for script in saveDict}
    return can_run, has_run

def copy_general_features(general_features, driver_scope, store_name):
    import shutil

    ### if a set of feature is the same across all videos within the driver_scope,
    ### we copy it from one store to all stores within the driver_scope,
    ### so that we don't have to re-run it senselessly.
    storeDict = ut.get_folder_struct('features_stored')
    
    ### Search for features to be copied
    foundDict = {f: None for f in general_features}
    for script in driver_scope:
        store_path = os.path.join(storeDict[script], store_name)
        features_still_needed = [f for f in general_features if foundDict[f] is None]

        features_in_store = os.listdir(store_path)
        available_features_still_needed = [f for f in features_still_needed if f in features_in_store]
        for f in available_features_still_needed:
            foundDict[f] = store_path
    
    ### Copy over features to all other files
    for feature, found_where in foundDict.items():
        if found_where is None:
            print(f'Could not find {feature} anywhere, did not copy')
        else:
            file_path_from = os.path.join(found_where, feature)
            for script in driver_scope:
                store_path = os.path.join(storeDict[script], store_name)
                file_path_to = os.path.join(store_path, feature)
        
                try:
                    shutil.copytree(file_path_from, file_path_to, dirs_exist_ok = False)
                except FileExistsError:
                    print(f'Did not copy {feature} to {script}: folder already exists')
                else:
                    print(f'Copied {feature} to {script}')
            
        
        
    
    












