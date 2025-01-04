# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:10:00 2024

@author: gianl
"""

import ATMANpy.scripts.pipeline_brest_tightseg as ppl
import ATMANpy.funcs.pipeline_funcs as ppl_fs

import ATMANpy.funcs.util_funcs as ut

import os
import numpy as np
import cv2 as cv

if __name__ == '__main__':

    ONLY_DO = None
    FORCE_REDO = None

    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default    

    loadDict, saveDict = ppl_fs.get_load_and_save_dicts(ppl)
    
    ######## GENERAL PARAMETERS ########
    blacklist_vids = ['2021-7341-ms019-brest-stefano-travaglia-vs-mohamed-safwat', 
                      '2021-7341-ms026-brest-thomas-fabbiano-vs-alexey-vatutin', ### these have a different camera angle on a different court
                      ]
    
    vidfileDict = ut.get_folder_struct('video_files') 
    driver_scope = [f for f in vidfileDict if 'brest' in f and 'match.mp4' in os.listdir(vidfileDict[f]) and f not in blacklist_vids]
    
    general_features = [*saveDict['court_lines'], *saveDict['court_roi']]
    
    ##### MAKE VID AND STO FOLDERS #####
    for vid in driver_scope: 
        conf = cnf_mod.get_conf(vid)
        sto = conf['store_name']
        
        featPath = ut.get_folder_struct('features_stored')['main']
        vidPath = ut.maybe_make_folder(featPath, vid)
        stoPath = ut.maybe_make_folder(vidPath, sto)
        
        graphPath = ut.get_folder_struct('graphs')['pipeline_checks']
        pipePath = ut.maybe_make_folder(graphPath, ppl._name)
        stoPath = ut.maybe_make_folder(pipePath, sto)
    
    ###### MAKE GENERAL PARAMETERS ##########
    # ppl_fs.copy_general_features(general_features, driver_scope, sto)
    
    ###### RUN A SCRIPT FOR EACH VIDEO IN SCOPE #####
    for vid in driver_scope:
        print('________')
        print(vid)
        conf = cnf_mod.get_conf(vid)
        sto = conf['store_name']
    
        can_run, has_run = ppl_fs.can_run_has_run(loadDict, saveDict, vid, sto)
        should_run = [s for s in can_run if can_run[s] and not has_run[s]]
        
        #### FIGURE OUT WHAT TO RUN
        if FORCE_REDO is None:
            if len(should_run) == 0:
                print('NO SCRIPT CAN BE RUN!')
                script_to_run = None
            else:        
                print('SCRIPTS TO RUN:')
                for s in should_run:
                    print(f'   - {s}')

                if ONLY_DO is not None:
                    script_to_run = ONLY_DO if ONLY_DO in should_run else None
                else:
                    script_to_run = should_run[0]

        else:
            script_to_run = FORCE_REDO
            
        ###### RUN #############
        if script_to_run is not None:
                    
            print(f'RUNNING {script_to_run}:')
            
            script_module = getattr(ppl, script_to_run)
            
            try:
                parameters_object = getattr(prm_mod, script_to_run)
            except AttributeError:
                parameters_dictionary = {}
            else:
                parameters_dictionary = parameters_object.get_parameter_dictionary()
            
            for parName, parDict in parameters_dictionary.items():
                print(f'  with {parName}: ')
                for parStr, parVal in parDict.items():
                    print(f'    {parStr}: {parVal}')
            
            try:
                script_module.main(conf, **parameters_dictionary, name = ppl._name)
            except Exception as e:
                print(e)
                