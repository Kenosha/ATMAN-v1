
import ATMANpy.scripts.pipeline_brest_tightseg as ppl
import ATMANpy.funcs.pipeline_funcs as ppl_fs

import ATMANpy.funcs.util_funcs as ut

import os
import numpy as np
import cv2 as cv

import shutil

if __name__ == '__main__':

    FEATURE_NAME = 'hitCnt'
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default    

    ######## GENERAL PARAMETERS ########
    blacklist_vids = ['2021-7341-ms019-brest-stefano-travaglia-vs-mohamed-safwat', 
                      '2021-7341-ms026-brest-thomas-fabbiano-vs-alexey-vatutin'] ### these have a different camera angle on a different court
    
    vidfileDict = ut.get_folder_struct('video_files') 
    driver_scope = [f for f in vidfileDict if 'brest' in f and 'match.mp4' in os.listdir(vidfileDict[f]) and f not in blacklist_vids]
    
    ##### MAKE VID AND STO FOLDERS #####
    for vid in driver_scope: 
        conf = cnf_mod.get_conf(vid)
        sto = conf['store_name']
        
        featPath = ut.get_folder_struct('features_stored')['main']
        vidPath = ut.maybe_make_folder(featPath, vid)
        stoPath = ut.maybe_make_folder(vidPath, sto)
        
        delFolder = os.path.join(stoPath, FEATURE_NAME)

        if os.path.isdir(delFolder):
            shutil.rmtree(delFolder)
            print(f'Removed {delFolder}')
        else:
            print(f'{delFolder} does not exist!')



