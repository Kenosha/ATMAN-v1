# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:10:00 2024

@author: gianl
"""

import ATMANpy.scripts.pipeline_brest as ppl
import ATMANpy.funcs.pipeline_funcs as ppl_fs

import ATMANpy.funcs.util_funcs as ut

import numpy as np
import cv2 as cv

loadDict, saveDict = ppl_fs.get_load_and_save_dicts(ppl)

######## GENERAL PARAMETERS ########
START_FRAME, STOP_FRAME = 0, 10000
VID = '2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa'
STO = 'driver_run_store'
NWO, NCH = 24, 5000
ADD_SCALE = 1

cap = ut.get_video(VID)   
width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
natural_scale = width / 1920

conf = {'start_frame': START_FRAME,
        'stop_frame': STOP_FRAME,
        'video_id': VID,
        'store_name': STO,
        'num_workers': NWO,
        'chunksize': NCH,
        'add_scale': ADD_SCALE,
        'lossless': 'True',
        'compression_ball': 32,
        'compression_players': 48,
        'natural_scale': natural_scale,
        'scale_factor': natural_scale * ADD_SCALE,
        'cap': cap}

##### MAKE VID AND STO FOLDERS #####
featPath = ut.get_folder_struct('features_stored')['main']
vidPath = ut.maybe_make_folder(featPath, VID)
stoPath = ut.maybe_make_folder(vidPath, STO)

###### DETERMINE WHAT SHOULD BE RUN #####

can_run, has_run = ppl_fs.can_run_has_run(loadDict, saveDict, VID, STO)
should_run = [s for s in can_run if can_run[s] and not has_run[s]]

print('SCRIPTS TO RUN:')
for s in should_run:
    print(f'   - {s}')


###### RUN #############

width_of_line = 8 * conf['scale_factor']
courtline_params = {'width_of_line': width_of_line, 
                    'int_diff_thresh': 10,
                    'white_thresh': 150,
                    'rho': 3, 'theta': np.pi / 180, 'min_votes': int(500 * conf['scale_factor'] ** 2),
                    'minLineLength': width_of_line + 1, 'maxLineGap': 1,
                    'merge_threshold': 2 * width_of_line, 
                    'horizontal_filter_margin': 40 * conf['scale_factor'],
                    'vertical_filter_margin': 40 * conf['scale_factor'],
                    'only_225': False, 
                    'verbose': False
                    }

ppl.court_lines.main(conf, courtline_params)

