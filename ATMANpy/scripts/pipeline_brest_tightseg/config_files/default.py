# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 03:18:25 2024

@author: gianl
"""

import ATMANpy.funcs.util_funcs as ut
import cv2 as cv

def get_conf(vid):
    START_FRAME, STOP_FRAME = 0, None
    STO = 'driver_run_store'
    NWO, NCH = 24, 5000
    ADD_SCALE = 1
    
    cap = ut.get_video(vid)   
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    natural_scale = width / 1920
    
    if STOP_FRAME is None:
        STOP_FRAME = cap.get(cv.CAP_PROP_FRAME_COUNT) - 5
    
    conf = {'start_frame': START_FRAME,
            'stop_frame': STOP_FRAME,
            'video_id': vid,
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
    
    return conf