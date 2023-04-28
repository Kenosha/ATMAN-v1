# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:13:36 2023

@author: gianl
"""



####### READ IN #######

import pandas as pd
import cv2 as cv
from ATMANpy.types.Abstract import Points, Point, Shots, Shot, Trajectory
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous

from ATMANpy.types.Visual3D import View, Camera, Scene
from ATMANpy.types.ShapesTennis3D import Ball, Player, Racquet, Mark

import ATMANpy.auxil.constants as const

video_id = 'pqxAmUWIoKc'
store_name = '10to100_ok2'

event_name = 'grammatic_event'
ball_name = '3D_Ball_smart_inference'
top_player_name = '3D_topPlayer'
bot_player_name = '3D_botPlayer'

infPts = Points.from_stored_features(video_id, store_name, 
                                     event_name = event_name, ball_name = ball_name,
                                     top_player_name = top_player_name, bot_player_name = bot_player_name)

event_name = 'true_event'
ball_name = '3D_Ball_smart_ground'
truPts = Points.from_stored_features(video_id, store_name, 
                                     event_name = event_name, ball_name = ball_name,
                                     top_player_name = top_player_name, bot_player_name = bot_player_name)


#%% 

###### criteria to evaluate the inference on:
    ## Tightness of events (temporally):
        # - how close are serves, hits
        # - how close are inferred bounce frames from real bounces?
        
    ## Correctness of events:
        # - in/out
        # general sequence of events
        
    ## General criteria:
        # determine winner of point correctly?

def winner_comp(infPt, truPt):
    return truPt.get_winner() == infPt.get_winner()
        
def event_comp(infPt, truPt):
    return truPt.get_events() == infPt.get_events()

winnerList = []
eventList = []
for infPt, truPt in zip(infPts, truPts):
    winnerList.append(winner_comp(infPt, truPt))
    eventList.append(event_comp(infPt, truPt))

evalFrm = pd.concat([pd.Series(winnerList, name = 'winner_correct'), 
                     pd.Series(eventList, name = 'event_correct')], 
                    axis = 1)


#%% Testing

for ptIdx in range(len(infPts)):
    print(f'Point {ptIdx}...')
    infPt = infPts[ptIdx]
    truPt = truPts[ptIdx]
    
    print(infPt.compare(truPt))
    
    # for infShot, truShot in zip(infPt, truPt):
    #     print(infShot.compare(truShot))

    print('______')

