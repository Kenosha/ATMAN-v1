# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 23:56:09 2023

@author: gianl
"""

####### READ IN #######

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

myPts = Points.from_stored_features(video_id, store_name, 
                                    event_name = event_name, ball_name = ball_name,
                                    top_player_name = top_player_name, bot_player_name = bot_player_name)




###### SPIN #####
### read theoretical trajectories
import ATMANpy.funcs.spin_funcs as sf
groundArr, vertArr, trajDict = sf.load_trajectories('TrajectoriesRPM_small', dtype = 'float32')

### initialise search spaces with proper arguments
import inspect
kwargList = inspect.getargspec(sf.initialize).args
kwargs = {k: v for k, v in trajDict.items() if k in kwargList}
v_init_ground_space, v_init_vert_space, spin_rate_space = sf.initialize(dtype = 'float32', 
                                                                        squeeze = True, 
                                                                        **kwargs)
#%%### PARAMS ######

PERSPECTIVE = 'tv'
BACKGROUND_RES = 1
FOREGROUND_RES = 2
MS_BETWEEN_FRAMES = 30

#%%%

myCam = Camera.calibrated_AO_behind()

if PERSPECTIVE == 'behind':
    ### low, angled
    myCam = myCam.left(20).into(14).pan(-0.06).down(4)

if PERSPECTIVE == 'lateral':
    # # ### side view
    myCam = myCam.right(55).pan(0.5).rotate(0.005).right(133).zoom(0.175)
    myCam = myCam.tilt(-0.05).pan(0.008)

backScn = Scene.make_court()
backView = View.make(myCam, backScn, resolution = BACKGROUND_RES)

myPt = myPts[0]
myShot = myPt[0]
myTraj = myShot.shotTraj

myTraj.get_frames()

myShot.get_frames()
myPt.get_frames()

myShot.shotTraj
myShot.bounceList

myShot.get_splines()

# myShotFit = myShot.fit_trajectory(groundArr, vertArr, 
#                                   v_init_ground_space, v_init_vert_space, spin_rate_space)

# ballsplineFeat, fitsplineFeat = myShotFit.get_splines('ball'), myShotFit.get_splines('ball_fit')

myPt[2]

myPt = myPt.fit_trajectory(groundArr, vertArr, 
                           v_init_ground_space, v_init_vert_space, spin_rate_space)

ballsplineFeat, fitsplineFeat = myPt.get_splines('ball'), myPt.get_splines('ball_fit')
ballsplineFeat = ballsplineFeat.recolor(const.BGR['ball_yellow'])
fitsplineFeat = fitsplineFeat.recolor(const.BGR['bright_blue'])

frntScn = Scene.make_empty()


for frame in ballsplineFeat:
    try:
        ballspLine = ballsplineFeat[frame]
    except:
        ballspLine = None
    
    try:
        fitspLine = fitsplineFeat[frame]
    except:
        fitspLine = None
    
    frntScn = frntScn.put(ballspLine).put(fitspLine)
    frontView = View.make(myCam, frntScn, resolution = FOREGROUND_RES)
    
    showFrm = backView.combine(frontView).snap()
    
    show_continuous(showFrm.array, ms_between_frames = MS_BETWEEN_FRAMES)
    

cv.destroyAllWindows()
