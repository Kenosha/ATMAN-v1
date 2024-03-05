# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:09:57 2024

@author: gianl
"""


import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import get_width_height

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points


from ATMANpy.types.Abstract import PlayerMovement, Trajectory, Shot, Shots, Point, Points

from ATMANpy.types.Visual3D import Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D


from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous

"""

### Events
# -> F: Flight
# -> STD/STA/SBD/SBA: Serve Top Deuce etc.
# -> HT/HB: Hit Top, Hit Bottom
# -> BI/BO: Bounce In, Bounce Out
# -> N: Net

"""

def load(vid, sto):
    HPCF = ConstFeatures.from_store(vid, sto, featnameList = ['H', 'P', 'M_int', 'M_ext'])
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    trajFrm = ff.read_feature_frame('trajTable', vid, sto)
    return HPCF, segBF, plySF, trajFrm

def main():
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = get_width_height(cap)
    
    HPCF, segBF, plySF, trajFrm = load(vid, sto)
    P = HPCF['P'].feature
    
    courtLines = CourtLines.make()
    
    
    
    ### find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    topSF = ply3DSF['topPlayer_3D'].apply(sa.point3D_to_coords)
    botSF = ply3DSF['botPlayer_3D'].apply(sa.point3D_to_coords)
    
    topSFs = topSF.conform(segBF).filter(segBF).split()
    botSFs = botSF.conform(segBF).filter(segBF).split()
    
    trajSFs = StreamFeatures.from_dataframe(trajFrm).conform(segBF).filter(segBF).split()
    
    myPnts = Points.from_SFs(trajSFs, topSFs, botSFs)
    
    myShots = myPnts.flatten()
    goodShots = myShots.filter('mean_deviation', is_below = 20, dof_correction = 3)
    b2tShots = goodShots.filter('is_bot_to_top')
    t2bShots = goodShots.filter('is_top_to_bot')
    t2bShots_flipped = t2bShots.flip()
    
    unidirShots = Shots([*b2tShots, *t2bShots_flipped])
    
    ad_serves = unidirShots.filter('is_serve_ad')
    deuce_serves = unidirShots.filter('is_serve_deuce')
        
    
    ballSF = ad_serves.feat_ball_flat()
    
    myCam = Camera.calibrated_AO_behind()
    emptyScn = Scene.make_empty()
    courtScn = Scene.make_court()
    backView = View.make(myCam, courtScn, resolution = 1)
    
    for frame in ballSF:
        ball = ballSF[frame]
        
        frontScn = emptyScn.put(ball)
        frontView = View.make(myCam, frontScn, resolution = 1)
    
        showFrm = backView.combine(frontView).snap()
        
        show_continuous(showFrm.array, ms_between_frames = 20)
    
    cv.destroyAllWindows()
    
    dof_correction = 3
    
    ad_bncPts = []
    for srv in ad_serves:
        try:
            first_bounce = srv.get_bounce_positions()[0]
        except IndexError:
            pass
        else:
            deviation = srv.shotTraj.mean_deviation(dof_correction = dof_correction) ### the higher, the redder we want the point to be colored!
            deviation_norm = min(deviation / 100, 1)
            
            bgr = (1 - deviation_norm) * np.array([0, 255, 0]) + deviation_norm * np.array([0, 0, 255])
            bgr = tuple(bgr.astype(int))
            
            first_bounce = first_bounce.recolor(bgr)
            
            ad_bncPts.append(first_bounce)
            
    [myShot.shotTraj.trajFrm.vx0.unique()[0] for myShot in ad_serves]
            
   
           
    
    
    ad_serves.feat_spline()
    
    Stream(cap, 66800, 67000).show(ms_between_frames = 100, frame_counter = True)
    
    deuce_bncPts = []
    for srv in deuce_serves:
        try:
            first_bounce = srv.get_bounce_positions()[0]
        except IndexError:
            pass
        else:
            deviation = srv.shotTraj.mean_deviation(dof_correction = dof_correction) ### the higher, the redder we want the point to be colored!
            deviation_norm = min(deviation / 100, 1)
            
            bgr = (1 - deviation_norm) * np.array([0, 255, 0]) + deviation_norm * np.array([0, 0, 255])
            bgr = tuple(bgr.astype(int))
            
            first_bounce = first_bounce.recolor(bgr)
            
            deuce_bncPts.append(first_bounce)
            
    
    
    myCam = Camera.calibrated_AO_behind()
    courtScn = Scene.make_court()
    backView = View.make(myCam, courtScn, resolution = 1)
    
    ballScn = Scene.make_empty().put(deuce_bncPts)
    frontView = View.make(myCam, ballScn, resolution = 1)
    
    showFrm = backView.combine(frontView).snap()
    showFrm.show()
    
    ballScn = Scene.make_empty().put(ad_bncPts)
    frontView = View.make(myCam, ballScn, resolution = 1)
    
    showFrm = backView.combine(frontView).snap()
    showFrm.show()
    
    ad_bncPts = [srv.get_bounce_positions() for srv in ad_serves]
    # deuce_bncPts = 
    
    
    deuce_serves[0].flip()
    
    
    
    ad_serves
    
    
    
    
    
    srvShot, retShot, lstShot = shots
    
    srvShot.is_fore()
    
    retShot.is_ground(srvShot)
    
    lstShot.is_overhead()
    
    retShot.is_fair(lstShot)
    
    myPnt = Point(shots)
    
    srvShot.shotTraj.feat_spline()[256]
    
    ### Visualize
    ballSF = myPnt.feat_ball()
    hitSF = myPnt.feat_hit_player()
    splnSF = myPnt.feat_spline()
    
    myCam = Camera.calibrated_AO_behind()
    emptyScn = Scene.make_empty()
    courtScn = Scene.make_court()
    backView = View.make(myCam, courtScn, resolution = 1)
    
    for frame in ballSF:
        ball = ballSF[frame]
        hitPlayer = hitSF[frame]
        ball_spline = splnSF[frame]
        
        frontScn = emptyScn.put(ball_spline).put(hitPlayer)
        frontView = View.make(myCam, frontScn, resolution = 1)
    
        showFrm = backView.combine(frontView).snap()
        
        show_continuous(showFrm.array, ms_between_frames = 20)
    
    cv.destroyAllWindows()
   
   
if __name__ == '__main__':
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    
    
    
    
    