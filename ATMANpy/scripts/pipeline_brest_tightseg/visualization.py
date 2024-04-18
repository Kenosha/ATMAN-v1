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

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.types.Abstract import PlayerMovement, Trajectory, Shot, Shots, Point, Points

from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D, Lines3D

from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous


from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

"""
### Events
# -> F: Flight
# -> STD/STA/SBD/SBA: Serve Top Deuce etc.
# -> HT/HB: Hit Top, Hit Bottom
# -> BI/BO: Bounce In, Bounce Out
# -> N: Net
"""

def fix_traj_frame(trajFrm):
    ### TODO: this can go as soon as ball_interpolation_plane is re-run 
    ### with overheads recognized as serves that have a serve event (SBA, SBD, STA, STD) at shot_idx > 0
    not_first = trajFrm.shot_idx > 0
    serve_recognized = trajFrm.event.isin(['SBA', 'SBD', 'STA', 'STD'])
    to_fix = not_first & serve_recognized
    print(f'Fixing {to_fix.sum()} instances of serves that are recognized during the rally.')
    trajFrm.event = np.where(not_first & serve_recognized, 'H' + trajFrm.event.str[1], trajFrm.event)
    return trajFrm

def load(vid, sto):
    HPCF = ConstFeatures.from_store(vid, sto, featnameList = ['H', 'P', 'M_int', 'M_ext'])
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally_warmup_removed')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    srvSF = StreamFeature.from_store(vid, sto, featname = 'srvFlag')
    trajFrm = ff.read_feature_frame('trajTable', vid, sto)
    return HPCF, segBF, plySF, srvSF, trajFrm

def save(vid, sto):
    pass

def main(conf, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    courtLines = CourtLines.make().add_singles_netline()

    HPCF, segBF, plySF, srvSF, trajFrm = load(vid, sto)
    trajFrm = fix_traj_frame(trajFrm)
    trajSF = StreamFeatures.from_dataframe(trajFrm)
    trajSF = trajSF.merge(srvSF.rename('event_info'))
    
    ### find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    topSF = ply3DSF['topPlayer_3D'].apply(sa.point3D_to_coords)
    botSF = ply3DSF['botPlayer_3D'].apply(sa.point3D_to_coords)
    
    topSFs = topSF.conform(segBF).filter(segBF).split()
    botSFs = botSF.conform(segBF).filter(segBF).split()
    
    trajSFs = trajSF.conform(segBF).filter(segBF).split()
    
    ### PREPARE RELEVANT POINTS ###
    myPnts = Points.from_SFs(trajSFs, topSFs, botSFs)

    intPnts = myPnts.filter('is_second_serve')
    intPnts = intPnts.enforce_direction('Bot')
    
    ad_serves = intPnts.filter('is_serve_ad').flip(across_middle = True, across_net = False)
    deuce_serves = intPnts.filter('is_serve_deuce')

    intPnts = ad_serves.join(deuce_serves)
    assert len(intPnts) == len(intPnts.filter('is_bot_to_top').filter('is_serve_deuce'))

    fairPnts = intPnts.filter('is_serve_fair')
    srvShots = fairPnts.get_serves()
    retShots = fairPnts.get_returns()

    # circles = []
    # for x, y, z in srvShots.get_first_bounce_position():
    #     crc = Circle3D.from_point3D(Point3D((x, y, z), color = const.BGR['strong_pink']))
    #     circles.append(crc)
    # crcs = Circles3D(circles)

    ### PREPARE CAMERAS ###
    courtScn = Scene.make_court()
    courtLines = courtLines
    courtLines.names()

    bot_2D = courtLines['Bottom Baseline'].center()
    cnt_2D = courtLines.get_center_point()
    lft_2D = courtLines['Net Line'].leftPt
    rev_2D = courtLines['Top Baseline'].center()
    srv_2D = courtLines['Top Service Line'].center()

    c_w = Point3D.from_point2D(bot_2D, z = 100).back(250).left(50)
    d_w = Point3D.from_point2D(cnt_2D, z = 0)
    backCam = Camera.from_to(c_w, d_w, f = 1700)
    # backView = View.make(backCam, courtScn, resolution = 1)
    # backView.snap().show()

    c_w = Point3D.from_point2D(lft_2D, z = 100).left(200)
    sideCam = Camera.from_to(c_w, d_w, f = 600)
    # View.make(sideCam, courtScn, resolution = 1).snap().show()
    
    c_w = Point3D.from_point2D(lft_2D, z = 50).left(150).front(200)
    e_w = Point3D.from_point2D(srv_2D, z = 50)
    siderevCam = Camera.from_to(c_w, e_w, f = 600)
    # View.make(siderevCam, courtScn, resolution = 1).snap().show()
    
    c_w = Point3D.from_point2D(rev_2D, z = 100).front(250).left(50)
    revCam = Camera.from_to(c_w, d_w, f = 1700)
    # View.make(revCam, courtScn, resolution = 1).snap().show()

    ### PARAMETERS ### 
    DESCRIPTOR = 'Second Serves'

    QUANTILES =  [1/3, 1/2, 2/3]
    FREEZE_FRAMES = 200
    SHOW_COLOR = const.BGR['strong_pink']
    TEXT_COLOR = const.BGR['grass_green']
    TEXT_SIZE = 1.5
    RESOLUTION_BACKGROUND = 0.5
    RESOLUTION_OBJECTS = 0.5
    RESOLUTION_KDE = 0.5
    BANDWIDTH_KDE = 30
    SCENE_FADE = 0.9

    CUTOFF_FLIGHT_FRAMES = 6
    EXPANSION_FACTOR = 4
    
    backScn = Scene.make_court()
    header_text = f'{DESCRIPTOR}: '
    permanent_animation_params = {'background_resolution': RESOLUTION_BACKGROUND,
                                   'foreground_resolution': RESOLUTION_OBJECTS,
                                   'dummy_resolution': RESOLUTION_KDE,
                                   'background_scene': backScn,
                                   'text_size': TEXT_SIZE,
                                   'text_color': TEXT_COLOR,
                                   'text_where': (50, 50),
                                   'text_offset': 50}

    ### BALL SHAPES ###
    ballSF = srvShots.feat_spline_simul(color = const.BGR['ball_yellow'], which_endpoint = 'smooth')
    shapes = list(ballSF.values())
    shapes_vectorized = [f.vectorize() for f in shapes if f is not None]

    ### Expand ball flight points
    out_lines = []
    for flightLn in shapes_vectorized:
        if flightLn is None:
            itpl_lines = [None] * EXPANSION_FACTOR
        else:
            itpl_lines = vs.divide_line(flightLn, EXPANSION_FACTOR)
                   
        out_lines.append(itpl_lines)

    shapes_vectorized_expanded = [l for lineList in out_lines for l in lineList]
    first_half_flight_expanded = shapes_vectorized_expanded[:CUTOFF_FLIGHT_FRAMES * EXPANSION_FACTOR]
    second_half_flight_expanded = shapes_vectorized_expanded[CUTOFF_FLIGHT_FRAMES * EXPANSION_FACTOR:]

    ### RETURN POSITION ###
    retArr = retShots.get_ball_hit_position()
    retArr_ground = np.concatenate([retArr[:, :2], np.zeros((len(retArr), 1))], axis = 1)

    heightLn_vectorized = Line3D(Point3D(retArr_ground.T), Point3D(retArr.T), color = SHOW_COLOR)
    heightLn_vectorized_list = vs.divide_line(heightLn_vectorized, 10, reverse_direction = True)
    
    large_courtReg = courtLines.get_court_region().scale(1.25)
    kdeScn = vs.kde_heatmap(retShots, position_func = 'get_ball_hit_position',
                             bandwidth = BANDWIDTH_KDE,
                             showReg = large_courtReg)
    
    ### STATISTICS ###
    spds = srvShots.get_initial_speed()
    rpms = srvShots.get_initial_spin()

    spd_low, spd_mid, spd_hi = np.quantile(spds, QUANTILES)
    rpm_low, rpm_mid, rpm_hi = np.quantile(rpms, QUANTILES)

    speed_text = f'Typical Speed: {spd_mid:.0f}km/h +- {(spd_hi - spd_low) / 2:.0f}km/h'
    spin_text = f'Typical RPM: {rpm_mid:.0f}rpm +- {(rpm_hi - rpm_low) / 2:.0f}rpm'

    r_lo, r_md, r_hi = np.quantile(retArr, QUANTILES, axis = 0)
    diff = r_hi - r_lo
    
    x_lo, y_lo, z_lo = r_lo
    x, y, z = r_md
    x_hi, y_hi, z_hi = r_hi
    
    return_text = 'Ball typically hit at the baseline.'
    
    z_lo_metres = z_lo / const.SCALE * const.FOOT_TO_METER
    z_hi_metres = z_hi / const.SCALE * const.FOOT_TO_METER
    height_text = f'Ball typically hit at height of between {z_lo_metres:.1f} and {z_hi_metres:.1f} metres'

    ### ANIMATION ###
    print('Animating ...')
    print(' ... 1) first half of ball flight')
    myFrms1, fore_scene1 = vs.animate(shapes = first_half_flight_expanded,
                                        start_cam = backCam, stop_cam = sideCam, 
                                        cumulative_foreground_transparence = SCENE_FADE,
                                        **permanent_animation_params
                                        )
    # DecapStream(myFrms1, 0, myFrms1.length()).show(ms_between_frames = 25)

    print(' ... 2) speed and spin text')
    text_lines = [header_text, speed_text, spin_text]
    myFrms2, fore_scene2 = vs.animate_static(previous_foreground_scene = fore_scene1,
                                            static_cam = sideCam,
                                            text_lines = text_lines, 
                                            num_frames = FREEZE_FRAMES,
                                            **permanent_animation_params)
    # DecapStream(myFrms2, 0, myFrms2.length()).show(ms_between_frames = 25)

    print(' ... 3) show rest of flight')
    myFrms3, fore_scene3 = vs.animate(shapes = second_half_flight_expanded, 
                                        previous_foreground_scene = fore_scene2,
                                        start_cam = sideCam, stop_cam = siderevCam, 
                                        cumulative_foreground_transparence = SCENE_FADE,
                                        **permanent_animation_params)
    # DecapStream(myFrms3, 0, myFrms3.length()).show(ms_between_frames = 25)

    print(' ... 4) Fade out ball flight slightly')
    myFrms4, fore_scene4 = vs.animate(previous_foreground_scene = fore_scene3,
                                     start_cam = siderevCam, stop_cam = siderevCam, num_frames = 10,
                                     cumulative_foreground_transparence = 0.95,
                                     **permanent_animation_params)
    # DecapStream(myFrms4, 0, myFrms4.length()).show(ms_between_frames = 200)

    print(' ... 5) Return Positions rel. to baseline')
    text_lines = [return_text]

    myFrms5, fore_scene5 = vs.animate(shapes = heightLn_vectorized_list, 
                                    start_cam = siderevCam, stop_cam = siderevCam,
                                    previous_foreground_scene = fore_scene4, 
                                    text_lines = text_lines,
                                    **permanent_animation_params)
    # DecapStream(myFrms5, 0, myFrms5.length()).show(ms_between_frames = 200)

    print(' ... 6) Return KDE')
    myFrms6, fore_scene6 = vs.animate_static(static_cam = siderevCam, num_frames = FREEZE_FRAMES, 
                                          previous_foreground_scene = fore_scene5,
                                          dummy_scene = kdeScn, 
                                          text_lines = text_lines,
                                          **permanent_animation_params)
    # DecapStream(myFrms6, 0, myFrms6.length()).show(ms_between_frames = 25)

    print(' ... 7) Camera Turn for Ball Heights')
    myFrms7, fore_scene7 = vs.animate(num_frames = 40,
                                   start_cam = siderevCam, stop_cam = revCam,
                                    previous_foreground_scene = fore_scene6,
                                    dummy_scene = kdeScn,
                                    text_lines = text_lines,
                                    **permanent_animation_params)
    # DecapStream(myFrms7, 0, myFrms7.length()).show(ms_between_frames = 100)

    print(' ... 8) Ball Height Information')
    text_lines = [return_text, height_text]

    myFrms8, fore_scene8 = vs.animate_static(static_cam = revCam, num_frames = FREEZE_FRAMES, 
                                          previous_foreground_scene = fore_scene7,
                                          dummy_scene = kdeScn, 
                                          text_lines = text_lines,
                                          **permanent_animation_params)
    # DecapStream(myFrms8, 0, myFrms8.length()).show(ms_between_frames = 25)


    jntFrms = BGRFrames.from_frames([myFrms1, myFrms2, myFrms3, myFrms4,
                                     myFrms5, myFrms6, myFrms7, myFrms8])
    DecapStream(jntFrms, 0, jntFrms.length()).show(ms_between_frames = 25)



if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[1])

    name = ppl._name

    main(conf, name = None)

    
    
    
    