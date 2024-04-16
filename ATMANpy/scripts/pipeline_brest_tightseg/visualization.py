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
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D

from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous

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

    circles = []
    for x, y, z in srvShots.get_first_bounce_position():
        crc = Circle3D.from_point3D(Point3D((x, y, z), color = const.BGR['strong_pink']))
        circles.append(crc)
    crcs = Circles3D(circles)

    ### ANIMATION ###
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

    View.make(sideCam, courtScn, resolution = 1).snap().show()
    
    c_w = Point3D.from_point2D(lft_2D, z = 50).left(150).front(200)
    e_w = Point3D.from_point2D(srv_2D, z = 50)

    siderevCam = Camera.from_to(c_w, e_w, f = 600)
    
    View.make(siderevCam, courtScn, resolution = 1).snap().show()
    
    c_w = Point3D.from_point2D(rev_2D, z = 100).front(250).left(50)
    revCam = Camera.from_to(c_w, d_w, f = 1700)

    View.make(revCam, courtScn, resolution = 1).snap().show()
    
    crcScn = Scene.make_court().put(crcs)
    View.make(revCam, crcScn).snap().show()

    ### ANIMATION 
    descriptor = 'Second Serves'
    ballSF = srvShots.feat_spline_simul(color = const.BGR['ball_yellow'])
    
    ### STATISTICS ###
    qs =  [0.25, 0.5, 0.75]
    spds = srvShots.get_initial_speed()
    rpms = srvShots.get_initial_spin()

    spd_low, spd_mid, spd_hi = np.quantile(spds, qs)
    rpm_low, rpm_mid, rpm_hi = np.quantile(rpms, qs)

    def interpolate(startCam, endCam, start_frame, stop_frame):
        assert startCam.sensor_dims == endCam.sensor_dims
        frames = range(start_frame, stop_frame)
        num_frames = len(frames)

        itpl_factors = np.linspace(0, 1, num_frames, endpoint = True)
        itpl_factors_brdcst = itpl_factors[:, np.newaxis, np.newaxis]
        M_ints = startCam.M_int[np.newaxis] * (1 - itpl_factors_brdcst) + endCam.M_int[np.newaxis] * itpl_factors_brdcst
        M_exts = startCam.M_ext[np.newaxis] * (1 - itpl_factors_brdcst) + endCam.M_ext[np.newaxis] * itpl_factors_brdcst

        itplCams = [Camera(M_int, M_ext, startCam.sensor_dims) for (M_int, M_ext) in zip(M_ints, M_exts)]
        camDict = dict(zip(frames, itplCams))
        return camDict
    

    ###### ANIMATION ######
    FREEZE_FRAMES = 50
    SHOW_COLOR = const.BGR['grass_green']
    TEXT_COLOR = const.BGR['grass_green']
    TEXT_SIZE = 1.5
    RESOLUTION_BACKGROUND = 0.5
    RESOLUTION_OBJECTS = 0.5
    BANDWIDTH_KDE = 30

    frontScn = Scene.make_empty()

    frameList = []
    cams = interpolate(backCam, sideCam, 0, 9)
    for frame in cams:
        ball = ballSF[frame]
        
        frontScn = frontScn.transparify(0.8)
        frontScn = frontScn.put(ball)

        myCam = cams[frame]
        backView = View.make(myCam, courtScn, resolution = RESOLUTION_BACKGROUND)
        frontView = View.make(myCam, frontScn, resolution = RESOLUTION_OBJECTS)

        showView = backView.combine(frontView)
        showFrm = showView.snap()
        
        frameList.append(showFrm)

        _ = show_continuous(showFrm.array, ms_between_frames = 20)
    
    text = f'{descriptor}: '
    speed_text = f'Typical Speed: {spd_mid:.0f}km/h +- {(spd_hi - spd_low) / 2:.0f}km/h'
    spin_text = f'Typical RPM: {rpm_mid:.0f}rpm +- {(rpm_hi - rpm_low) / 2:.0f}rpm'
    showFrm_with_text = showFrm
    showFrm_with_text = showFrm_with_text.put(text, where = (50, 50), color = TEXT_COLOR, size = TEXT_SIZE)
    showFrm_with_text = showFrm_with_text.put(speed_text, where = (50, 100), color = TEXT_COLOR, size = TEXT_SIZE)
    showFrm_with_text = showFrm_with_text.put(spin_text, where = (50, 150), color = TEXT_COLOR, size = TEXT_SIZE)

    for frame in range(FREEZE_FRAMES):
        frameList.append(showFrm_with_text)
        _ = show_continuous(showFrm_with_text.array, ms_between_frames = 20)


    cams = interpolate(sideCam, siderevCam, 9, 23)
    for frame in cams:
        ball = ballSF[frame]
        
        frontScn = frontScn.transparify(0.8)
        frontScn = frontScn.put(ball)

        myCam = cams[frame]
        backView = View.make(myCam, courtScn, resolution = RESOLUTION_BACKGROUND)
        frontView = View.make(myCam, frontScn, resolution = RESOLUTION_OBJECTS)

        showView = backView.combine(frontView)
        showFrm = showView.snap()
        
        frameList.append(showFrm)

        _ = show_continuous(showFrm.array, ms_between_frames = 20)

    ### ADD IN RETURN DATA
    retArr = retShots.get_ball_hit_position()
    retArr_ground = np.concatenate([retArr[:, :2], np.zeros((len(retArr), 1))], axis = 1)
    r_lo, r_md, r_hi = np.quantile(retArr, qs, axis = 0)
    diff = r_hi - r_lo

    x_lo, y_lo, z_lo = r_lo
    x, y, z = r_md
    x_hi, y_hi, z_hi = r_hi

    retReg = Region3D.from_region2D(Region.from_points(Point2D((x_lo, y_lo)), Point2D((x_hi, y_hi))), z = 0)
    retReg = retReg.recolor(SHOW_COLOR)

    pos_radius = (diff[:2] ** 2).sum() ** 0.5 / 2
    height_radius = diff[2] / 2

    posCircle = Circle3D.from_point3D(Point3D((x, y, 0)), radius = pos_radius)
    heightCircle = Circle3D.from_point3D(Point3D((x, y, z)), radius = height_radius, fixed_dimension = 'x')
    
    heightLn_base = Line3D(Point3D((x, y, 0)), Point3D((x, y, z_lo)), thickness = 2, color = const.BGR['smokey_grey'])
    heightLn = Line3D(Point3D((x, y, z_lo)), Point3D((x, y, z_hi)), thickness = 2, color = SHOW_COLOR)

    from ATMANpy.types.Shapes3D import Lines3D
    heightLn_vectorized = Line3D(Point3D(retArr_ground), Point3D(retArr), color = SHOW_COLOR)

    kdeView = vs.kde_heatmap(retShots, myCam = siderevCam, position_func = 'get_ball_hit_position',
                             resolution = RESOLUTION_OBJECTS, bandwidth = BANDWIDTH_KDE)

    # return position
    crcScn = Scene.make_empty().put(retReg)
    crcView = View.make(siderevCam, crcScn)
    showView = backView.combine(frontView).combine(kdeView)
    showFrm = showView.snap()
    
    text = 'Ball typically hit at the baseline.'
    showFrm_with_text = showFrm.put(text, color = TEXT_COLOR, size = TEXT_SIZE)

    for frame in range(FREEZE_FRAMES):    
        frameList.append(showFrm_with_text)
        _ = show_continuous(showFrm_with_text.array, ms_between_frames = 20)
        
    cams = interpolate(siderevCam, revCam, 0, 10)
    for frame in cams:
        myCam = cams[frame]
        backView = View.make(myCam, courtScn, resolution = RESOLUTION_BACKGROUND)
        frontView = View.make(myCam, frontScn, resolution = RESOLUTION_OBJECTS)

        # crcScn = crcScn.transparify(0.8)
        # crcView = View.make(myCam, crcScn)
        
        kdeView = vs.kde_heatmap(retShots, myCam = myCam, position_func = 'get_ball_hit_position')

        showView = backView.combine(frontView).combine(kdeView)
        showFrm = showView.snap()
        
        frameList.append(showFrm)

        _ = show_continuous(showFrm.array, ms_between_frames = 20)

    # lnScn = Scene.make_empty().put(heightLn_base).put(heightLn)
    lnScn = Scene.make_empty().put(heightLn_vectorized)
    lnView = View.make(myCam, lnScn)
    showView = showView.combine(lnView)
    showFrm = showView.snap()

    z_lo_metres = z_lo / const.SCALE * const.FOOT_TO_METER
    z_hi_metres = z_hi / const.SCALE * const.FOOT_TO_METER
    text = f'Ball typically hit at height of between {z_lo_metres:.1f} and {z_hi_metres:.1f} metres'
    showFrm_with_text = showFrm.put(text, color = TEXT_COLOR, size = TEXT_SIZE)

    for frame in range(FREEZE_FRAMES):    
        frameList.append(showFrm_with_text)
        _ = show_continuous(showFrm_with_text.array, ms_between_frames = 20)

    cv.destroyAllWindows()

    from ATMANpy.types.MultiFrames import BGRFrames
    from ATMANpy.types.Streams import DecapStream
    myFrms = BGRFrames(np.array([f.array for f in frameList]))
    DecapStream(myFrms, 0, myFrms.length()).show(ms_between_frames = 50)

if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[1])

    name = ppl._name

    main(conf, name = None)

    
    
    
    