# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:24:00 2024

@author: gianl
"""

import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

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

from ATMANpy.types.Visual3D import Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D

# def get_ball_cloud_around(centerPt3D, hitFlag):
    
def load(vid, sto):
    HPCF = ConstFeatures.from_store(vid, sto, featnameList = ['H', 'P', 'M_int', 'M_ext'])
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
    ballSF = StreamFeature.from_store(vid, sto, featname = 'ball_choicePt')
    hitSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)
    return HPCF, segBF, ballSF, hitSF, plySF, audFrm

def save(vid, sto):
    pass

def check(myPlayer3D, myBall3D, resolution = 10):
    ### Visualise
    courtScn = Scene.make_court()
    myScene = courtScn.put(myPlayer3D).put(myBall3D)
    
    ctPt = CourtLines.make().get_center_point()
    x, y = ctPt.coords
    ctBall3D = Ball3D((x, y, 0), radius = 10, color = const.BGR['intense_red'])
    testScene = Scene.make_empty().put(ctBall3D)
    
    myCamera = Camera.calibrated_AO_behind()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    myCamera = Camera.low_angled_behind()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    myCamera = Camera.mid_lateral()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    s_x, s_y = 1920, 1080
    sensor_dims = s_x, s_y
    cam_height = 4000
    o_x = s_x / 2
    o_y = s_y / 2
    
    myCamera = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    # testView = View.make(myCamera, testScene, resolution = resolution)
    # testView.snap().show()


def view3D(thing3D, resolution = 10):
    ### Visualise
    courtScn = Scene.make_court()
    myScene = courtScn.put(thing3D)
    
    myCamera = Camera.calibrated_brest_behind()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()

sample_frame = 255
resolution = 5

distance_threshold = 10
def filter_hit_candidates_with_ball(hitPts3D, ballPt2D, P, distance_threshold = 10):
    ptArr = hitPts3D.rasterize()
    if ballPt2D is None:
        ptArr_filtered = ptArr
    else:
        projArr = sa.project_hit_point_cloud(hitPts3D, P, unique = False)
        dist = ((projArr - ballPt2D.coords) ** 2).sum(axis = 1) ** 0.5
        ptArr_filtered = ptArr[dist < distance_threshold]
    return ptArr_filtered
        

def main(conf, params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = get_width_height(cap)
    
    HPCF, segBF, ballSF, hitSF, plySF, audFrm = load(vid, sto)
    
    courtLines = CourtLines.make()
    courtDims = courtLines.see_dimensions()
    
    P = HPCF['P'].feature
    H = HPCF['H'].feature
    M_int = HPCF['M_int'].feature
    M_ext = HPCF['M_ext'].feature
    matchCamera = Camera(M_int, M_ext, (width, height))
    
    ### find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    ### make point cloud
    mySF = hitSF.merge(plySF).merge(HPCF['P'])
    params = {'reaches_in_feet': [2.5, 3, 3.5, 4, 4.5, 5],
              'resolution': 0.01}
    cld3DSF = mySF.apply(sa.get_3D_hit_point_cloud, **params)
    
    ### check point cloud
    check_frame = 275
    player3D = ply3DSF[check_frame]['topPlayer_3D']
    check(player3D, cld3DSF[check_frame])
    BGRFrame.from_video(cap, check_frame).show()
    
    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(segBF).filter(segBF).split()
    mySF = mySFs[0]
    
    anchorSF = mySF['cldPt3D']
    shapeSF = mySF['ball_choicePt']
    camlnSF = shapeSF.apply(sa.make_camline, myCamera = matchCamera)
    
    anchor_frames = anchorSF.dropna().get_frames()
    
    idx = 0
    frm_currhit, frm_nexthit = anchor_frames[idx], anchor_frames[idx + 1]
    anchorPts_curr, anchorPts_next = anchorSF[frm_currhit], anchorSF[frm_nexthit]
    ballPt_curr, ballPt_next = shapeSF[frm_currhit], shapeSF[frm_nexthit]
    
    flight_frames = list(range(frm_currhit, frm_nexthit + 1))
    non_na_frmArr = np.array([(i, f) for i, f in enumerate(flight_frames) if f in shapeSF.dropna().get_frames()])
    
    distance_threshold = 5
    ptArr_curr = filter_hit_candidates_with_ball(anchorPts_curr, ballPt_curr, P, distance_threshold = distance_threshold)
    ptArr_next = filter_hit_candidates_with_ball(anchorPts_next, ballPt_next, P, distance_threshold = distance_threshold)
    
    ptArr_curr_ground = np.copy(ptArr_curr)
    ptArr_curr_ground[:, 2] = 0
    ptArr_next_ground = np.copy(ptArr_next)
    ptArr_next_ground[:, 2] = 0
        
    num_curr_points = len(ptArr_curr_ground)
    num_next_points = len(ptArr_next_ground)
    num_flight_frames = len(flight_frames)
    
    coordArr = np.full((num_curr_points, num_next_points, num_flight_frames, 3), fill_value = np.nan)
    paramArr = np.full((num_curr_points, num_next_points, num_flight_frames, 3), fill_value = np.nan)
    for curr_idx in range(num_curr_points):
        currptArr = ptArr_curr_ground[curr_idx]
        currPt = Point3D(tuple(currptArr))
        for next_idx in range(num_next_points):
            nextptArr = ptArr_next_ground[next_idx]
            nextPt = Point3D(tuple(nextptArr))
    
            ### make flight plane
            flightLn = Line3D(currPt, nextPt)
            flightReg = Region3D.from_line(flightLn, 100, dim = 'z')
            
            for frame_idx, frame_num in enumerate(flight_frames):
                camLn = camlnSF[frame_num]
                if camLn is not None:
                    ballPt3D, params = camLn.intersect(flightReg)
                    
                    coordArr[curr_idx, next_idx, frame_idx] = ballPt3D.coords
                    paramArr[curr_idx, next_idx, frame_idx] = params
        
    ### verify the integrity of the coordinates
    # first, get the non nan arrays 
    non_na_idxs = non_na_frmArr[:, 0]
    coordArr_non_nan = coordArr[:, :, non_na_idxs, :]    
    paramArr_non_nan = paramArr[:, :, non_na_idxs, :]     
    
    num_trajectories = num_curr_points * num_next_points
    print(f'Out of {num_trajectories} trajectories...')

    # a) they can not be negative
    is_pos = coordArr_non_nan >= 0 ### TODO: maybe have a slightly negative allowed value here.
    
    pos_coords = is_pos.all(axis = (2, 3))
    num_pos_coords = pos_coords.sum()    
    print(f'   ... {num_pos_coords} are always above ground')
    
    # b) they must have monotonously increasing lambdas along the flight path at ground
    lambdaArr = paramArr_non_nan[:, :, :, 0] ## get only flight path lambdas along ground
    delta_lambdaArr = np.diff(lambdaArr, axis = 2) ## first difference these in time
    
    monotonous_lambdas = (delta_lambdaArr > 0).all(axis = 2)
    num_monotonous_lambdas = monotonous_lambdas.sum()    
    print(f'   ... {num_monotonous_lambdas} are monotonously increasing')
    
    # c) the lambdas must have monotonously decreasing first derivatives
    ### -> the ball decelerates in flight!
    # for this, we calculate the lambda divided by the index
    lambda_per_time = lambdaArr / non_na_idxs[np.newaxis, np.newaxis, :]
    # this metric (essentially measuring the average ground speed up to each observed point)
    # ought to be decreasing -> at the beginning, speed is very high, then decreases
    delta_lambda_per_time = np.diff(lambda_per_time, axis = 2)
    
    decelerating_lambdas = (delta_lambda_per_time < 0).all(axis = 2)
    num_decelerating_lambdas = decelerating_lambdas.sum()    
    print(f'   ... {num_decelerating_lambdas} are decelerating')
    
    # d) start and end points must be 'reasonably close' to 0 and 1
    
    # e) ball heigth must be consistent with one or two parabolas
    
    # join all the conditions
    good_trajectory = pos_coords & monotonous_lambdas & decelerating_lambdas
    num_good_trajectories = good_trajectory.sum()
    print(f'   ... {num_good_trajectories} are ok!')
    
    
    
    ###
    paramArr_reshape = paramArr.reshape((num_curr_points * num_next_points, num_flight_frames, 3))
    paramArr_reshape_non_nan = paramArr_reshape[:, non_na_idxs, :] 
    
    paramArr_filtered = paramArr_reshape_non_nan[good_trajectory.flatten()]
    
    plt.plot(paramArr_filtered[0, :, 0], paramArr_filtered[0, :, 1])
    
    def plot_trajectory(para)
    
    
    
    
    delta_lambda_per_time[0, 0]
    np.interpo
    
    paramArr[0, 0, :, 0]
    
    np.isnan(coordArr[0, 0]) | (coordArr[0, 0] > 0)
    
    
    
    
    chkArr = (coordArr > 0).all(axis = (2, 3))       
    chkArr = (coordArr > 0).mean(axis = (2, 3))
    
    
    ### visualize clouds
    mySF = cld3DSF.merge(HPCF['P'])
    cld2DSF = mySF.apply(sa.project_hit_point_cloud)
    
    frm = frm_currhit
    ptList = [Point2D(pt) for pt in cld2DSF[frm]]
    BGRFrame.from_video(cap, frm).show()
    BGRFrame.from_video(cap, frm).put(ptList).show()

    num_points, dims = ptArr_next.shape
    ### Turn coordinates homogenous
    homArr = np.concatenate([ptArr_next, np.ones((num_points, 1))], axis = 1)
    
    ### 
    homArr_2D = np.einsum('ij, nj -> ni', P, homArr)
    pixArr = homArr_2D[:, :2] / homArr_2D[:, 2][:, np.newaxis]
    
    ptList = [Point2D(pt) for pt in pixArr]
    BGRFrame.from_video(cap, frm).show()
    BGRFrame.from_video(cap, frm).put(ptList).show()

    ### visualize camlines
    courtScn = Scene.make_court()
    emptyScn = Scene.make_empty()
    
    s_x, s_y = 1920, 1080
    sensor_dims = s_x, s_y
    cam_height = 4000
    o_x = s_x / 2
    o_y = s_y / 2
    showCam = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    
    showCam = Camera.calibrated_AO_behind()
    
    for frame in flight_frames:
        camLn = camlnSF[frame]
        
        if not camLn is None:
            longLn = Line3D(camLn.leftPt, camLn.leftPt + camLn.to_vector() * 1000, color = const.BGR['intense_red'])
            
            myScn = courtScn.put(longLn).put(flightReg)
            myView = View.make(showCam, myScn)
            
            myView.snap().show()
    
        
    
    coordArr[0, 0]
    
    
            
    
    np.array(coordList)
    np.array(paramList)
    
    s_x, s_y = 1920, 1080
    sensor_dims = s_x, s_y
    cam_height = 12000
    o_x = s_x / 2
    o_y = s_y / 2
    
    myCamera = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    
    from ATMANpy.funcs.HMM.HPost import calculate_interpolation_lambda
    for frame in range(frm_currhit, frm_nexthit):
        try:
            ballPt = shapeSF[frame]
        except KeyError:
            pass
        else:
            if ballPt is not None:
                
                m_w, b_w = matchCamera.get_world_ray_params(ballPt)
                camPt, camVec = Point3D(b_w), Point3D(m_w)
                camLine = Line3D(camPt, camPt - camVec * 10000, color = const.BGR['strong_pink'])
        
                intPt = camLine.intersect(myReg)
        
                myScene = Scene.make_court().put(camLine).put(myReg)
                
                View.make(myCamera, myScene).snap().show()
        
                calculate_interpolation_lambda(m_w, b_w, pt1, pt2)
              
    
    Stream.from_SF(shapeSF, cap).show(ms_between_frames = 500)
              
    BGRFrame.from_video(cap, frame).put(ballPt).show()
                
    
    view3D(myReg)
    
    
    
    
    
    mySF = cld3DSF.merge(HPCF['P'])
    cld2DSF = mySF.apply(sa.project_hit_point_cloud)
    
    
    coords = cld2DSF[sample_frame]
    myPts = [Point2D(coord) for coord in coords]
    BGRFrame.from_video(cap, sample_frame).put(myPts).show()
    
    outArr = sa.project_hit_point_cloud(cldPt3D, P)
    
    HPCF['M_int'].feature
    
    cldPt3D = cld3DSF[sample_frame]
    ptArr = cldPt3D.rasterize()
    
    cldPt3D.project(P)
    
    myCamera
    
    
    myFrm = BGRFrame.from_video(cap, sample_frame).put([Point2D(tuple(pt)) for pt in outArr])
    myFrm.show()
    
    courtScn = Scene.make_court()
    myScene = courtScn.put(cldPt3D)
    
    myCamera = Camera.calibrated_brest_behind()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    np.sort(myView.pxArr)
    
    myCamera.project_points(ptArr)
    
    mySF 
    
    cldPt3D, P = mySF[sample_frame].values()
    
    myFrm = BGRFrame.from_video(cap, sample_frame).put([Point2D(tuple(pt)) for pt in cldPts2D_float])
    myFrm.show()
    
    

    

    

    
    
if __name__ == '__main__':
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    params = {}

