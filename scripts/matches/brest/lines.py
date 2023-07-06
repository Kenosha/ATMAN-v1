# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:45:37 2023

@author: gianl
"""

import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import join_video_captures

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
RUN_LENGTH = 10000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'test'
NUM_WORKERS = 24
CHUNKSIZE = 1000

stop_frame = START_FRAME + RUN_LENGTH
cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE

if __name__ == '__main__':

    ### Make folders for feature storage
    featPath = ut.get_folder_struct('features_stored')['main']
    feat_vidPath = os.path.join(featPath, VIDEO_ID)
    try:
        os.mkdir(feat_vidPath)
    except FileExistsError:
        pass
    
    storePath = os.path.join(feat_vidPath, STORE_NAME)
    try:
        os.mkdir(storePath)
    except FileExistsError:
        pass
    
    ### Read Courtlines
    # -> since camera is fixed, lines are simply a parameter! we will make a single homography by getting a few random shots
    randStm = RandStream(cap, START_FRAME, stop_frame, number_of_frames = 1000,
                         cap_color_space = 'bgr', target_color_space = 'grey',
                         scale_factor = ADD_SCALE)
    
    WIDTH_OF_LINE_AT_FULL_HD = 8
    WIDTH_OF_LINE = WIDTH_OF_LINE_AT_FULL_HD * SCALE_FACTOR
    params = {'width_of_line': WIDTH_OF_LINE, 
              'int_diff_thresh': 10,
              'white_thresh': 150,
              'rho': 3, 'theta': np.pi / 180, 'min_votes': int(500 * SCALE_FACTOR ** 2),
              'minLineLength': WIDTH_OF_LINE + 1, 'maxLineGap': 1,
              'merge_threshold': 2 * WIDTH_OF_LINE, 
              'horizontal_filter_margin': 40 * SCALE_FACTOR,
              'vertical_filter_margin': 40 * SCALE_FACTOR,
              'only_225': False, 
              'verbose': False
              }
    courtSF = randStm.extract_mp(ex.get_courtlines, NUM_WORKERS, **params)
    
    ### Average out courtlines found.
    HSF = courtSF['H'].dropna()
    cornerSF = HSF.apply(sa.find_courtlines).apply(sa.get_court_corner_points).apply(sa.points_to_coords)
    
    median_points = np.median(cornerSF.to_array(), axis = 0).astype(int)
    m1, m2, m3, m4 = median_points
    H = sa.calculate_homography(Point2D(m1), Point2D(m2), Point2D(m3), Point2D(m4))
    courtLines = sa.find_courtlines(H)
    
    ### find netlines ###
    courtStm = NCStream.from_SF(courtSF['courtLines'].dropna(), cap, target_color_space = 'grey', name = 'gFrm')
    netSF = courtStm.extract(ex.find_netlines)
    
    netpointsSF = netSF.apply(sa.get_net_corner_points).apply(sa.points_to_coords_net)
    median_points = np.median(netpointsSF.to_array(), axis = 0).astype(int)
    bl, tl, tm, tr, br = median_points

    netLines = NetLines.from_points(Point2D(bl), Point2D(tl), Point2D(tm), Point2D(tr), Point2D(br))
    
    ### Check the quality of line finding!
    myStm = Stream(cap, START_FRAME, stop_frame)
    myStm[4000].put(courtLines).put(netLines).show()
    
    myStm[4000].put(courtLines.get_court_region('Singles')).show()
    schemaLines.get_3D_corner_points('Singles')
    schemaLines.get_3D_corner_points('Doubles')
    
    P = sa.calibrate_camera(courtLines, netLines, width = 'Singles')
    P2 = sa.calibrate_camera(courtLines, netLines, width = 'Doubles')
    M_int, M_ext = sa.find_intrinsic_extrinsic_M(P, normalize_K = True)
    
    if np.isnan(M_ext).any():
        print('Decomposition of P-Matrix failed!')
        ### Wait! This could be because the net line lies directly on the projected line between corner points!
        ### This might be what fucks it up!!!
        ### Update: Not really, sadly. see above.
    
    def save_constant_array(npArr, saveName, storePath):
        constPath = os.path.join(storePath, 'constants')
        try:
            os.mkdir(constPath)
        except FileExistsError:
            pass
        fullPath = os.path.join(constPath, f'{saveName}.npy')
        np.save(fullPath, npArr)
    
    save_constant_array(H, 'H', storePath)
    save_constant_array(P, 'P', storePath)
    save_constant_array(M_int, 'M_int', storePath)
    save_constant_array(M_ext, 'M_ext', storePath)
    
    ### Use the Frames where we did find lines to calculate a background frame!
    # the nice thing is: Since the homography is constant, there is no need to warp!!!
    # We basically construct a NCDecapStream from where a homography has been found,
    # and calculate mean and std over all constituent frames.
    # This will not only constitute our background (for object detection),
    # but also be used to check whether a given frame is actually in play (so we can segment the stream.)
    normBF = sf.get_normal_bool(courtSF['courtLines'].dropna())
    normDecStm = NCDecapStream.from_bool(normBF, cap)
    medFrm, stdFrm = normDecStm.frames.median(), normDecStm.frames.std()
    medFrm, stdFrm = medFrm.rename('median_backFrm'), stdFrm.rename('std_backFrm')
    
    medFrm.show()
    stdFrm.show()
    
    medFrm.save_like_feature(VIDEO_ID, STORE_NAME, start_frame = START_FRAME, stop_frame = stop_frame)
    stdFrm.save_like_feature(VIDEO_ID, STORE_NAME, start_frame = START_FRAME, stop_frame = stop_frame)
    
    
    
    
    
    
    
    
    
    
    