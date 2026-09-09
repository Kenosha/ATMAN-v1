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

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, RandStream, NCStream
from ATMANpy.types.Features import ConstFeature

from ATMANpy.types.Points import Point2D

def load(vid, sto):
    pass

def save(H, P, M_int, M_ext, vid, sto):
    ConstFeature('H', H).save(vid, sto)
    ConstFeature('P', P).save(vid, sto)
    ConstFeature('M_int', M_int).save(vid, sto)
    ConstFeature('M_ext', M_ext).save(vid, sto)

# def check(ppl, vid, sto, courtSF, randStm, courtLines, netLines):
#     graphPath_main = ut.get_folder_struct('graphs')['pipeline_checks']
#     graphPath = os.path.join(graphPath_main, ppl._name, vid, sto)
#     graphPath_specific = ut.maybe_make_folder(graphPath, 'court_lines')
    
#     courtSF_found = courtSF.filter(courtSF['trackFlag'] == 'Found')
#     ex_frame_num = courtSF_found.get_frames()[0]
#     exampleFrm = randStm.get(ex_frame_num).to_BGR()
    
#     saveFrm = exampleFrm.put(courtLines).put(netLines)
#     saveFrm.save_as_png(graphPath_specific, 'court_and_net')


def main(conf, courtline_params, name = None):
    
    vid, sto = conf['video_id'], conf['store_name']
    start, stop = conf['start_frame'], conf['stop_frame']
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    scale_factor = conf['scale_factor']
    
    ### Read Courtlines
    # -> since camera is fixed, lines are simply a parameter! we will make a single homography by getting a few random shots
    sample_count = min(conf.get('court_sample_frames', 1000), stop - start + 1)
    randStm = RandStream(cap, conf['start_frame'], conf['stop_frame'], number_of_frames = sample_count,
                         cap_color_space = 'bgr', target_color_space = 'grey')
    
    courtSF = randStm.extract_mp(ex.get_courtlines, nwo, **courtline_params)
    
    ### Average out courtlines found.
    HSF = courtSF['H'].dropna()
    if HSF.length() == 0:
        raise ValueError('No court could be calibrated from the sampled video frames.')
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
    
    P = sa.calibrate_camera(courtLines, netLines, width = 'Singles')
    M_int, M_ext = sa.find_intrinsic_extrinsic_M(P, normalize_K = True)
    
    if np.isnan(M_ext).any():
        print('Decomposition of P-Matrix failed!')
        ### Wait! This could be because the net line lies directly on the projected line between corner points!
        ### This might be what fucks it up!!!
        ### Update: Not really, sadly. see above.
    else:
        save(H, P, M_int, M_ext, vid, sto)
    
    ###### DO CHECKS ######
    # check(ppl, vid, sto, courtSF, randStm, courtLines, netLines)
    
if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default

    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    courtline_params = prm_mod.court_lines.courtline_params
    
    name = ppl._name
    main(conf, courtline_params, name = name)

    # ### Check the quality of line finding!
    # myStm = Stream(cap, conf['start_frame'], conf['stop_frame'])
    # myStm.get(4000, put = False).put(courtLines).put(netLines).show()
    
    # myStm.get(4000, put = False).put(courtLines.get_court_region('Singles')).show()
    # schemaLines.get_3D_corner_points('Singles')
    # schemaLines.get_3D_corner_points('Doubles')
    
    
    
    
    

