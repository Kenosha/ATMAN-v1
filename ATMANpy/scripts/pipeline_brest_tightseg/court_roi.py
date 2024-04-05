# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:45:37 2023

@author: gianl
"""

from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import get_width_height
from ATMANpy.funcs.feature_funcs import extract_region_values

import ATMANpy.funcs.sf_apply_funcs as sa

from ATMANpy.types.Streams import RandStream
from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.Lines import Region, CourtLines
from ATMANpy.types.Points import Point2D


def load(vid, sto):
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    return HSF

def save(courtCF, topsrvCF, botsrvCF, vid, sto):
    courtCF.rename('courtroiReg').save(vid, sto, extract_func = extract_region_values)
    topsrvCF.rename('topserveReg').save(vid, sto, extract_func = extract_region_values)
    botsrvCF.rename('botserveReg').save(vid, sto, extract_func = extract_region_values)

# def check(cap, start, stop, courtROI, topserveROI, botserveROI):
#     from random import randint 
    
#     NUM_FRAMES = 1000
#     randStm = RandStream(cap, start, stop, number_of_frames = NUM_FRAMES,
#                          cap_color_space = 'bgr', target_color_space = 'grey')
    
#     keep_showing = True
#     while keep_showing:
        
#         some_frame = randint(0, NUM_FRAMES)
#         someFrm = randStm.get(some_frame)
    
#         someFrm.put(courtROI).show()
#         someFrm.put(topserveROI).show()
#         someFrm.put(botserveROI).show()
        
#         againStr = input('show again? (y/n): ')
#         keep_showing = againStr == 'y'
        
def main(conf, courtroi_params, serveroi_params):
    vid, sto = conf['video_id'], conf['store_name']
    start, stop = conf['start_frame'], conf['stop_frame']
    cap = get_video(vid)
    width, height = get_width_height(cap)
    
    ### Serve ROIs
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    H = HSF.feature
    schemaLines = CourtLines.make().add_singles_netline()
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    
    p1_pxs, p2_pxs, p3_pxs, p4_pxs = courtroi_params['p1'], courtroi_params['p2'], courtroi_params['p3'], courtroi_params['p4']
    p1, p2, p3, p4 = Point2D(p1_pxs), Point2D(p2_pxs), Point2D(p3_pxs), Point2D(p4_pxs)
    courtROI = Region.from_points(p1, p2, p3, p4)
    courtCF = ConstFeature('courtroiReg', courtROI)
    
    # a) Top Serve
    dn_px, up_px = serveroi_params['top_down'], serveroi_params['top_up']
    loLn, hiLn = courtLines['Top Baseline'].down(dn_px), courtLines['Top Baseline'].up(up_px)
    p1, p2, p3, p4 = hiLn.leftPt, hiLn.rightPt, loLn.leftPt, loLn.rightPt
    topserveROI = Region.from_points(p1, p2, p3, p4)
    topsrvCF = ConstFeature('topserveReg', topserveROI)
    
    # b) Bot Serve
    dn_px, up_px = serveroi_params['bot_down'], serveroi_params['bot_up']
    loLn, hiLn = courtLines['Bottom Baseline'].down(dn_px), courtLines['Bottom Baseline'].up(up_px)
    p1, p2, p3, p4 = hiLn.leftPt, hiLn.rightPt, loLn.leftPt, loLn.rightPt
    botserveROI = Region.from_points(p1, p2, p3, p4)
    botsrvCF = ConstFeature('botserveReg', botserveROI)
    
    # check(cap, start, stop, courtROI, topserveROI, botserveROI)
    
    save(courtCF, topsrvCF, botsrvCF, vid, sto)
       
if __name__ == '__main__':
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
    
    conf = cnf_mod.conf
    courtroi_params = prm_mod.court_roi.courtroi_params
    serveroi_params = prm_mod.court_roi.serveroi_params
    
    main(conf, courtroi_params, serveroi_params)
    
    