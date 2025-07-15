import os 
import numpy as np
import pandas as pd
        
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames, MaskFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Visual3D import Camera, View, Scene

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D

from ATMANpy.types.Abstract import NullPoint, Points, Shots, read_in_data_tables, read_in_placeholder_tables, read_in_colinear_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

#############################
#### READ IN CONFIGS ########
#############################
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import collect_video_frames

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

###################################
######### READ IN DATA ############
###################################
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)
coliTable = read_in_colinear_tables(confs)

### replace points suspected of co-linearity with the co-linear stand-in
to_replace = [(0, 110, 1), (0, 112, 2)] ### match index, point index, shot index

replTable = dataTable.copy()

for mtc_idx, pnt_idx, sht_idx in to_replace:
    mtc_coliTable = coliTable.loc[mtc_idx]
    mtc_replTable = replTable.loc[mtc_idx]

    sht_coliTable = mtc_coliTable[(mtc_coliTable['point_idx'] == pnt_idx) & (mtc_coliTable['shot_idx'] == sht_idx)]
    mtc_replTable.loc[sht_coliTable.index] = sht_coliTable

dataPnts = Points.from_datatable(replTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

### PARAMETERS

DIMS = (1920 // 3, 1080 // 3)

### FILTER ANIMATION
import ATMANpy.funcs.filter_shots_funcs as filt_s
import ATMANpy.funcs.filter_points_funcs as filt_p

def is_tracked(somePnt):
    return not isinstance(somePnt, NullPoint)

def is_returned_by(somePnt, name):
    return filt_p.is_returner(somePnt, name = name) and filt_p.is_longer_than(somePnt, threshold = 1)

def is_first_serve(somePnt):
    return somePnt.is_first_serve()

def is_serve_ad(somePnt):
    return somePnt.is_serve_ad()

def is_serve_out_wide(somePnt):
    return filt_p.is_serve_out_wide(somePnt)

def summ_pnts(somePnts):
    return f'{len(somePnts)} points available.'

def summ_pnts2(somePnts):
    return f'{len(somePnts)} points remaining.'

allPnts
s1 = summ_pnts(allPnts)
trackPnts = allPnts.filter(is_tracked)
s2 = summ_pnts2(trackPnts)
filtPnts = trackPnts.filter(is_returned_by, name = 'Brandon Nakashima')
s3 = summ_pnts2(filtPnts)
filtPnts = filtPnts.filter(is_first_serve)
winArr = vis_ut.get_winners_from_score(filtPnts, allPnts)
decArr = winArr != 0
filtPnts = filtPnts.filter(decArr)
s4 = summ_pnts2(filtPnts)
filtPnts = filtPnts.filter(is_serve_ad)
s5 = summ_pnts2(filtPnts)
filtPnts = filtPnts.filter(is_serve_out_wide)
s6 = summ_pnts2(filtPnts)
ss = [s1, s2, s3, s4, s5, s6]

p1 = None
p2 = 'is_tracked'
p3 = 'is_returned_by, "Brandon Nakashima"'
p4 = 'is_first_serve'
p5 = 'is_serve_ad'
p6 = 'is_serve_out_wide'
ps = [p1, p2, p3, p4, p5, p6]

l1 = 'dataPnts'
l2 = f'filtPnts = dataPnts.filter({p2})'
l3 = f'filtPnts = filtPnts.filter({p3})'
l4 = f'filtPnts = filtPnts.filter({p4})'
l5 = f'filtPnts = filtPnts.filter({p5})'
l6 = f'filtPnts = filtPnts.filter({p6})'
ls = [l1, l2, l3, l4, l5, l6]

frame_offset = len('filtPnts = filtPnts.filter(')

from ATMANpy.types.Overlays import typewriter_effect
import cv2 as cv

type_kwargs = {'frames_per_char': 1, 'text_size': 1, 'text_font': cv.FONT_HERSHEY_PLAIN}
code_color = (0, 255, 0) #const.BGR['leafy_green']
emph_color = const.BGR['soft_yellow']
cons_color = const.BGR['bright_grey']
arrw_color = const.BGR['bright_blue']
left_base_x = 37
right_base_x = 400
offset_y = 20
offset_x_emph = 203

curr_y = offset_y

all_frames = []
currFrm = BGRFrame.make_black(*DIMS)
currFrm_with_arr = currFrm.put('\n>>', color = emph_color, where = (10, curr_y), font = cv.FONT_HERSHEY_PLAIN)

for l, p, s in zip(ls, ps, ss):

    outFrms = typewriter_effect(l, **type_kwargs, overFrm = currFrm, 
                                text_where = (left_base_x, curr_y), 
                                text_color = code_color)
    
    if p is not None:
        if 'data' in l:
            pos_x = left_base_x + offset_x_emph + 12
        else:
            pos_x = left_base_x + offset_x_emph

        emphFrms = typewriter_effect(p, **type_kwargs, overFrm = BGRFrame.make_black(*DIMS),
                                     text_where = (pos_x, curr_y),
                                     text_color = emph_color)
        
        prevFrms = BGRFrames.from_frame(BGRFrame.make_black(*DIMS), frame_offset)
        postFrms = BGRFrames.from_frame(emphFrms[-1], 1)
        emphFrms = BGRFrames.from_frames([prevFrms, emphFrms, postFrms])
    else:
        emphFrms = BGRFrames.from_frame(currFrm, outFrms.length())

    outFrms = outFrms + emphFrms
    outFrms_with_arr = outFrms.put('\n>>', color = arrw_color, where = (10, curr_y), font = cv.FONT_HERSHEY_PLAIN)

    currFrm = outFrms[-1]

    curr_y += offset_y
    
    statFrms_1 = BGRFrames.from_frame(currFrm, 5)

    outFrms = typewriter_effect(f'{s:>25}', **type_kwargs, overFrm = currFrm, 
                                text_where = (right_base_x, curr_y), text_color = cons_color)
    statFrms_2 = BGRFrames.from_frame(outFrms[-1], 15)

    currFrm = statFrms_2[-1]
    
    curr_y += 2 * offset_y
    
    statFrms_1_with_arr = statFrms_1.put('\n>>', color = arrw_color, where = (10, curr_y), font = cv.FONT_HERSHEY_PLAIN)
    statFrms_2_with_arr = statFrms_2.put('\n>>', color = arrw_color, where = (10, curr_y), font = cv.FONT_HERSHEY_PLAIN)

    ### add little arrows for line indicator
    all_frames.append(outFrms_with_arr)
    all_frames.append(statFrms_1)
    all_frames.append(statFrms_2_with_arr)

jointFrms = BGRFrames.from_frames(all_frames)
# DecapStream(jointFrms, 0, 100).show()

jointFrms[-1].show()









