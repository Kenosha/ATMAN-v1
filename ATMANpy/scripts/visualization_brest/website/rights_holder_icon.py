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

from ATMANpy.types.ShapesTennis3D import Arrow
from ATMANpy.types.Shapes3D import Circle3D

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'rights_holder_icon'
SAVE_FOLDER = 'website_snippets'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]


from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

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
cap = confs[0]['cap']


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

dataPnts = Points.from_datatable(replTable, split_by_match = True)
# nullPnts = Points.from_placeholder_table(nullTable)

# allPnts = dataPnts.join(nullPnts).sort()
from ATMANpy.types.Overlays import scoreboard

##################

def shorten_scoreboard(scoreFrm):
    lfFrm, rtFrm = scoreFrm[:, :135], scoreFrm[:, 192:]
    shortFrm = BGRFrame(np.concatenate([lfFrm.array, rtFrm.array], axis = 1))
    return shortFrm

def make_sequential_scoreboards(conf_idx, winner = 'upper', upper_idx = 0, lower_idx = 0, 
                                repl_name_upper = None, repl_name_lower = None):
    vid, sto = confs[conf_idx]['video_id'], confs[conf_idx]['store_name']
    scoreSF = StreamFeatures.from_store(vid, sto, featnameList = [  'sb_games_upper_cleaned_Flag',
                                                                    'sb_games_lower_cleaned_Flag'])
    nameSF = StreamFeatures.from_store(vid, sto, featnameList = ['sb_name_lower_Flag',
                                                                'sb_name_upper_Flag'])

    from ATMANpy.scripts.visualization_brest.website.utils import join_games

    scores = list(scoreSF.apply(join_games).values())
    unique_idxs = np.unique(scores, return_index = True)[1]
    sorted_idxs = sorted(unique_idxs)
    unique_scores = [scores[idx] for idx in sorted_idxs]

    upper_name = nameSF['sb_name_upper_Flag'].value_counts().index[upper_idx]
    lower_name = nameSF['sb_name_lower_Flag'].value_counts().index[lower_idx]

    ### correction for Van Assche
    if upper_name == 'L.VANASSCHE':
        upper_name = 'L.VAN ASSCHE'
    if lower_name == 'L.VANASSCHE':
        lower_name = 'L.VAN ASSCHE'

    ### replacement names
    if repl_name_upper is not None:
        upper_name = repl_name_upper
    if repl_name_lower is not None:
        lower_name = repl_name_lower

    scoreboards = []
    for score in unique_scores:
        upper_score, lower_score = score.replace('-', '').split('__')

        ### who's serving
        upper_sum = sum([int(game_score) for game_score in upper_score])
        lower_sum = sum([int(game_score) for game_score in lower_score])

        serving = (np.ceil((upper_sum + lower_sum) / 2)) % 2 + 1

        upper_score = upper_score + (3 - len(upper_score)) * ' '
        lower_score = lower_score + (3 - len(lower_score)) * ' '

        scoreFrm = scoreboard(upper_name, lower_name, 
                                '', '', upper_score, lower_score, 
                                serving = serving,
                                frame_thickness = 2)
        scoreboards.append(scoreFrm)

    upper_score, lower_score = score.replace('-', '').split('__')

    if winner == 'upper':
        upper_score =  f'{upper_score[:-1]}{int(upper_score[-1]) + 1}'
    elif winner == 'lower':
        lower_score = f'{lower_score[:-1]}{int(lower_score[-1]) + 1}'

    upper_score = upper_score + (3 - len(upper_score)) * ' '
    lower_score = lower_score + (3 - len(lower_score)) * ' '

    scoreFrm = scoreboard(upper_name, lower_name, 
                            '', '', upper_score, lower_score, 
                            serving = None,
                            frame_thickness = 2,
                            winner = winner)
    scoreboards.append(scoreFrm)

    ### shorten
    scoreboards = [shorten_scoreboard(sb) for sb in scoreboards]

    return scoreboards

final_boards = make_sequential_scoreboards(0, upper_idx = 0, lower_idx = 1, winner = 'upper')

semi1_boards = make_sequential_scoreboards(1, upper_idx = 0, lower_idx = 0, winner = 'lower')
semi2_boards = make_sequential_scoreboards(2, upper_idx = 0, lower_idx = 1, winner = 'lower')

qurt1_boards = make_sequential_scoreboards(3, upper_idx = 0, lower_idx = 1, winner = 'lower')
qurt2_boards = make_sequential_scoreboards(4, upper_idx = 0, lower_idx = 0, winner = 'upper')
# qurt3_boards = make_sequential_scoreboards(5, upper_idx = 0, lower_idx = 0, winner = 'upper')
qurt4_boards = make_sequential_scoreboards(6, upper_idx = 0, lower_idx = 1, winner = 'lower')

repl_boards = make_sequential_scoreboards(7, upper_idx = 0, lower_idx = 0,
                                          repl_name_upper = 'R.GASQUET', repl_name_lower = 'M.JANVIER',
                                          winner = 'lower')

### TODO: when ready to animate, put line_thickness of scoreboard to 0.01 and up the resolution to 15000
######### (at least for the last, winner frame thing)


#### A LOOK AT FINAL BOARDS
# final_boards[-1].show()

# semi1_boards[-1].show()
# semi2_boards[-1].show()

# qurt1_boards[-1].show()
# qurt2_boards[-1].show()
# # qurt3_boards[0].show()
# qurt4_boards[-1].show()

# repl_boards[-1].show()

blackFrm = BGRFrame.make_black(cols = 640, rows = 360).rescale(1.5)

len(qurt1_boards), len(qurt2_boards), len(repl_boards), len(qurt4_boards)

rs, cs, _ = qurt1_boards[0].shape
tot_rs, tot_cs, _ = blackFrm.shape

cPt = Point2D((tot_cs // 2, tot_rs // 2))
cPt_adj = cPt.up(rs // 2).left(cs // 2)

spacing_l_r = 0.2
spacing_u_d = 0.2

width_l_r = int((1 + spacing_l_r) * cs)
width_u_d = int((1 + spacing_u_d) * rs)

fPt = cPt_adj.right(width_l_r)

s1Pt = cPt_adj.up(int(1.5 * width_u_d))
s2Pt = cPt_adj.down(int(1.5 * width_u_d))

q1Pt = cPt_adj.left(width_l_r).up(2 * width_u_d)
q2Pt = cPt_adj.left(width_l_r).up(1 * width_u_d)
q3Pt = cPt_adj.left(width_l_r).down(1 * width_u_d)
q4Pt = cPt_adj.left(width_l_r).down(2 * width_u_d)

###### A LOOK AT PLACEMENT 
# fFrm = overlay(blackFrm, final_boards[-1], top_left_point = fPt)

# sFrm = overlay(fFrm, semi1_boards[-1], top_left_point = s1Pt)
# sFrm = overlay(sFrm, semi2_boards[-1], top_left_point = s2Pt)

# qFrm = overlay(sFrm, qurt1_boards[-1], top_left_point = q1Pt)
# qFrm = overlay(qFrm, qurt2_boards[-1], top_left_point = q2Pt)
# qFrm = overlay(qFrm, repl_boards[-1], top_left_point = q3Pt)
# qFrm = overlay(qFrm, qurt4_boards[-1], top_left_point = q4Pt)

# qFrm.show()


