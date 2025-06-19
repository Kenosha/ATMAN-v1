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

from ATMANpy.models.ParticleFilter import ParticleFilter
from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points

SAVE = False

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'short_intro'
SAVE_FOLDER = 'atman_capabilities'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())

if SAVE:
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

#### TESTING COLIPOINTS ####
coliPnts = Points.from_datatable(coliTable, split_by_match = False)

prevPnt3_coli = coliPnts.get(0, 110)
prevPnt2_coli = coliPnts.get(0, 111)
prevPnt1_coli = coliPnts.get(0, 112)
lastPnt_coli = coliPnts.get(0, 113)

# coliPnts.get(0, 112).stream(confs = confs).show()

########## Filtering for ad wide serves before the current one

import ATMANpy.funcs.filter_shots_funcs as filt_s
import ATMANpy.funcs.filter_points_funcs as filt_p

filtPnts1 = dataPnts.filter(filt_p.is_returner, name = 'Brandon Nakashima').filter(filt_p.is_longer_than, threshold = 1)

filtPnts2 = filtPnts1.filter('is_first_serve')
winArr = np.array(vis_ut.get_winners_from_score(filtPnts2, allPnts))
decArr = winArr != 0

filtPnts3 = filtPnts2.filter(decArr)
filtPnts4 = filtPnts3.filter('is_serve_deuce')
filtPnts5 = filtPnts4.filter(filt_p.is_serve_out_wide)

filtPnts_before = filtPnts5.filter(filt_p.is_not_server, name = 'Joao Sousa')
filtPnts_after = filtPnts5.filter(filt_p.is_server, name = 'Joao Sousa')

##############################
######### CAMERAS ############
##############################

brestCam_third = Camera.calibrated_brest_behind()
brestCam_half = brestCam_third.resize(1.5)
brestCam_full = brestCam_third.resize(3)

##############################
######### PARAMETERS #########
##############################

# prmDict = {'background_resolution': 0.1, 'argument_resolution': 0.1, 'foreground_resolution': 0.1,
#            'interpolate_func': vs.interpolate_camera_naive}
prmDict = {'background_resolution': 2, 'argument_resolution': 10, 'foreground_resolution': 10,
           'interpolate_func': vs.interpolate_camera_naive} ### TAKES VERY LONG! TRY LOWER OR DO SOMETHING ELSE IN THE MEANTIME

######################################
### CHECK DISTRO OF BEFORE RETURNS ###
######################################

# retShots = filtPnts_before.get_returns()
# retShots_rect = retShots.enforce_direction('Top')

# retShots_rect_out = retShots_rect.filter('is_fair', nextShot = None, negate = True)
# retShots_rect_in = retShots_rect.filter('is_fair', nextShot = None)

# heatScn_in = vs.kde_heatmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'],
#                             bandwidth = 40, resolution = 2, normalize_density = True)

# crtView = View.make(brestCam_full, Scene.make_court())
# heatView = View.make(brestCam_full, heatScn_in)
# # crtView.combine(heatView).snap().show()

# markScn_in = vs.scatter_markmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'])
# markScn_out = vs.scatter_markmap(retShots_rect_out, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'])

# crtView = View.make(brestCam_full, Scene.make_court())
# markView = View.make(brestCam_full, markScn_in)
# # crtView.combine(markView).snap().show()

# crtView = View.make(brestCam_full, Scene.make_court())
# heatView = View.make(brestCam_full, markScn_out)
# # crtView.combine(heatView).snap().show()

### LOOK AT CHOSEN CANDIDATES ####
best_idx = 14
lastPnt = filtPnts_after[best_idx]
mtc_idx, pnt_idx = lastPnt.get_match_index(), lastPnt.get_point_index()

prevPnt1 = allPnts.get(mtc_idx, pnt_idx - 1)
prevPnt2 = allPnts.get(mtc_idx, pnt_idx - 2)
# prevPnt3 = allPnts.get(mtc_idx, pnt_idx - 3)

# animFrms3 = vis_ut.animate_bounce_and_mark_scenes(prevPnt3, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, over = True,
#                                                     sub_region = False, argument_resolution = 10)
# # DecapStream(animFrms3, 0, 100).show()

mark_color = const.BGR['bright_blue']
mark_size = 6

animFrms2 = vis_ut.animate_bounce_and_mark_scenes(prevPnt2, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, over = True,
                                                    sub_region = False,  mark_size = mark_size, mark_color = mark_color,
                                                    **prmDict)
# DecapStream(animFrms2, 0, 100).show()

animFrms1 = vis_ut.animate_bounce_and_mark_scenes(prevPnt1, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, over = True,
                                                    sub_region = False, mark_size = mark_size, mark_color = mark_color,
                                                    **prmDict)
# DecapStream(animFrms1, 0, 100).show()

animFrms0 = vis_ut.animate_bounce_and_mark_scenes(lastPnt, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, over = True,
                                                    sub_region = False, mark_size = mark_size, mark_color = mark_color,
                                                    **prmDict)
# DecapStream(animFrms0, 0, 100).show()

### figure out intermezzi.
initial_widen_by = 450 + 21 + 12 + 9 + 12 ### -> lineFrms_silent, lineFrms_voice, playerFrms_silent, playerFrms_voice
preFrms = prevPnt2.stream(confs = confs).widen(initial_widen_by).decap().frames[:initial_widen_by].rescale(3)

# frms3 = prevPnt3.get_frames()
frms2 = prevPnt2.get_frames()
frms1 = prevPnt1.get_frames()
frms0 = lastPnt.get_frames()

# d32 = frms2[0] - frms3[-1]
d21 = frms1[0] - frms2[-1]
d10 = frms0[0] - frms1[-1]

# interFrms32 = prevPnt2.stream(confs = confs).widen(d32).decap().frames[:d32].rescale(3)
interFrms21 = prevPnt1.stream(confs = confs).widen(d21).decap().frames[:d21].rescale(3)
interFrms10 = lastPnt.stream(confs = confs).widen(d10).decap().frames[:d10].rescale(3)

# interFrms32_speed = vis_ut.speed_up_frames_progressively(interFrms32, ramp_up_frames = 25)
interFrms21_speed = vis_ut.speed_up_frames_progressively(interFrms21, ramp_up_frames = 25)
interFrms10_speed = vis_ut.speed_up_frames_progressively(interFrms10, ramp_up_frames = 150, ramp_down_frames = 150)

#TODO: vis_ut.show feels fast

#####################################
#### ATMAN contraction animation ####
#####################################

from ATMANpy.types.Shapes3D import Point3D
z = 1000
oPt, cPt = Point3D((0, 0, 0)), Point3D((0, 0, z))
downCam = Camera.from_to(cPt, oPt)
text_color = const.BGR['leafy_green']
text_size = 2
left_pts = 325

titleStr = 'The Automated Tennis Match ANnotator'
titlePt = oPt.annotate(titleStr).left(left_pts).recolor(text_color).resize(text_size)
titleView = View.make(downCam, Scene.make_empty(), annotatedPoints = [titlePt])
titleFrm = titleView.snap()
# titleFrm.show()

auxil_offset = 0
move1 = left_pts - 76
move2 = left_pts - 270
move3 = left_pts - 392
move4 = left_pts - 506
move5 = left_pts - 526

p1 = oPt.annotate('A').front(auxil_offset).left(move1).recolor(text_color).resize(text_size)
p2 = oPt.annotate('T').front(auxil_offset).left(move2).recolor(text_color).resize(text_size)
p3 = oPt.annotate('M').front(auxil_offset).left(move3).recolor(text_color).resize(text_size)
p4 = oPt.annotate('A').front(auxil_offset).left(move4).recolor(text_color).resize(text_size)
p5 = oPt.annotate('N').front(auxil_offset).left(move5).recolor(text_color).resize(text_size)

target1 = - 40
target2 = - 20
target3 = 0
target4 = 25
target5 = 47

p1_target = p1.right(target1 + move1)
p2_target = p2.right(target2 + move2)
p3_target = p3.right(target3 + move3)
p4_target = p4.right(target4 + move4)
p5_target = p5.right(target5 + move5)

titleView = View.make(downCam, Scene.make_empty(), annotatedPoints = [titlePt])
titleFrm = titleView.snap()
# titleFrm.show()

letterView = View.make(downCam, Scene.make_empty(), annotatedPoints = [p1, p2, p3, p4, p5])
letterFrm = letterView.snap()
# letterFrm.show()

itpl_func = lambda l: [vs.interpolate_shape(p1, p1_target, l), 
                       vs.interpolate_shape(p2, p2_target, l),
                       vs.interpolate_shape(p3, p3_target, l),
                       vs.interpolate_shape(p4, p4_target, l),
                       vs.interpolate_shape(p5, p5_target, l)]

N = 50
pre_dim_factor = 0.5

pre_staticFrm = BGRFrames.from_frame(titleFrm, 25)

pre_dimFrms = BGRFrames.from_frame(titleFrm, N).dim_progressively(fs_dim_factor = 1, ls_dim_factor = 0.5)
post_dimFrms = BGRFrames.from_frame(titleFrm, N).dim_progressively(fs_dim_factor = 0.5, ls_dim_factor = 0)
lambdas = np.linspace(0, 1, N, endpoint = True)

### deal with pre move dim frames
pre_dimFrms_lettered, _ = vs.animate_sequence([Scene.make_empty() for _ in range(N)], start_cam = downCam, stop_cam = downCam,
                            annotated_points = [p1, p2, p3, p4, p5], animate_over = pre_dimFrms)

### deal with post move dim frames
pt_frames = []
for i in range(N):
    l = lambdas[i]
    frm = post_dimFrms[i]

    annotated_points = itpl_func(l)
   
    ptView = View.make(downCam, Scene.make_empty(), annotatedPoints = annotated_points)
    ptFrm = ptView.snap(over_array = frm.array)
    pt_frames.append(ptFrm)

post_dimFrms_lettered = BGRFrames.from_single_frames(pt_frames)

### deal with preFrms
fade_in_length = 150

post_dimFrms_fade_out = BGRFrames.from_frame(post_dimFrms_lettered[-1], fade_in_length).dim_progressively(1, 0)
preFrms_fade_in = preFrms[:fade_in_length].dim_progressively(0, 1)

preFrms_fade = post_dimFrms_fade_out + preFrms_fade_in
# DecapStream(preFrms_fade, 0, 100).show()

pre_lines_and_players_frames = 40
# post_lines_and_players_frames = 5

line_silent_frames = 21
line_voiced_frames = 12
player_silent_frames = 9
player_voiced_frames = 12

frame_list_in_order = [pre_lines_and_players_frames, line_silent_frames, line_voiced_frames,
                       player_silent_frames, player_voiced_frames]

cumsum = np.cumsum(frame_list_in_order)

lines_and_players_frames = sum(frame_list_in_order)

preFrms_rest = preFrms[fade_in_length:-lines_and_players_frames]

preFrms_last = preFrms[-lines_and_players_frames:]

preFrms_last_pre0 = preFrms_last[:cumsum[0]]                    ### "ATMAN autocamtically recognizes the lines..."     
preFrms_last_pre1 = preFrms_last[cumsum[0]:cumsum[1]]           ### line; silent
preFrms_last_pre2 = preFrms_last[cumsum[1]:cumsum[2]]           ### line; voice (says "the players")
preFrms_last_pre3 = preFrms_last[cumsum[2]:cumsum[3]]           ### players; silent
preFrms_last_pre4 = preFrms_last[cumsum[3]:]                    ### players; voiced (says "the ball")

introFrms = BGRFrames.from_frames([pre_staticFrm, pre_dimFrms_lettered, post_dimFrms_lettered, preFrms_fade, preFrms_rest])
# DecapStream(introFrms, 0, 100).show()

checkFrms = BGRFrames.from_frames([preFrms_last_pre0, preFrms_last_pre1, preFrms_last_pre2, 
                                   preFrms_last_pre3, preFrms_last_pre4])
# DecapStream(checkFrms, 0, 100).show()

# DecapStream(BGRFrames.from_frames([introFrms, checkFrms]), 0, 100).show()

###############################
### TEASE LINES AND PLAYERS ###
###############################

line_frame_fs = prevPnt2.get_frames()[0] - 1 - lines_and_players_frames + pre_lines_and_players_frames
line_frame_ls = line_frame_fs + line_silent_frames + line_voiced_frames
line_frames = list(range(line_frame_fs, line_frame_ls))

player_frame_fs = line_frame_ls
player_frame_ls = player_frame_fs + player_silent_frames + player_voiced_frames
player_frames = list(range(player_frame_fs, player_frame_ls))

H = ConstFeature.from_store(vid, sto, featname = 'H').feature
courtLines = sa.find_courtlines(H).drop('Net Line').rescale(3)

plSF = StreamFeatures.from_store(vid, sto, featnameList = ['botPlayerCnt', 'topPlayerCnt'])
plSF_small = plSF[player_frames]

player_contours = list(plSF_small.values())
player_contours_silent = player_contours[:player_silent_frames]
player_contours_voiced = player_contours[player_silent_frames:]

len(player_contours_silent), len(player_contours_voiced)

botCntSF, topCntSF = plSF_small['botPlayerCnt'], plSF_small['topPlayerCnt']

preFrms_last_pre1_feature = preFrms_last_pre1.put(courtLines, color = const.BGR['sign_orange'], thickness = 4)
preFrms_last_pre2_feature = preFrms_last_pre2.put(courtLines, color = const.BGR['sign_orange'], thickness = 4)

my_frames = []
for someFrm, top_bot_cnts in zip(preFrms_last_pre3, player_contours_silent):
    topCnt, botCnt = top_bot_cnts['topPlayerCnt'].rescale(3), top_bot_cnts['botPlayerCnt'].rescale(3)
    cntFrm = someFrm.put([topCnt, botCnt], color = const.BGR['sign_orange'], thickness = 4)
    my_frames.append(cntFrm)

preFrms_last_pre3_feature = BGRFrames.from_single_frames(my_frames)

my_frames = []
for someFrm, top_bot_cnts in zip(preFrms_last_pre4, player_contours_voiced):
    topCnt, botCnt = top_bot_cnts['topPlayerCnt'].rescale(3), top_bot_cnts['botPlayerCnt'].rescale(3)
    cntFrm = someFrm.put([topCnt, botCnt], color = const.BGR['sign_orange'], thickness = 4)
    my_frames.append(cntFrm)

preFrms_last_pre4_feature = BGRFrames.from_single_frames(my_frames)

# chkFrms = BGRFrames.from_frames([preFrms_last_pre3_feature, preFrms_last_pre4_feature])
# DecapStream(chkFrms, 0, 100).show()

# offFrm = preFrms_last_post[0]
# onFrm_line = offFrm.put(courtLines, color = const.BGR['sign_orange'], thickness = 4)

# lineFrms_blink = vs.blinking(offFrm, onFrm_line, blinks_per_second = 4, duration_in_frames = 50)
# # DecapStream(lineFrms_blink, 0, 100).show()

# onFrm_players = offFrm.put([botCnt, topCnt], color = const.BGR['sign_orange'], thickness = 4)

# playerFrms_blink = vs.blinking(offFrm, onFrm_players, blinks_per_second = 4, duration_in_frames = 50)
# # DecapStream(playerFrms_blink, 0, 100).show()

# preFrms_last = BGRFrames.from_frames([preFrms_last_pre, lineFrms_blink, playerFrms_blink])
 
# ATMAN automatically recognizes the lines, [PAUSE] the players, [PAUSE] and the ball ## 7 seconds
# DecapStream(preFrms_last, 0, 100).show()

###############################
##### FIRST RALLY TO SHOW #####
###############################
# TODO: idea: first time we show the shot type with the point desc. box, freeze and show it next to the player making the shot.
### (maybe with highlighting -> speed, spin, launch angle.)
### move the shot descriptors only when the overhead view is introduced
# TODO: also add a red x where the ball lands in the net

num_frames = 250
fade_in_frames = 25

dim_factor = 0.5
frm_offset = 4 ### so the motion is complete.

cols, rows = 243, 100
back_color = (60, 60, 60)
text_color = const.BGR['leafy_green']
o_x, o_y = 10, 20
para_px = 25
skip_px = 18

fullFrms_hi0 = vis_ut.make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), 
                                                para_px = para_px, skip_px = skip_px,
                                                highlight_which = 0,
                                                back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                                border_color = const.BGR['bright_grey'], border_thickness = 2)

fullFrms_hi1 = vis_ut.make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), 
                                                para_px = para_px, skip_px = skip_px,
                                                highlight_which = 1,
                                                back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                                border_color = const.BGR['bright_grey'], border_thickness = 2)

fullFrms_hi2 = vis_ut.make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), 
                                                para_px = para_px, skip_px = skip_px,
                                                highlight_which = 2,
                                                back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                                border_color = const.BGR['bright_grey'], border_thickness = 2)

fullFrms_hi3 = vis_ut.make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), 
                                                para_px = para_px, skip_px = skip_px,
                                                highlight_which = 3,
                                                back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                                border_color = const.BGR['bright_grey'], border_thickness = 2)

fs_frm = prevPnt2.get_frames()[0]
frm4 = prevPnt2[-4].get_frames()[0] - fs_frm + frm_offset
frm3 = prevPnt2[-3].get_frames()[0] - fs_frm + frm_offset
frm2 = prevPnt2[-2].get_frames()[0] - fs_frm + frm_offset
frm1 = prevPnt2[-1].get_frames()[0] - fs_frm + frm_offset

fullFrm4 = fullFrms_hi0[frm4]
fullFrm3 = fullFrms_hi1[frm3]
fullFrm2 = fullFrms_hi2[frm2]
fullFrm1 = fullFrms_hi3[frm1]

### anim frames split to have little progressive dim
animFrms_pre_normal = animFrms2[:frm4 - 25]
animFrms_pre_fade = animFrms2[frm4 - 25:frm4].dim_progressively(1, dim_factor)
animFrms_pre = BGRFrames.from_frames([animFrms_pre_normal, animFrms_pre_fade])

stopFrms4 = animFrms2[frm4:frm3]
stopFrms3 = animFrms2[frm3:frm2]
stopFrms2 = animFrms2[frm2:frm1]
stopFrms1 = animFrms2[frm1:]

combFrms4 = overlay(stopFrms4.dim(dim_factor), fullFrm4, top_left_point = Point2D((1100, 40)))
combFrms3 = overlay(stopFrms3.dim(dim_factor), fullFrm3, top_left_point = Point2D((1200, 540)))
combFrms2 = overlay(stopFrms2.dim(dim_factor), fullFrm2, top_left_point = Point2D((1100, 40)))
combFrms1 = overlay(stopFrms1.dim(dim_factor), fullFrm1, top_left_point = Point2D((1300, 540)))

combFrms4_freeze = BGRFrames.from_frame(combFrms4[0], 50)
combFrms3_freeze = BGRFrames.from_frame(combFrms3[0], 25)
combFrms2_freeze = BGRFrames.from_frame(combFrms2[0], 25)
combFrms1_freeze = BGRFrames.from_frame(combFrms1[0], 75)

combFrms4_joint = BGRFrames.from_frames([combFrms4_freeze, combFrms4])
combFrms3_joint = BGRFrames.from_frames([combFrms3_freeze, combFrms3])
combFrms2_joint = BGRFrames.from_frames([combFrms2_freeze, combFrms2])
combFrms1_joint = BGRFrames.from_frames([combFrms1_freeze, combFrms1])

# vis_ut.show([combFrms4_joint, combFrms3_joint, combFrms2_joint, combFrms1_joint])

### disappear the box, and brighten the frames during the intermezzo
fade_out_frames = 60
cam_move_frames = 120
# interFrms21_speed = vis_ut.speed_up_frames_progressively(interFrms21, ramp_up_frames = fade_out_frames, ramp_down_frames = cam_move_frames)
interFrms21_speed = vis_ut.speed_up_frames_progressively(interFrms21, ramp_up_frames = 25, ramp_down_frames = 25)
interFrms21_speed_begin, interFrms21_speed_rest = interFrms21_speed[:fade_out_frames], interFrms21_speed[fade_out_frames:]

_, markScns = vis_ut.get_shot_scenes(prevPnt2, mark_size = mark_size, mark_color = mark_color)
markScn = markScns[-1]

alpha_factors = np.linspace(1, 0, num = interFrms21_speed_begin.length())

markScns_fade = []
for alpha in alpha_factors:
    markScns_fade.append(markScn.realpha(alpha))

interFrms21_speed_begin_fade_marks, _ = vs.animate_sequence(markScns_fade, start_cam = brestCam_full, stop_cam = brestCam_full,
                                                            animate_over = interFrms21_speed_begin, 
                                                            **prmDict)
# DecapStream(interFrms21_speed_begin_fade_marks, 0, 100).show()

interFrms21_speed_fade = overlay(interFrms21_speed_begin_fade_marks.dim_progressively(dim_factor, 1), fullFrm1,
                                 start_alpha = 1, stop_alpha = 0,
                                 top_left_point = Point2D((1300, 540)))
# DecapStream(interFrms21_speed_fade, 0, 100).show()

### move the court into the top down view
interFrms21_speed_middle, interFrms21_speed_last = interFrms21_speed_rest[:-cam_move_frames], interFrms21_speed_rest[-cam_move_frames:]

# DecapStream(interFrms21_speed_middle, 0, 100).show()

M_ext = ConstFeature.from_store(vid, sto, featname = 'M_ext').feature
M_int = ConstFeature.from_store(vid, sto, featname = 'M_int').feature

brestCam_third_fit = Camera(M_int, M_ext, (640, 360))
brestCam_half_fit = brestCam_third.resize(1.5)
brestCam_full_fit = brestCam_third.resize(3)

c_x, c_y, c_z = -900, 900, 1500
downCam = Camera.from_to(Point3D((c_x, c_y, c_z)), Point3D((c_x, c_y, 0)))

courtScn = Scene.make_court(net_mesh_density = 0.75)

alpha_factors = np.linspace(0.055, 1, num = interFrms21_speed_last.length())

# blue_color = interFrms21_speed_last[0].put(Point2D((300, 700)), size = 3).show()
blue_color = tuple(interFrms21_speed_last[0].array[700, 300])

from ATMANpy.funcs.util_funcs import interpolate_tuples

court_scenes_fadein = []
for alpha in alpha_factors:
    itpl_color = interpolate_tuples(blue_color, back_color, alpha)

    subReg = CourtLines.make().get_court_region().scale(1.25)
    # subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(back_color).realpha(alpha)
    subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(back_color).realpha(alpha)
    curr_courtScn = courtScn.put(subReg3D)
    court_scenes_fadein.append(curr_courtScn)

# View.make(brestCam_full_fit, court_scenes_fadein[0], resolution = 10).snap(over_array = interFrms21_speed_last[0].array).show()

moveFrms, _ = vs.animate_sequence(court_scenes_fadein, start_cam = brestCam_full_fit, stop_cam = downCam,
                                  animate_over = interFrms21_speed_last,
                                  **prmDict)


animFrms_down = vis_ut.animate_bounce_and_mark_scenes(prevPnt1, start_cam = downCam, stop_cam = downCam, confs = confs, 
                                                        background_scene = Scene.make_court(net_mesh_density = 0.75), 
                                                        over = True, sub_region = True, region_color = back_color,
                                                        init_over_dim = 1, last_over_dim = 1,
                                                        start_later = 0, stop_earlier = 300,
                                                        mark_size = mark_size * 2, mark_color = mark_color,
                                                        **prmDict)

_, markScns = vis_ut.get_shot_scenes(prevPnt1, mark_size = mark_size * 2, mark_color = mark_color)
markScn = markScns[-1]


fullFrms = vis_ut.make_point_descriptor_box(prevPnt1, dims = (cols, rows), origin = (o_x, o_y), 
                                            para_px = para_px, skip_px = skip_px,
                                            highlight_which = None,
                                            back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                            border_color = const.BGR['bright_grey'], border_thickness = 2)

over_x, over_y = 1635, 650
tlPt = Point2D((over_x, over_y))

animFrms_down_full = overlay(animFrms_down, fullFrms, top_left_point = tlPt)
# DecapStream(animFrms_down, 0, 100).show()
# DecapStream(animFrms_down_full, 0, 100).show()


intermezzo_stop_early = 50 ### stop early so that we have the windup of the serve later.
interFrms10_speed_pre, interFrms_10_speed_post = interFrms10_speed[:-intermezzo_stop_early], interFrms10_speed[-intermezzo_stop_early:]

# DecapStream(interFrms10_speed_pre, 0, 100).show()

animFrms_inter, _ = vs.animate_sequence([curr_courtScn for _ in range(interFrms10_speed_pre.length())], start_cam = downCam, stop_cam = downCam,
                                        previous_foreground_scene = markScn, cumulative_foreground_transparence = 0.99,
                                        animate_over = interFrms10_speed_pre, 
                                        **prmDict)
# DecapStream(animFrms_inter, 0, 100).show()

cntPt = CourtLines.make().get_center_point()
cntPt3D = Point3D.from_point2D(cntPt, z = 0).left(670).back(770)
camPt3D = Point3D((-300, 2000, 600))

thirdCam = Camera.from_to(camPt3D, cntPt3D).zoom(1.1)

trans_frames = 50

interFrms10_speed_pre_last = BGRFrames.from_frame(interFrms10_speed_pre[-1], trans_frames).dim_progressively(1, 0)

animFrms_trans, _ = vs.animate_sequence([curr_courtScn for _ in range(trans_frames)], start_cam = downCam, stop_cam = thirdCam,
                                        animate_over = interFrms10_speed_pre_last, 
                                        **prmDict)
# DecapStream(animFrms_trans, 0, 100).show()

#### available frames in order
# introFrms -> text contraction, fade in of play
# preFrms_last_pre, lineFrms_blink, playerFrms_blink -> lines, players flashing
# preFrms_last_post -> 10 frames right before serve impact
# animFrms_pre -> initial bit of play
# combFrms4_joint, combFrms3_joint, combFrms2_joint, combFrms1_joint --> shot type, speed, spin, launch angle.
# interFrms21_speed_fade -> fading out shot type markers
# interFrms21_speed_middle -> sped up intermezzo frames
# moveFrms -> moving camera to down view
# animFrms_down -> second to last point with overhead view
# animFrms_inter -> intermezzo before last point
# animFrms_trans -> transformation to pre-raster

# vis_ut.show(all_frames)

######### CONSTRUCT ALL FRAMES ############
all_frames = []
script_lines = []

script_line =   """
                    The Automated Tennis Match Annotator, or ATMAN for short,
                    is a start-to-end framework for the generation of insights
                    from tennis matches. 
                    ATMAN visually analyzes match footage, 
                    transforms it into shot-by-shot data, 
                    and allows the user to reveal patterns in the style of play of a player.  
                """ ### 18 sec

all_frames.append(introFrms)
script_lines.append(script_line)

script_line =   """
                    ATMAN automatically recognizes the lines ...
                """

all_frames.append(preFrms_last_pre0.slow(2))
script_lines.append(script_line)

script_line =   ""
all_frames.append(preFrms_last_pre1_feature.slow(2))
script_lines.append(script_line)

script_line =   """
                    ... the players ...
                """

all_frames.append(preFrms_last_pre2_feature.slow(2))
script_lines.append(script_line)

script_line = ""

all_frames.append(preFrms_last_pre3_feature.slow(2))
script_lines.append(script_line)

script_line =   """
                    ... and the ball.
                """

all_frames.append(preFrms_last_pre4_feature.slow(2))
script_lines.append(script_line)

script_line =   """
                    Articifial intelligence models analyze swing motion and the sound of the rally.
                    This data is checked against flight simulations to generate ball trajectories even on low quality video. 
                """ ### 11 sec

all_frames.append(animFrms_pre)
script_lines.append(script_line)

script_line  =  """
                    Every shot is automatically descibed...
                """ ## 3 secs

all_frames.append(combFrms4_joint)
script_lines.append(script_line)

script_line  =  """
                    ... and its speed, ...
                """ ## 2 secs

all_frames.append(combFrms3_joint)
script_lines.append(script_line)

script_line  =  """
                    ... spin, ...
                """ ## 1 secs

all_frames.append(combFrms2_joint)
script_lines.append(script_line)

script_line  =  """
                    ... and launch angle off the racket is calculated.
                """ ## 3 secs

all_frames.append(combFrms1_joint)
script_lines.append(script_line)

script_line =   "" ### have a litte pause

all_frames.append(interFrms21_speed_fade)
script_lines.append(script_line)

script_line =   """
                    ATMAN also includes a visual engine, which allows
                    any point to be displayed in arbitrary virtual space.
                """ ## 6 secs

# interFrms21_speed_middle_accel = vis_ut.speed_up_frames_progressively(interFrms21_speed_middle, ramp_up_frames = 40, ramp_down_frames = 40)
interFrms21_speed_middle_accel = interFrms21_speed_middle

jointFrms = BGRFrames.from_frames([interFrms21_speed_middle_accel, moveFrms])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    Points detected by ATMAN can be easily searched, filtered and aggregated.
                    The visual engine can then be seamlessly employed to display results.
                """ # 8 seconds

animFrms_inter_fs, animFrms_inter_ls = animFrms_inter[:160], animFrms_inter[160:] ### split, so we can fill this script line
jointFrms = BGRFrames.from_frames([animFrms_down, animFrms_inter_fs])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    For example, in the final of the 2021 Brest Challenger Tournament, 
                    Joao Sousa, a break behind and serving at deuce to stay in the match, 
                    is about to hit a first serve out wide to Brandon Nakashima.
                """ # 12 seconds
all_frames.append(animFrms_inter_ls)
script_lines.append(script_line)


script_line =   """
                    It might be advantageous for Joao Sousa to know if Brandon Nakashima has a pattern on these returns.
                """ # 6 seconds
all_frames.append(animFrms_trans)
script_lines.append(script_line)


vis_ut.show(all_frames)

### Saving the separate MultiFrame objects and the Joint Script
if SAVE:
    assert len(script_lines) == len(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        print(idx)
        subFrms, script_line = frames_and_script
        subFrms.save(fullPath, f'seg_{idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{idx}.txt'), mode = 'w') as f:
            f.write(script_line)

