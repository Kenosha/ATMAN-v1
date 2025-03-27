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

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'short_intro'
SAVE_FOLDER = 'atman_capabilities'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

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

######################################
### CHECK DISTRO OF BEFORE RETURNS ###
######################################

retShots = filtPnts_before.get_returns()
retShots_rect = retShots.enforce_direction('Top')

retShots_rect_out = retShots_rect.filter('is_fair', nextShot = None, negate = True)
retShots_rect_in = retShots_rect.filter('is_fair', nextShot = None)

heatScn_in = vs.kde_heatmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'],
                            bandwidth = 40, resolution = 2, normalize_density = True)

crtView = View.make(brestCam_full, Scene.make_court())
heatView = View.make(brestCam_full, heatScn_in)
# crtView.combine(heatView).snap().show()

markScn_in = vs.scatter_markmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'])
markScn_out = vs.scatter_markmap(retShots_rect_out, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'])

crtView = View.make(brestCam_full, Scene.make_court())
markView = View.make(brestCam_full, markScn_in)
# crtView.combine(markView).snap().show()

crtView = View.make(brestCam_full, Scene.make_court())
heatView = View.make(brestCam_full, markScn_out)
# crtView.combine(heatView).snap().show()

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

animFrms2 = vis_ut.animate_bounce_and_mark_scenes(prevPnt2, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, over = True,
                                                    sub_region = False, argument_resolution = 10)
# DecapStream(animFrms2, 0, 100).show()

animFrms1 = vis_ut.animate_bounce_and_mark_scenes(prevPnt1, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, over = True,
                                                    sub_region = False, argument_resolution = 10)
# DecapStream(animFrms1, 0, 100).show()

animFrms0 = vis_ut.animate_bounce_and_mark_scenes(lastPnt, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, over = True,
                                                    sub_region = False, argument_resolution = 10)
# DecapStream(animFrms0, 0, 100).show()

### figure out intermezzi.
initial_widen_by = 150
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
interFrms10_speed = vis_ut.speed_up_frames_progressively(interFrms10, ramp_up_frames = 25)

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
fade_in_length = 50

post_dimFrms_fade_out = BGRFrames.from_frame(post_dimFrms_lettered[-1], fade_in_length).dim_progressively(1, 0)
preFrms_fade_in = preFrms[:fade_in_length].dim_progressively(0, 1)

preFrms_fade = post_dimFrms_fade_out + preFrms_fade_in
# DecapStream(preFrms_fade, 0, 100).show()

preFrms_rest = preFrms[fade_in_length:]

introFrms = BGRFrames.from_frames([pre_dimFrms_lettered, post_dimFrms_lettered, preFrms_fade, preFrms_rest])
# DecapStream(introFrms, 0, 100).show()

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
frm_offset = 0 ### so the motion is complete.

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

stopFrms4 = animFrms2[frm4:frm3]
stopFrms3 = animFrms2[frm3:frm2]
stopFrms2 = animFrms2[frm2:frm1]
stopFrms1 = animFrms2[frm1:]

combFrms4 = overlay(stopFrms4.dim(dim_factor), fullFrm4, top_left_point = Point2D((1100, 40)))
combFrms3 = overlay(stopFrms3.dim(dim_factor), fullFrm3, top_left_point = Point2D((1200, 540)))
combFrms2 = overlay(stopFrms2.dim(dim_factor), fullFrm2, top_left_point = Point2D((1100, 40)))
combFrms1 = overlay(stopFrms1.dim(dim_factor), fullFrm1, top_left_point = Point2D((1300, 540)))

combFrms4_freeze = BGRFrames.from_frame(combFrms4[0], 25)
combFrms3_freeze = BGRFrames.from_frame(combFrms3[0], 25)
combFrms2_freeze = BGRFrames.from_frame(combFrms2[0], 25)
combFrms1_freeze = BGRFrames.from_frame(combFrms1[0], 25)

combFrms4_joint = BGRFrames.from_frames([combFrms4_freeze, combFrms4])
combFrms3_joint = BGRFrames.from_frames([combFrms3_freeze, combFrms3])
combFrms2_joint = BGRFrames.from_frames([combFrms2_freeze, combFrms2])
combFrms1_joint = BGRFrames.from_frames([combFrms1_freeze, combFrms1])

vis_ut.show([combFrms4_joint, combFrms3_joint, combFrms2_joint, combFrms1_joint])






DecapStream(animFrms_post, 0, 100).show()

# DecapStream(fullFrms, 0, 100).show()

fullFrms_fade = fullFrms[num_frames:num_frames+fade_in_frames]
fullFrms_post = fullFrms[num_frames+fade_in_frames:]

over_x, over_y = 1635, 650
tlPt = Point2D((over_x, over_y))
fadeFrms = overlay(animFrms_fade, fullFrms_fade, top_left_point = tlPt, start_alpha = 0, stop_alpha = 1)
postFrms = overlay(animFrms_post, fullFrms_post, top_left_point = tlPt)

### intermezzo frames
fullFrms_blank = make_point_descriptor_box(prevPnt3, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = back_color, text_size = 1,
                                     border_color = const.BGR['bright_grey'], border_thickness = 2)

interp_fullFrms = vs.interpolate_frames(fullFrms[-1], fullFrms_blank[-1], num_frames = 100)
rest_fullFrms = BGRFrames.from_frame(fullFrms_blank[-1], interFrms32_speed.length() - interp_fullFrms.length())
intermezzo_fullFrms = BGRFrames.from_frames([interp_fullFrms, rest_fullFrms])

interFrms_32_speed_dim = interFrms32_speed.dim(fade_factor)
interFrms_32_speed_dim = overlay(interFrms_32_speed_dim, intermezzo_fullFrms, top_left_point = tlPt)

jointFrms = BGRFrames.from_frames([fadeFrms, postFrms, interFrms_32_speed_dim])

all_frames.append(jointFrms)
script_lines.append(script_line)

#### available frames in order
# introFrms -> text contraction, fade in of play
# animFrms_pre -> initial bit of play

# vis_ut.show(all_frames)

##############################
##### SEC. RALLY TO SHOW #####
##############################

script_line =   """
                    ATMAN also includes a visual engine, which allows
                    any point to be displayed in arbitrary virtual space.
                """ ## 6 secs

back_color = (60, 60, 60)

c_x, c_y, c_z = -900, 900, 1500

brestCam = Camera.calibrated_brest_behind().resize(3)
downCam = Camera.from_to(Point3D((c_x, c_y, c_z)), Point3D((c_x, c_y, 0)))

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D
subReg = CourtLines.make().get_court_region().scale(1.25)
subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(back_color)

# testScn = Scene.make_court().put(subReg3D)
# View.make(stopCam, testScn, resolution = 1).snap(over_array = prevPnt2.stream(confs = confs).decap().frames[0].rescale(3).array).show()

prmDict = {'background_resolution': 0.1, 'argument_resolution': 0.1, 'interpolate_func': vs.interpolate_camera_naive}

animFrms = animate_bounce_and_mark_scenes(prevPnt2, start_cam = brestCam, stop_cam = downCam, background_scene = Scene.make_court(), 
                                          over = True, sub_region = True, region_color = back_color,
                                          init_over_dim = 0.5, last_over_dim = 0.5,
                                          start_later = 0, stop_earlier = 300,
                                          **prmDict)

fullFrms = make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                     border_thickness = 2, border_color = const.BGR['bright_grey'])

over_x, over_y = 1635, 650
tlPt = Point2D((over_x, over_y))
postFrms = overlay(animFrms, fullFrms, top_left_point = tlPt)
# DecapStream(postFrms, 0, 100).show()

### intermezzo frames
fullFrms_blank = make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = back_color, text_size = 1,
                                     border_color = const.BGR['bright_grey'], border_thickness = 2)

interp_fullFrms = vs.interpolate_frames(fullFrms[-1], fullFrms_blank[-1], num_frames = 100)
rest_fullFrms = BGRFrames.from_frame(fullFrms_blank[-1], interFrms21_speed.length() - interp_fullFrms.length())
intermezzo_fullFrms = BGRFrames.from_frames([interp_fullFrms, rest_fullFrms])

interFrms_21_speed_dim = interFrms21_speed.dim(fade_factor)
interFrms_21_speed_dim = overlay(interFrms_21_speed_dim, intermezzo_fullFrms, top_left_point = tlPt)

courtScn = Scene.make_court()
subReg = CourtLines.make().get_court_region().scale(1.25)
subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(back_color)
courtScn = courtScn.put(subReg3D)

interFrms_21_speed_dim_over, _ = vs.animate_sequence([Scene.make_empty() for _ in range(interFrms_21_speed_dim.length())], 
                                                        start_cam = downCam, stop_cam = downCam, 
                                                        background_scene = courtScn, animate_over = interFrms_21_speed_dim,
                                                        **prmDict)

jointFrms = BGRFrames.from_frames([postFrms, interFrms_21_speed_dim_over])

# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

# vis_ut.show(all_frames)

###############################
###### THIRD RALLY TO SHOW ####
###############################

script_line =   """
                    Points detected by ATMAN can be easily searched, filtered and aggregated.
                    The visual engine can then be seamlessly employed to display results.
                """
# -> while showing third (short) rally...
# ... transform top down into cooler 3D view, move into top right raster box.
# might need a little fade between the two views, 
# since we can approximately move them over each other, but not properly.
# alternative to fading in and out: just leave the full camera view, and have a hole in the raster.

# ####### CAM TESTING ########
# cntPt = CourtLines.make().get_center_point()
# cntPt3D = Point3D.from_point2D(cntPt, z = 0).left(750).down(600)
# camPt3D = Point3D((-300, 2000, 200))

# thirdCam = Camera.from_to(camPt3D, cntPt3D)

# testFrm = View.make(thirdCam, Scene.make_court(), resolution = 5).snap()
# testFrm.show()

# rs, cs = testFrm.rows, testFrm.cols
# testFrm[:rs//3, -cs//3:].show()

# cntPt = CourtLines.make().get_center_point()
# cntPt3D = Point3D.from_point2D(cntPt, z = 0).left(670).back(770)
# camPt3D = Point3D((-300, 2000, 600))

# thirdCam = Camera.from_to(camPt3D, cntPt3D).zoom(1.1)

# testFrm = View.make(thirdCam, Scene.make_court(), resolution = 5).snap()
# testFrm.show()
# ###############################

cntPt = CourtLines.make().get_center_point()
cntPt3D = Point3D.from_point2D(cntPt, z = 0).left(670).back(770)
camPt3D = Point3D((-300, 2000, 600))

thirdCam = Camera.from_to(camPt3D, cntPt3D).zoom(1.1)

animFrms = animate_bounce_and_mark_scenes(prevPnt1, start_cam = downCam, stop_cam = thirdCam, background_scene = Scene.make_court(), 
                                          over = True, sub_region = True, region_color = back_color,
                                          init_over_dim = 0.5, last_over_dim = 0.5,
                                          start_later = 0, stop_earlier = 50,
                                          **prmDict)

fullFrms = make_point_descriptor_box(prevPnt1, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                     border_thickness = 2, border_color = const.BGR['bright_grey'])

over_x, over_y = 1635, 650
tlPt = Point2D((over_x, over_y))
postFrms = overlay(animFrms, fullFrms, top_left_point = tlPt)
# DecapStream(postFrms, 0, 100).show()

all_frames.append(postFrms)
script_lines.append(script_line)

### go up to moment of serve (roughly), and say that he's about to hit a serve wide, on the deuce side.
script_line =   """
                    For example, in the final of the 2021 Brest Challenger Tournament, 
                    Joao Sousa, a break behind and serving at deuce to stay in the match, 
                    is about to hit a first serve out wide to Brandon Nakashima.
                """

### intermezzo frames
fullFrms_blank = make_point_descriptor_box(prevPnt1, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = back_color, text_size = 1,
                                     border_color = const.BGR['bright_grey'], border_thickness = 2)

intermezzo_stop_early = 50 ### stop early so that we have the windup of the serve later.
interFrms10_speed_pre, interFrms_10_speed_post = interFrms10_speed[:-intermezzo_stop_early], interFrms10_speed[-intermezzo_stop_early:]

interp_fullFrms = vs.interpolate_frames(fullFrms[-1], fullFrms_blank[-1], num_frames = 100)
rest_fullFrms = BGRFrames.from_frame(fullFrms_blank[-1], interFrms10_speed_pre.length() - interp_fullFrms.length())
intermezzo_fullFrms = BGRFrames.from_frames([interp_fullFrms, rest_fullFrms])

interFrms_10_speed_pre_dim = interFrms10_speed_pre.dim(fade_factor)
interFrms_10_speed_pre_dim = overlay(interFrms_10_speed_pre_dim, intermezzo_fullFrms, top_left_point = tlPt)

courtScn = Scene.make_court()
subReg = CourtLines.make().get_court_region().scale(1.25)
subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(back_color)
courtScn = courtScn.put(subReg3D)

interFrms_10_speed_dim_over, _ = vs.animate_sequence([Scene.make_empty() for _ in range(interFrms_10_speed_pre_dim.length())], 
                                                        start_cam = thirdCam, stop_cam = thirdCam, 
                                                        background_scene = courtScn, animate_over = interFrms_10_speed_pre_dim,
                                                        **prmDict)

# DecapStream(interFrms_10_speed_dim_over, 0, 100).show()

prmDict_no_interp = {k: v for k, v in prmDict.items() if not k in ['interpolate_func']}
onlycourtFrms, _ = vs.animate_static(courtScn, static_cam = thirdCam, background_scene = courtScn, **prmDict_no_interp, num_frames = 25)
onlycourtFrm = onlycourtFrms[0]

fadingFrms = vs.interpolate_frames(interFrms_10_speed_dim_over[-1], onlycourtFrm, num_frames = 50)
# DecapStream(fadingFrms, 0, 100).show()

pre_raster_frames = [interFrms_10_speed_dim_over, fadingFrms]

pre_rstFrms = BGRFrames.from_frames(pre_raster_frames)

all_frames.append(pre_rstFrms)
script_lines.append(script_line)


##################################
### SCRIPT_LINES #################
##################################

script_line =   """
                    The Automated Tennis Match Annotator, or ATMAN for short,
                    is a start-to-end framework for the generation of insights
                    from tennis matches. 
                    ATMAN visually recognizes tennis matches, 
                    transforms them into data shot-by-shot, 
                    and allows one to reveal patterns in the style of play of a player.  
                """
### TITLE CONTRACTION 

script_line =   """
                    ATMAN recognizes the lines, players and the ball
                    and employs a suite of articifial intelligence models
                    to analyze player swing motion and the sound of the rally
                    to generate ball trajectories even on low quality video. 
                """ ### 12 sec
### TRAJECTORIES AND LANDING SPOTS

script_line  =  """
                    Every shot is described, and its speed, spin and launch angle
                    off the racket is calculated.
                """ ## 4 secs
### SHOW DATA
# -> dim play, show box

script_line =   """
                    ATMAN also includes a visual engine, which allows
                    any point to be displayed in arbitrary virtual space.
                """ ## 6 secs
### TRANSFORM VISUAL INTO TOP DOWN
# -> keep play dim, bright View transforming into top down

script_line =   """
                    Points detected by ATMAN can be easily searched, filtered and aggregated.
                    The visual engine can then be seamlessly employed to display results.
                """
# -> transform top down into cooler 3D view, move into top right raster box.
# might need a little fade between the two views, 
# since we can approximately move them over each other, but not properly.
# alternative to fading in and out: just leave the full camera view, and have a hole in the raster.

script_line =   """
                    For example, in the final of the 2021 Brest Challenger Tournament, 
                    Joao Sousa is about to hit a first serve from the ad side out wide to Brandon Nakashima.
                    With ATMAN, one can filter for occurences in previous matches of the same tournament
                    in which Brandon Nakashima faced a similar serve, and take a look at where his return
                    previously landed.
                """
# raster with previous examples, trajectories and bounce points shown in the cool view top right.
# maybe bounce points are made into heatmap directly (or transform in a cool animation, like shockwave)

script_line =   """
                    [Comment on where the ball lands vs. "expectation"]
                """
# fade the 3D view in raster back to 3D view with full camera
# transform the cool 3D view in full camera back to the court, seamlessly superimposing the heatmap
# check where the ball landed

script_line =   """
                    [Closing statement]
                """
# resume rally, fade out with the rest of the rally playing.

######### CONSTRUCT ALL FRAMES ############

all_frames = []
script_lines = []

script_line =   """
                    The Automated Tennis Match Annotator, or ATMAN for short,
                    is a start-to-end framework for the generation of insights
                    from tennis matches. 
                    ATMAN visually recognizes tennis matches, 
                    transforms them into data shot-by-shot, 
                    and allows one to reveal patterns in the style of play of a player.  
                """

script_line =   """
                    ATMAN recognizes the lines, players and the ball
                    and employs a suite of articifial intelligence models
                    to analyze player swing motion and the sound of the rally
                    to generate ball trajectories even on low quality video. 
                """ ### 12 sec

script_line  =  """
                    Every shot is described, and its speed, spin and launch angle
                    off the racket is calculated.
                """ ## 4 secs