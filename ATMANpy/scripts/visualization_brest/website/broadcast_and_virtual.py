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

DESCRIPTOR = 'broadcast_and_virtual'
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

####################
#### 3D Engine #####
####################

from ATMANpy.types.ShapesTennis3D import Net
net_color = const.BGR['smokey_grey']
net_alpha = 0.75
net_mesh_density = 3

line_color = const.BGR['bright_grey']

###### -> idea: show the lines rotating in space with a small, arbitrary 2D representation with axes!
###### -> also: show the actual projection matrix on the screen!
from ATMANpy.types.Shapes3D import Point3D, Line3D, Lines3D
from ATMANpy.auxil.court import cnt_2D, courtLines
cnt_3D = Point3D.from_point2D(cnt_2D, z = 0)

# court
courtScn = Scene.make_court(make_net = True, singles_net = False, line_color = line_color, 
                            net_color = net_color, net_alpha = net_alpha, net_mesh_density = net_mesh_density)
courtLines3D = Lines3D.from_lines2D(courtLines, z = 0, name = 'CourtLines3D')

# corner points
c1 = courtLines3D['Top Baseline'].leftPt
c2 = courtLines3D['Top Baseline'].rightPt
c3 = courtLines3D['Bottom Baseline'].rightPt
c4 = courtLines3D['Bottom Baseline'].leftPt
l, w = 2 * const.SIDE_LENGTH * const.SCALE, const.COURT_WIDTH * const.SCALE

# camera checkpoints to interpolate between
z = 100
f = 1200
tilt = 0.0

oPt1 = c1.left(w/2).front(l/2).up(z)
cam1 = Camera.from_to(oPt1, cnt_3D, f = f).tilt(tilt).resize(1/3)
# View.make(cam1, courtaxScn).snap().show()

oPt2 = c2.right(w/2).front(l/2).up(z)
cam2 = Camera.from_to(oPt2, cnt_3D, f = f).tilt(tilt).resize(1/3)
# View.make(cam2, courtaxScn).snap().show()

oPt3 = c3.right(w/2).back(l/2).up(z)
cam3 = Camera.from_to(oPt3, cnt_3D, f = f).tilt(tilt).resize(1/3)
# View.make(cam3, courtaxScn).snap().show()

oPt4 = c4.left(w/2).back(l/2).up(z)
cam4 = Camera.from_to(oPt4, cnt_3D, f = f).tilt(tilt).resize(1/3)
# View.make(cam4, courtaxScn).snap().show()

oPt2a = cnt_3D.right(4*w/2).up(z)
cam2a = Camera.from_to(oPt2a, cnt_3D, f = f).tilt(tilt).resize(1/3)
# View.make(cam2a, courtaxScn).snap().show()

oPt4a = cnt_3D.left(4*w/2).up(z)
cam4a = Camera.from_to(oPt4a, cnt_3D, f = f).tilt(tilt).resize(1/3)
# View.make(cam4a, courtaxScn).snap().show()

### show camera points in 3D space
from ATMANpy.types.ShapesTennis3D import Mark
from ATMANpy.types.Shapes3D import Ellipse3D

tstCam = Camera.from_to(cnt_3D.up(2000), cnt_3D)
tstScn = Scene([Mark.make(oPt1, size = 30), Mark.make(oPt2, size = 30), Mark.make(oPt2a, size = 30), 
                Mark.make(oPt3, size = 30), Mark.make(oPt4, size = 30), Mark.make(oPt4a, size = 30)])

camEll = Ellipse3D(cnt_3D.coords, radius_short = 2 * w, radius_long = 1.1 * l, rot_z = 3 * np.pi / 2).up(z)

tstScn = tstScn.combine(courtScn).put(camEll)

# View.make(tstCam, tstScn).snap().show()

### create camera points from ellipse
originArr = camEll.rasterize(0.0075)
originPts = [Point3D(tuple(coords)) for coords in originArr]
cams = [Camera.from_to(origPt, cnt_3D).resize(1/3).zoom(1.2) for origPt in originPts]

res = 5
prmDict = {'foreground_resolution': 0.35 * res, 'background_resolution': res, 'argument_resolution': 0.4 * res,
           'cumulative_foreground_transparence': 0.8, 'cumulate_argument': True,
           'background_scene': courtScn, 'interpolate_func': vs.interpolate_camera_naive}

res = 13
prmDict = {'foreground_resolution': res, 'background_resolution': res, 'argument_resolution': res,
           'cumulative_foreground_transparence': 0.85, 'cumulate_argument': True,
           'background_scene': courtScn, 'interpolate_func': vs.interpolate_camera_naive}

prmDict_no_cum = {**prmDict}
prmDict_no_cum['cumulate_argument'] = False
prmDict_no_cum['cumulative_foreground_transparence'] = 1

prmDict_static = {key: value for key, value in prmDict.items() if key not in [  'cumulate_argument',
                                                                                'interpolate_func',
                                                                                'cumulative_foreground_transparence']}

from itertools import cycle
cam_cycle = cycle(cams)
curr_cam = next(cam_cycle)
num_cams = len(cams)

# get the point of interest
mtc_idx, pnt_idx = 0, 111 ### the first point of atman short intro video
prevPnt2 = allPnts.get(mtc_idx, pnt_idx)

mark_color = const.BGR['bright_blue']
mark_size = 6
ball_color = const.BGR['ball_yellow']
net_ball_color = const.BGR['intense_red']

shtScns, bncScns = vis_ut.get_shot_scenes(  prevPnt2[:-5], 
                                            mark_size = mark_size, mark_color = mark_color,
                                            ball_color = ball_color, net_color = net_ball_color)
num_shot_frames = len(shtScns)

num_rotations = 2
increase_rot_frames_by = 2

rot_frames = num_shot_frames // (num_rotations * num_cams) + increase_rot_frames_by

animDict = {}

overFrm = BGRFrame.make_black(640, 360) + const.BGR['darker_grey']

### previous rotations, without ball
frames_before = 25
frames_per = 5

cams_before = frames_before // frames_per

curr_cam = cams[-cams_before]
next_cams = [*cams[-cams_before+1:], cams[0]]
pre_frames = []
for next_cam in next_cams:
    animFrms, animDict = vs.animate_sequence([Scene.make_empty() for _ in range(frames_per)], 
                                            start_cam = curr_cam, stop_cam = next_cam, 
                                            **animDict, **prmDict, 
                                            animate_over = overFrm)
    pre_frames.append(animFrms)
    curr_cam = next_cam

preFrms = BGRFrames.from_frames(pre_frames)
# DecapStream(preFrms, 0, 100).show()

all_frames = []

fs_frame = 0
while fs_frame < num_shot_frames:
    ls_frame = min(fs_frame + rot_frames, num_shot_frames)

    next_cam = next(cam_cycle)

    animFrms, animDict = vs.animate_sequence(shtScns[fs_frame:ls_frame], 
                                            start_cam = curr_cam, stop_cam = next_cam, 
                                            **animDict, **prmDict,
                                            animate_over = overFrm)
    all_frames.append(animFrms)

    curr_cam = next_cam
    fs_frame = ls_frame

rotFrms = BGRFrames.from_frames(all_frames)
# DecapStream(rotFrms, 0, 100).show()


### after the rotations, we take one shots time to move into position for the axis

moveShot = prevPnt2[-5]
stopShot4 = prevPnt2[-4]
stopShot3 = prevPnt2[-3]
stopShot2 = prevPnt2[-2]
stopShot1 = prevPnt2[-1]

shtScns_move, bncScns_move = vis_ut.get_shot_scenes(  [moveShot], 
                                            mark_size = mark_size, mark_color = mark_color,
                                            ball_color = ball_color, net_color = net_ball_color)
shtScns4, bncScns_move = vis_ut.get_shot_scenes(  [stopShot4], 
                                            mark_size = mark_size, mark_color = mark_color,
                                            ball_color = ball_color, net_color = net_ball_color)
shtScns3, bncScns_move = vis_ut.get_shot_scenes(  [stopShot3], 
                                            mark_size = mark_size, mark_color = mark_color,
                                            ball_color = ball_color, net_color = net_ball_color)
shtScns2, bncScns_move = vis_ut.get_shot_scenes(  [stopShot2], 
                                            mark_size = mark_size, mark_color = mark_color,
                                            ball_color = ball_color, net_color = net_ball_color)
shtScns1, bncScns_move = vis_ut.get_shot_scenes(  [stopShot1], 
                                            mark_size = mark_size, mark_color = mark_color,
                                            ball_color = ball_color, net_color = net_ball_color)

frm_offset = 4
frm_grow = 15
frm_hold = 75

frm_hold_net = frm_hold - frm_grow - frm_offset

shtScns4_pre, shtScns4_post = shtScns4[:frm_offset], shtScns4[frm_offset:]
shtScns3_pre, shtScns3_post = shtScns3[:frm_offset], shtScns3[frm_offset:]
shtScns2_pre, shtScns2_post = shtScns2[:frm_offset], shtScns2[frm_offset:]
shtScns1_pre, shtScns1_post = shtScns1[:frm_offset], shtScns1[frm_offset:]

shtScns4_pre_hold = [*shtScns4_pre, *[shtScns4_pre[-1] for _ in range(frm_grow - frm_offset)]]
shtScns3_pre_hold = [*shtScns3_pre, *[shtScns3_pre[-1] for _ in range(frm_grow - frm_offset)]]
shtScns2_pre_hold = [*shtScns2_pre, *[shtScns2_pre[-1] for _ in range(frm_grow - frm_offset)]]
shtScns1_pre_hold = [*shtScns1_pre, *[shtScns1_pre[-1] for _ in range(frm_grow - frm_offset)]]

someShot = stopShot4

def get_cam(someShot, x_off = 20, y_off = 20, z_off = 0, shift_target_point = False,
            zoom_factor = 1):
    hitPt = Point3D(someShot.get_ball_hit_position())
    side = someShot.get_hit_side()
    if side == 'Top':
        offset = y_off
    elif side == 'Bot':
        offset = -y_off

    fromPt = hitPt.right(x_off).front(offset)
    if shift_target_point:
        toPt = hitPt.front(offset).up(z_off)
    else:
        toPt = hitPt.up(z_off)
    return Camera.from_to(fromPt, toPt).resize(1/3).zoom(zoom_factor)

def get_axes(someShot):
    ax_len = 15
    hat_length = 2
    ax_color = const.BGR['bright_grey']
    ax_thickness = 0
    
    hitPt = Point3D(someShot.get_ball_hit_position())

    x_dir, y_dir = someShot.get_flight_line(force_l2r = False).to_vector().normalize().coords

    xVec = Point3D((x_dir, y_dir, 0)) * ax_len
    zVec = Point3D((0, 0, 1)) * ax_len

    xArr = Arrow.make(hitPt, hitPt + xVec, hat_length = hat_length, rot_angle = 0, 
                        color = ax_color, thickness = ax_thickness)
    zArr = Arrow.make(hitPt, hitPt + zVec, hat_length = hat_length, rot_angle = np.pi / 2, 
                        color = ax_color, thickness = ax_thickness)

    xArr_null = Arrow.make(hitPt, hitPt + xVec / 1000, hat_length = hat_length / 100, rot_angle = 0, 
                        color = ax_color, thickness = ax_thickness)

    zArr_null = Arrow.make(hitPt, hitPt + zVec / 1000, hat_length = hat_length / 100, rot_angle = np.pi / 2, 
                        color = ax_color, thickness = ax_thickness)

    axScn = Scene([xArr, zArr])
    nullScn = Scene([xArr_null, zArr_null])

    return axScn, nullScn

def get_angle(someShot):
    radius = 50
    color = const.BGR['bright_blue']
    thickness = 0.05
    extra_angle = 1.5
    dimin_angle = 3

    hitPt = Point3D(someShot.get_ball_hit_position())
    ang = someShot.get_launch_angle()
    hit_from = someShot.get_hit_side()

    net_ang = ang - dimin_angle

    if hit_from == 'Bot':
        limit_theta = ((180 - net_ang) * const.ANGLE_TO_RAD, (180 + extra_angle) * const.ANGLE_TO_RAD)
        null_limit_theta = ((180 + extra_angle - 0.001) * const.ANGLE_TO_RAD, (180 + extra_angle) * const.ANGLE_TO_RAD)
        start_theta = 0
    elif hit_from == 'Top':
        limit_theta = (0 * const.ANGLE_TO_RAD, (net_ang + extra_angle) * const.ANGLE_TO_RAD)
        null_limit_theta = (0 * const.ANGLE_TO_RAD, 0.001 * const.ANGLE_TO_RAD)
        start_theta = - extra_angle * const.ANGLE_TO_RAD

    degCrc = Circle3D.from_point3D(hitPt, fixed_dimension = 'x', radius = radius, 
                                   limit_theta = limit_theta, start_theta = start_theta).recolor(color)
    nullCrc = Circle3D.from_point3D(hitPt, fixed_dimension = 'x', radius = radius, 
                                    limit_theta = null_limit_theta, start_theta = start_theta).recolor(color)

    return degCrc, nullCrc

def get_speed(someShot, rot_angle = np.pi / 2):
    hat_length = 4
    arr_length = 50
    arr_color = const.BGR['bright_blue']
    arr_thickness = 0.1

    hitPt = Point3D(someShot.get_ball_hit_position())
    
    flightLn = someShot.get_flight_line(force_l2r = False)
    hori_speed = someShot.get_initial_horizontal_speed()
    vert_speed = someShot.get_initial_vertical_speed()
    init_speed = someShot.get_initial_speed()

    x_norm, y_norm = flightLn.to_vector().normalize().coords
    hori_norm, vert_norm = hori_speed / init_speed, vert_speed / init_speed

    flight_norm_vec = Point3D((x_norm * hori_norm, y_norm * hori_norm, vert_norm))
    
    startPt = hitPt.up(10)
    endPt = startPt + flight_norm_vec * arr_length
    endPt_null = startPt + flight_norm_vec * 0.01

    flyArr = Arrow.make(startPt, endPt, hat_length = hat_length, rot_angle = rot_angle,
                        color = arr_color, thickness = arr_thickness)

    nullArr = Arrow.make(startPt, endPt_null, hat_length = hat_length * 0.01, rot_angle = rot_angle,
                         color = arr_color, thickness = arr_thickness)

    return flyArr, nullArr

def get_spin(someShot):
    radius = 5
    color = const.BGR['bright_blue']
    thickness = 0.05
    space_ang = 40

    hitPt = Point3D(someShot.get_ball_hit_position())
    ang = someShot.get_launch_angle()
    hit_from = someShot.get_hit_side()

    ang_and_space = ang + space_ang
    full_and_space = 360 - space_ang

    if hit_from == 'Bot':
        limit_theta = ((ang_and_space + 25) * const.ANGLE_TO_RAD, full_and_space * const.ANGLE_TO_RAD)
        null_limit_theta = ((ang_and_space + 25) * const.ANGLE_TO_RAD, (ang_and_space + 30) * const.ANGLE_TO_RAD)
        start_theta = np.pi
    elif hit_from == 'Top':
        limit_theta = ((ang_and_space + 25) * const.ANGLE_TO_RAD, full_and_space * const.ANGLE_TO_RAD)
        null_limit_theta = ((full_and_space - 5) * const.ANGLE_TO_RAD, full_and_space * const.ANGLE_TO_RAD)
        start_theta = 0

    degCrc = Circle3D.from_point3D(hitPt, fixed_dimension = 'x', radius = radius, 
                                   limit_theta = limit_theta, start_theta = start_theta).recolor(color)
    nullCrc = Circle3D.from_point3D(hitPt, fixed_dimension = 'x', radius = radius, 
                                    limit_theta = null_limit_theta, start_theta = start_theta).recolor(color)

    return degCrc, nullCrc

def interpolate_initial_condition_scenes(someShot, itpl_frames = 25, rot_angle = np.pi / 2):     
    crc_arr_share = 0.1

    hit_side = someShot.get_hit_side()

    degCrc, degCrc_null = get_angle(someShot)
    velArr, velArr_null = get_speed(someShot, rot_angle = rot_angle)
    rpmCrc, rpmCrc_null = get_spin(someShot)

    fullScn = Scene([degCrc, velArr, rpmCrc])
    nullScn = Scene([degCrc_null, velArr_null, rpmCrc_null])

    itplScns = vs.interpolate_scenes(nullScn, fullScn, itpl_frames)

    ### add in the arrows for the spin circle
    rpmScns = vs.interpolate_scenes(Scene([rpmCrc_null]), Scene([rpmCrc]), itpl_frames)

    spin_arrows = []
    for idx, scn in enumerate(rpmScns):
        currCrc = scn.shapes[0]

        hat_length = 0.5 * velArr[2].length() * (idx / (len(rpmScns) - 1)) ### scale arrow hat length
        
        crcArr = currCrc.rasterize(10)
        critical_index = int(len(crcArr) * crc_arr_share)
        
        if critical_index == 0:
            spin_arrows.append(None)
        else:
            if hit_side == 'Top':
                lsPt0 = Point3D((tuple(crcArr[critical_index])))
                lsPt1 = Point3D((tuple(crcArr[0])))
            elif hit_side == 'Bot':
                lsPt0 = Point3D((tuple(crcArr[-critical_index-1])))
                lsPt1 = Point3D((tuple(crcArr[-1])))

            rotArr = Arrow.make(lsPt0, lsPt1, hat_length = hat_length, thickness = 0, 
                                hat_angle = np.pi / 6, rot_angle = np.pi / 2, 
                                color = currCrc.color)
            spin_arrows.append(rotArr)

    ### add in the arrows for the degree circle
    degScns = vs.interpolate_scenes(Scene([degCrc_null]), Scene([degCrc]), itpl_frames)

    ang_arrows = []
    for idx, scn in enumerate(degScns):
        currCrc = scn.shapes[0]

        hat_length = 0.5 * velArr[2].length() * (idx / (len(degScns) - 1)) ### scale arrow hat length
        
        crcArr = currCrc.rasterize(10)
        critical_index = int(len(crcArr) * crc_arr_share)
        
        if critical_index == 0:
            ang_arrows.append(None)
        else:
            if hit_side == 'Top':
                lsPt0 = Point3D((tuple(crcArr[-critical_index-1])))
                lsPt1 = Point3D((tuple(crcArr[-1])))
            elif hit_side == 'Bot':
                lsPt0 = Point3D((tuple(crcArr[critical_index])))
                lsPt1 = Point3D((tuple(crcArr[0])))

            rotArr = Arrow.make(lsPt0, lsPt1, hat_length = hat_length, thickness = 0, 
                                hat_angle = np.pi / 6, rot_angle = np.pi / 2, 
                                color = currCrc.color)
            ang_arrows.append(rotArr)

    jointScns = [itplScn.put(crcArr).put(degArr) for itplScn, crcArr, degArr in zip(itplScns, spin_arrows, ang_arrows)]
    return jointScns

def make_annotated_points(someShot, 
                          desc_offset_up = 30, desc_offset_back = 40,
                          vel_offset_up = 15, vel_offset_back = 30,
                          spin_offset_up = 0, spin_offset_back = -10,
                          ang_offset_up = 0, ang_offset_back = 48):
    color = const.BGR['bright_blue']

    hitPt = Point3D(someShot.get_ball_hit_position())

    desc = someShot.get_shot_description()
    rpm = someShot.get_initial_spin()
    vel = someShot.get_initial_speed()
    ang = someShot.get_launch_angle()

    descStr = desc
    rpmStr = f'{rpm:.0f} rpm'
    velStr = f'{vel:.0f} km/h'
    angStr = f'{ang:.0f} deg.'

    descPt = hitPt.up(desc_offset_up).back(desc_offset_back).annotate(descStr).recolor(color).rethicken(2)
    velPt = hitPt.up(vel_offset_up).back(vel_offset_back).annotate(velStr).recolor(color)
    rpmPt = hitPt.up(spin_offset_up).back(spin_offset_back).annotate(rpmStr).recolor(color)
    angPt = hitPt.up(ang_offset_up).back(ang_offset_back).annotate(angStr).recolor(color)

    annPnts = [descPt, velPt, rpmPt, angPt]
    return annPnts

# cam_kwargs = {'x_off': 50, 'y_off': -25, 'z_off': 10, 'shift_target_point': True, 'zoom_factor': 1}
cam_kwargs = {'x_off': 100, 'y_off': -25, 'z_off': 10, 'shift_target_point': True, 'zoom_factor': 1.5}
cam_kwargs_last = {**cam_kwargs}
cam_kwargs_last['z_off'] = -5

axCam4 = get_cam(stopShot4, **cam_kwargs)
axCam3 = get_cam(stopShot3, **cam_kwargs)
axCam2 = get_cam(stopShot2, **cam_kwargs)
axCam1 = get_cam(stopShot1, **cam_kwargs_last)

tst_combScn = shtScns4_pre[1].combine(shtScns4_pre[2]).combine(shtScns4_pre[3])
tstScn = tst_combScn.combine(courtScn)
tstView = View.make(axCam4, tstScn)
# tstView.snap().show()

moveFrms4, animDict4 = vs.animate_sequence(shtScns_move, start_cam = curr_cam, stop_cam = axCam4,
                                    **animDict, **prmDict,
                                    animate_over = overFrm)

holdFrms4, animDict4a = vs.animate_sequence(shtScns4_pre, start_cam = axCam4, stop_cam = axCam4,
                                    **animDict, **prmDict,
                                    animate_over = overFrm)

itplScns = interpolate_initial_condition_scenes(stopShot4, itpl_frames = frm_grow)

tstScn = tst_combScn.combine(courtScn).combine(itplScns[-1])
tstView = View.make(axCam4, tstScn)
# tstView.snap().show()

growFrms4, animDict4b = vs.animate_sequence(itplScns, start_cam = axCam4, stop_cam = axCam4,
                                            **animDict4a, **prmDict_no_cum,
                                            animate_over = overFrm)
# DecapStream(growFrms4, 0, 100).show()

### add held frames with annotated points

annPnts = make_annotated_points(stopShot4, ang_offset_up = -1)

stopFrms4, animDict4c = vs.animate_static(itplScns[-1], static_cam = axCam4,
                                            num_frames = frm_hold_net,
                                            annotated_points = annPnts,
                                            **animDict4b, **prmDict_static,
                                            animate_over = overFrm)

### move on to next shot
moveFrms3, animDict3 = vs.animate_sequence(shtScns4, start_cam = axCam4, stop_cam = axCam3,
                                           **animDict4c, **prmDict,
                                           stop_earlier = 3,
                                            animate_over = overFrm)


jointFrms4 = BGRFrames.from_frames([holdFrms4, growFrms4, stopFrms4, moveFrms3])
# DecapStream(jointFrms4, 0, 100).show()

holdFrms3, animDict3a = vs.animate_sequence(shtScns3_pre, start_cam = axCam3, stop_cam = axCam3,
                                            **prmDict,
                                            animate_over = overFrm)

# DecapStream(holdFrms3, 0, 100).show()


itplScns = interpolate_initial_condition_scenes(stopShot3, itpl_frames = frm_grow)

tstScn = tst_combScn.combine(courtScn).combine(itplScns[-1])
tstView = View.make(axCam3, tstScn)
# tstView.snap().show()

growFrms3, animDict3b = vs.animate_sequence(itplScns, start_cam = axCam3, stop_cam = axCam3,
                                            **animDict3a, **prmDict_no_cum,
                                            animate_over = overFrm)
# DecapStream(growFrms3, 0, 100).show()

annPnts = make_annotated_points(stopShot3, 
                                desc_offset_back = -10, 
                                vel_offset_back = -15, vel_offset_up = 18,
                                spin_offset_back = 23, spin_offset_up = -1,
                                ang_offset_back = -37, ang_offset_up = 1)

stopFrms3, animDict3c = vs.animate_static(itplScns[-1], static_cam = axCam3,
                                            num_frames = frm_hold_net,
                                            annotated_points = annPnts,
                                            **animDict3b, **prmDict_static,
                                            animate_over = overFrm)

### and the next
moveFrms2, _ = vs.animate_sequence(shtScns3, start_cam = axCam3, stop_cam = axCam2,
                                   **animDict3c, **prmDict,
                                   animate_over = overFrm)

jointFrms3 = BGRFrames.from_frames([holdFrms3, growFrms3, stopFrms3, moveFrms2])
# DecapStream(jointFrms3, 0, 100).show()

holdFrms2, animDict2a = vs.animate_sequence(shtScns2_pre, start_cam = axCam2, stop_cam = axCam2,
                                            **prmDict,
                                            animate_over = overFrm)

itplScns = interpolate_initial_condition_scenes(stopShot2, itpl_frames = frm_grow)

growFrms2, animDict2b = vs.animate_sequence(itplScns, start_cam = axCam2, stop_cam = axCam2,
                                            **animDict2a, **prmDict_no_cum,
                                            animate_over = overFrm)
# DecapStream(growFrms2, 0, 100).show()

### add held frames with annotated points

annPnts = make_annotated_points(stopShot2, 
                                vel_offset_up = 20,
                                ang_offset_up = 3,
                                spin_offset_back = -7,
                                spin_offset_up = -1)

stopFrms2, animDict2c = vs.animate_static(itplScns[-1], static_cam = axCam2,
                                            num_frames = frm_hold_net,
                                            annotated_points = annPnts,
                                            **animDict2b, **prmDict_static,
                                            animate_over = overFrm)
# stopFrms2[-1].show()


### and the last
moveFrms1, _ = vs.animate_sequence(shtScns2, start_cam = axCam2, stop_cam = axCam1,
                                           **animDict2c, **prmDict,
                                           stop_earlier = 3,
                                            animate_over = overFrm)

jointFrms2 = BGRFrames.from_frames([holdFrms2, growFrms2, stopFrms2, moveFrms1])
# DecapStream(jointFrms2, 0, 100).show()

holdFrms1, animDict1a = vs.animate_sequence(shtScns1_pre, start_cam = axCam1, stop_cam = axCam1,
                                            **prmDict,
                                            animate_over = overFrm)

# DecapStream(holdFrms1, 0, 100).show()

itplScns = interpolate_initial_condition_scenes(stopShot1, itpl_frames = frm_grow, rot_angle = np.pi)

growFrms1, animDict1b = vs.animate_sequence(itplScns, start_cam = axCam1, stop_cam = axCam1,
                                            **animDict1a, **prmDict_no_cum,
                                            animate_over = overFrm)
# DecapStream(growFrms1, 0, 100).show()

annPnts = make_annotated_points(stopShot1, 
                                desc_offset_back = -10, desc_offset_up = 22,
                                vel_offset_back = -15, vel_offset_up = 12,
                                spin_offset_back = 23, spin_offset_up = -1,
                                ang_offset_back = -37, ang_offset_up = -5)

stopFrms1, animDict1c = vs.animate_static(itplScns[-1], static_cam = axCam1,
                                            num_frames = frm_hold_net,
                                            annotated_points = annPnts,
                                            **animDict1b, **prmDict_static,
                                            animate_over = overFrm)
# stopFrms1[-1].show()

axCam0 = axCam1.right(45)

prmDict_final = {**prmDict}
prmDict_final['cumulative_foreground_transparence'] = 1

moveFrms0, _ = vs.animate_sequence(shtScns1, start_cam = axCam1, stop_cam = axCam0,
                                           **animDict1c, **prmDict_final,
                                           stop_earlier = 20,
                                           animate_over = overFrm)
# DecapStream(moveFrms0, 0, 100).show()

jointFrms1 = BGRFrames.from_frames([holdFrms1, growFrms1, stopFrms1, moveFrms0])
# DecapStream(jointFrms1, 0, 100).show()


startstopFrms = BGRFrames.from_frames([jointFrms4, jointFrms3, jointFrms2, jointFrms1])
# DecapStream(startstopFrms, 0, 100).show()


#### after -> 50 frames to move back to starting position

postFrms, _ = vs.animate_sequence([Scene.make_empty() for _ in range(50)],
                                  start_cam = axCam0, stop_cam = cams[-cams_before],
                                  **prmDict, animate_over = overFrm)
# DecapStream(postFrms, 0, 100).show()

virtualFrms = BGRFrames.from_frames([preFrms, rotFrms, moveFrms4, startstopFrms, postFrms])
# DecapStream(virtualFrms, 0, 100).show()

virtualFrms = virtualFrms.rescale(2).gaussian_noise(std = 10).gaussian_blur((5, 5)).rescale(0.5)
# DecapStream(virtualFrms, 0, 100).show()

virtualFrms.save(fullPath, f'virtual', lossless = True, save_as = 'mp4')

############################
#### THE MAIN ANIMATION ####
############################

# on the landing page: an animation similar to the beginning of atman_short
# -> ball recognized, shots categorized.


res = 13
# prmDict = {'background_resolution': 2, 'argument_resolution': 10, 'foreground_resolution': 10,
#            'interpolate_func': vs.interpolate_camera_naive} ### TAKES VERY LONG! TRY LOWER OR DO SOMETHING ELSE IN THE MEANTIME
prmDict = {'background_resolution': res, 'argument_resolution': res, 'foreground_resolution': res,
           'interpolate_func': vs.interpolate_camera_naive} ### TAKES VERY LONG! TRY LOWER OR DO SOMETHING ELSE IN THE MEANTIME

brestCam_full = Camera.calibrated_brest_behind().resize(1.5)
animFrms2 = vis_ut.animate_bounce_and_mark_scenes(  prevPnt2, start_cam = brestCam_full, stop_cam = brestCam_full, confs = confs, 
                                                    over = True, over_scale = 1.5,
                                                    sub_region = False,  mark_size = mark_size, mark_color = mark_color,
                                                    ball_color = ball_color, net_color = net_ball_color,
                                                    spline_cumulative_transparence = 0.75,
                                                    **prmDict)
# DecapStream(animFrms2, 0, 100).show()

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
                                                highlight_which = 0, highlight_color = const.BGR['ball_yellow'],
                                                back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                                border_color = const.BGR['bright_grey'], border_thickness = 2)

fullFrms_hi1 = vis_ut.make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), 
                                                para_px = para_px, skip_px = skip_px,
                                                highlight_which = 0, highlight_color = const.BGR['ball_yellow'],
                                                back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                                border_color = const.BGR['bright_grey'], border_thickness = 2)

fullFrms_hi2 = vis_ut.make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), 
                                                para_px = para_px, skip_px = skip_px,
                                                highlight_which = 0, highlight_color = const.BGR['ball_yellow'],
                                                back_color = back_color, text_color = (255, 255, 255), text_size = 1,
                                                border_color = const.BGR['bright_grey'], border_thickness = 2)

fullFrms_hi3 = vis_ut.make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), 
                                                para_px = para_px, skip_px = skip_px,
                                                highlight_which = 0, highlight_color = const.BGR['intense_red'],
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

combFrms4 = overlay(stopFrms4.dim(dim_factor), fullFrm4, top_left_point = Point2D((600, 20)))
combFrms3 = overlay(stopFrms3.dim(dim_factor), fullFrm3, top_left_point = Point2D((650, 270)))
combFrms2 = overlay(stopFrms2.dim(dim_factor), fullFrm2, top_left_point = Point2D((600, 20)))
combFrms1 = overlay(stopFrms1.dim(dim_factor), fullFrm1, top_left_point = Point2D((675, 270)))

combFrms4_freeze = BGRFrames.from_frame(combFrms4[0], 75)
combFrms3_freeze = BGRFrames.from_frame(combFrms3[0], 75)
combFrms2_freeze = BGRFrames.from_frame(combFrms2[0], 75)
combFrms1_freeze = BGRFrames.from_frame(combFrms1[0], 75)

combFrms4_joint = BGRFrames.from_frames([combFrms4_freeze, combFrms4])
combFrms3_joint = BGRFrames.from_frames([combFrms3_freeze, combFrms3])
combFrms2_joint = BGRFrames.from_frames([combFrms2_freeze, combFrms2])
combFrms1_joint = BGRFrames.from_frames([combFrms1_freeze, combFrms1])

# vis_ut.show([combFrms4_joint, combFrms3_joint, combFrms2_joint, combFrms1_joint])

transition_frames = 25

wideFrms = prevPnt2.stream(confs = confs).widen(transition_frames).decap().frames.rescale(1.5)

fsFrms = wideFrms[:transition_frames]
lsFrms = wideFrms[-transition_frames:]


_, markScns = vis_ut.get_shot_scenes(prevPnt2, mark_size = mark_size, mark_color = mark_color)
markScn = markScns[-1]

alpha_factors = np.linspace(1, 0, num = lsFrms.length())

markScns_fade = []
for alpha in alpha_factors:
    markScns_fade.append(markScn.realpha(alpha))

lsFrms_with_marks, _ = vs.animate_sequence(markScns_fade, start_cam = brestCam_full, stop_cam = brestCam_full,
                                            animate_over = lsFrms, 
                                            **prmDict)

lsFrms_with_marks = lsFrms_with_marks.dim_progressively(dim_factor, 1)

### make it so the whole thing loops!
widerFrms = prevPnt2.stream(confs = confs).widen(2 * transition_frames).decap().frames.rescale(1.5)

fsFrms2 = widerFrms[:transition_frames]
lsFrms2 = widerFrms[-transition_frames:]

transFrms = fsFrms2.dim_progressively(0, 1) + lsFrms2.dim_progressively(1, 0)

jointFrms = BGRFrames.from_frames([fsFrms, animFrms_pre,
                                   combFrms4_joint, combFrms3_joint, combFrms2_joint, combFrms1_joint,
                                   lsFrms_with_marks, transFrms])

### check the lengths

animFrms_pre.length() + combFrms4_joint.length() + combFrms3_joint.length() +  combFrms2_joint.length() + combFrms1_joint.length()

rotFrms.length() + moveFrms4.length() + startstopFrms.length()

### TODO: put boxes on the same x-coordinate -> no jumping left and right!

# DecapStream(jointFrms, 0, 100).show()

jointFrms.save(fullPath, f'broadcast', lossless = True, save_as = 'mp4')

# lsFrms[-1].show()
