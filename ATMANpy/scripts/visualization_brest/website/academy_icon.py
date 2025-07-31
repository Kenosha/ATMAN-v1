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

DESCRIPTOR = 'academy_icon'
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

# court
def make_court(courtLines, inner_color, outer_color, 
               line_color = const.BGR['bright_grey'], 
               net_color = const.BGR['smokey_grey'],
               singles_net = False, net_alpha = 0.75, net_mesh_density = 3):
        
    courtScn = Scene.make_court(courtLines, make_net = True, singles_net = singles_net, line_color = line_color, 
                                net_color = net_color, net_alpha = net_alpha, net_mesh_density = net_mesh_density)

    courtLines3D, courtNet = courtScn

    courtReg_inner = courtLines.get_court_region()
    courtReg_outer = courtReg_inner.scale(1.4)

    courtReg_inner_3D = Region3D.from_region2D(courtReg_inner, z = -1).recolor(inner_color)
    courtReg_outer_3D = Region3D.from_region2D(courtReg_outer, z = -2).recolor(outer_color)

    return Scene([courtLines3D, courtNet, courtReg_inner_3D, courtReg_outer_3D])

def make_many_courts(courtLines):
    clay_inner_color = const.BGR['clay_orange']
    clay_outer_color = const.BGR['clay_orange']
        
    hard_inner_color = const.BGR['bright_blue']
    hard_outer_color = const.BGR['marine_blue']

    w, l = courtLines.court_width, courtLines.side_length * 2
    scale_factor = 1.45

    num_courts = 2
    
    shift_x = 0
    shift_y = 0
    court_scenes = []
    shifts = []
    
    ### clay courts
    for i in range(num_courts):
        shift_y = 0
        for j in range(num_courts):
            currScn = make_court(courtLines.right(shift_x).down(shift_y), clay_inner_color, clay_outer_color)
            court_scenes.append(currScn)

            shifts.append((shift_x, shift_y))
            shift_y += int(l * scale_factor)
        shift_x += int(w * scale_factor)
    
    shift_x += 100

    ### hard courts
    for i in range(num_courts):
        shift_y = 0
        for j in range(num_courts):
            currScn = make_court(courtLines.right(shift_x).down(shift_y), hard_inner_color, hard_outer_color)
            court_scenes.append(currScn)

            shifts.append((shift_x, shift_y))
            shift_y += int(l * scale_factor)
        shift_x += int(w * scale_factor)

    return court_scenes, shifts
    
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

orig_3D = Point3D((-4000, -3000, 3500))
# trgt_3D = Point3D((3000, 3000, 0))
trgt_3D = Point3D((1000, 1000, 0))

# starting cam
myCam = Camera.from_to(orig_3D, trgt_3D, f = 3000, sensor_dims = (640, 360))



scenes, shifts = make_many_courts(courtLines)
allScn = Scene.make_empty()
for scn in scenes:
    allScn = allScn.combine(scn)

overFrm = BGRFrame.make_black(640, 360) + const.BGR['darker_grey']

# from ATMANpy.types.Overlays import FrameRaster
# testRst = FrameRaster(3, 3)
# stds = [10, 15, 20]
# ress = [0.25, 0.5, 1]
# blrs = [(3, 3), (5, 5), (7, 7)]
# test_frames = []
# for i, std in enumerate(stds):
#     for j, blr in enumerate(blrs):
#         testFrm = View.make(myCam, allScn, resolution = ress[1]).snap(over_array = overFrm.array.copy())
#         testFrm = testFrm.gaussian_noise(std = std).gaussian_blur(blr)
#         testRst.add(testFrm, i, j)

# testRst.render().show()

res = 0.5
std = 10
blr = (5, 5)

backFrm = View.make(myCam, allScn, resolution = res).snap(over_array = overFrm.array)
backFrm = backFrm.gaussian_noise(std = std).gaussian_blur(blr)
backFrm.show()

### test out points playing out on the court.

myShots = dataPnts[0]
def get_joint_scenes(myShots, shift_x = 0, shift_y = 0,
                     smoothing_span = 1, rolling_window = 1,
                     bot_player_color = const.BGR['sign_orange'], top_player_color = const.BGR['sign_orange'],
                     pad_to_length = 250, roll_by = 0):
    shtScns, bncScns = vis_ut.get_shot_scenes(myShots, mark_color = const.BGR['light_brown'], mark_size = 1)

    botSF = myShots.feat_bot_player(smoothing_span = smoothing_span, rolling_window = rolling_window)
    topSF = myShots.feat_top_player(smoothing_span = smoothing_span, rolling_window = rolling_window)

    botPlys, topPlys = list(botSF.values()), list(topSF.values())

    ### move shapes
    joint_scenes = []
    for shtScn, botPly, topPly in zip(shtScns, botPlys, topPlys):
        shapes = []
        if not shtScn.is_empty():
            shtLn = shtScn[0]
            shtLn = shtLn.right(shift_x).back(shift_y)
            shapes.append(shtLn)

        botPly = botPly.right(shift_x).back(shift_y).recolor(bot_player_color)
        topPly = topPly.right(shift_x).back(shift_y).recolor(top_player_color)

        shapes.append(botPly)
        shapes.append(topPly)

        moveScn = Scene(shapes)
        joint_scenes.append(moveScn)

    botPly_fs, botPly_ls = botPlys[0], botPlys[-1]
    topPly_fs, topPly_ls = topPlys[0], topPlys[-1]
    
    botPly_fs = botPly_fs.right(shift_x).back(shift_y).recolor(bot_player_color)
    botPly_ls = botPly_ls.right(shift_x).back(shift_y).recolor(bot_player_color)
    topPly_fs = topPly_fs.right(shift_x).back(shift_y).recolor(top_player_color)
    topPly_ls = topPly_ls.right(shift_x).back(shift_y).recolor(top_player_color)

    break_frames = pad_to_length - len(joint_scenes)
    assert break_frames >= 10

    intermediate_scenes = vs.interpolate_scenes(Scene([botPly_ls, topPly_ls]), Scene([botPly_fs, topPly_fs]),
                                                num_frames = break_frames)
    
    joint_scenes.extend(intermediate_scenes)

    joint_scenes = [*joint_scenes[roll_by:], *joint_scenes[:roll_by]]

    return joint_scenes

from ATMANpy.funcs.filter_shots_funcs import is_out_wide
from ATMANpy.types.ShapesTennis3D import Player

def get_serve_practice(myShots, shift_x = 0, shift_y = 0, coach_back = 50, coach_left = 50,
                       smoothing_span = 1, rolling_window = 1,
                       coach_color = const.BGR['sign_orange'], player_color = const.BGR['sign_orange'],
                       pad_to_length = 250, roll_by = 0):
    ## coach
    x_avg, y_avg = np.mean([shot.hitPlayer.get_initial_position() for shot in myShots], axis = 0)
    coaPly = Player.make((x_avg, y_avg), color = coach_color)
    coaPly = coaPly.back(coach_back).left(coach_left).right(shift_x).back(shift_y)

    joint_scenes = []
    for idx, myShot in enumerate(myShots):
        ### shots
        shtScns, bncScns = vis_ut.get_shot_scenes([myShot], mark_color = const.BGR['light_brown'], mark_size = 1)
        
        ### server
        srv_from = myShot.get_hit_side()

        if srv_from == 'Top':
            plySF = myShot.feat_top_player(smoothing_span = smoothing_span, rolling_window = rolling_window)
        elif srv_from == 'Bot':
            plySF = myShot.feat_bot_player(smoothing_span = smoothing_span, rolling_window = rolling_window)
        srvPlys = list(plySF.values())
        
        ### move shapes
        for shtScn, srvPly in zip(shtScns, srvPlys):
            shapes = []
            if not shtScn.is_empty():
                shtLn = shtScn[0]
                shtLn = shtLn.right(shift_x).back(shift_y)
                shapes.append(shtLn)

            srvPly = srvPly.right(shift_x).back(shift_y).recolor(player_color)

            shapes.append(srvPly)
            shapes.append(coaPly)

            moveScn = Scene(shapes)
            joint_scenes.append(moveScn)

        ### little break: make the player move to the next serve position
        next_idx = idx + 1 if idx + 1 < len(myShots) else 0
        
        next_srvPly = Player.make(myShots[next_idx].hitPlayer.get_initial_position())
        next_srvPly = next_srvPly.right(shift_x).back(shift_y).recolor(player_color)

        if next_idx == 0:
            break_frames = pad_to_length - len(joint_scenes)
            assert break_frames >= 10
        else:
            break_frames = 50
        
        intermediate_scenes = vs.interpolate_scenes(Scene([srvPly]), Scene([next_srvPly]), num_frames = break_frames)
        intermediate_scenes = [scn.put(coaPly) for scn in intermediate_scenes]

        joint_scenes.extend(intermediate_scenes)

    joint_scenes = [*joint_scenes[roll_by:], *joint_scenes[:roll_by]]

    return joint_scenes

pad_to_length = 250

### court 0
x, y = shifts[0]
joint_scenes_a = get_joint_scenes(dataPnts[0], x, y, rolling_window = 15, pad_to_length = pad_to_length)
# print(len(joint_scenes_a))
# animFrms, _ = vs.animate_sequence(joint_scenes_a, start_cam = myCam, stop_cam = myCam, animate_over = backFrm)
# DecapStream(animFrms, 0, 100).show()

### court 1
x, y = shifts[1]
joint_scenes_b = get_joint_scenes(dataPnts[12], x, y, rolling_window = 15, pad_to_length = pad_to_length, roll_by = 50)
# print(len(joint_scenes_b))
# animFrms, _ = vs.animate_sequence(joint_scenes_b, start_cam = myCam, stop_cam = myCam, animate_over = backFrm)
# DecapStream(animFrms, 0, 100).show()

### court 5 ### GOOD!
x, y = shifts[5]
joint_scenes_c = get_joint_scenes(dataPnts[4], x, y, rolling_window = 15, pad_to_length = pad_to_length, roll_by = 150)
# print(len(joint_scenes_c))
# animFrms, _ = vs.animate_sequence(joint_scenes_c, start_cam = myCam, stop_cam = myCam, animate_over = backFrm)
# DecapStream(animFrms, 0, 100).show()

### court 4 ### GOOD!
def is_from_bot(myShot):
    return myShot.get_hit_side() == 'Bot'

srvShots = dataPnts.get_serves().filter('is_serve_ad').filter(is_out_wide).filter('is_first_serve').filter(is_from_bot)
srvShots = srvShots[:3]

x, y = shifts[2]
joint_scenes_d = get_serve_practice(srvShots, x, y, coach_left = 100, coach_back = 25, rolling_window = 15,
                                    player_color = const.BGR['sign_orange'], coach_color = const.BGR['sign_orange'],
                                    pad_to_length = pad_to_length, roll_by = 25)
# print(len(joint_scenes_d))
# animFrms, _ = vs.animate_sequence(joint_scenes_d, start_cam = myCam, stop_cam = myCam, animate_over = backFrm)
# DecapStream(animFrms, 0, 100).show()

### court 6 ### GOOD!
x, y = shifts[6]
joint_scenes_e = get_joint_scenes(dataPnts[17], x, y, rolling_window = 15, pad_to_length = pad_to_length, roll_by = 200)
# print(len(joint_scenes_e))
# animFrms, _ = vs.animate_sequence(joint_scenes_e, start_cam = myCam, stop_cam = myCam, animate_over = backFrm)
# DecapStream(animFrms, 0, 100).show()

#### join all the joint_scenes
scenes_list = [joint_scenes_a, joint_scenes_b, joint_scenes_c, joint_scenes_d, joint_scenes_e]

joint_scenes = [scn_a.combine(scn_b).combine(scn_c).combine(scn_d).combine(scn_e) for scn_a, scn_b, scn_c, scn_d, scn_e in zip(*scenes_list)]

animFrms, _ = vs.animate_sequence(joint_scenes, start_cam = myCam, stop_cam = myCam, animate_over = backFrm)

repeatFrms = BGRFrames.from_frames([animFrms, animFrms, animFrms])
DecapStream(repeatFrms, 0, 100).show()

animFrms.save(fullPath, f'academy', lossless = True, save_as = 'mp4')


