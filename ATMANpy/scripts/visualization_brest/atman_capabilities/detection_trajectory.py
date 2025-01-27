import os
import numpy as np
import pandas as pd
import sounddevice as sd
        
from ATMANpy.funcs.util_funcs import get_folder_struct, split
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames, MaskFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Visual3D import Camera, View, Scene

from ATMANpy.types.Abstract import Points, Shots, read_in_data_table, read_in_placeholder_table

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

DESCRIPTOR = 'trajectory'
SAVE_FOLDER = 'atman_capabilities'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

#########################################

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
### CHOOSE VIDEO
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
for idx, vid in enumerate(avail_vids):
    print(idx, vid)

idx = 12
conf = cnf_mod.get_conf(avail_vids[idx])

### GET ABSTRACT POINT REPRESENTATION 
dataTable = read_in_data_table(conf, conf_idx = idx)
nullTable = read_in_placeholder_table(conf, conf_idx = idx)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

################################
###### GET USEFUL STUFF ########
################################

vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
fs, ls = conf['start_frame'], conf['stop_frame']
cap = conf['cap']

segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()


dim_factor = 0.5

myPnt = allPnts[0]
# myPnt.stream_ball(P, cap = cap).show()

sht0, sht1, sht2 = myPnt[0], myPnt[1], myPnt[2]
stm0, stm1, stm2 = sht0.stream(cap), sht1.stream(cap), sht2.stream(cap)

frms0, frms1, frms2 = stm0.decap().frames.rescale(1.5), stm1.decap().frames.rescale(1.5), stm2.decap().frames.rescale(1.5)

### get ball splines

line_thickness = 1.5

ball_color = const.BGR['ball_yellow']

splns0 = sht0.feat_spline_cumul(which_endpoint = 'penultimate').values()
splns1 = myPnt[1].feat_spline_cumul(which_endpoint = 'penultimate').values()
splns2 = myPnt[2].feat_spline_cumul(which_endpoint = 'penultimate').values()

from ATMANpy.types.Visual3D import Camera, Scene
myCam_small = Camera.calibrated_brest_behind()
myCam = Camera.calibrated_brest_behind().resize(1.5)
myCam_full = Camera.calibrated_brest_behind().resize(3)

scns0 = [Scene([spln.recolor(ball_color).resize(line_thickness) for spln in splns if spln is not None]) for splns in splns0]
scns1 = [Scene([spln.recolor(ball_color).resize(line_thickness) for spln in splns if spln is not None]) for splns in splns1]
scns2 = [Scene([spln.recolor(ball_color).resize(line_thickness) for spln in splns if spln is not None]) for splns in splns2]

from ATMANpy.funcs.vis_funcs import animate_sequence

spln_frms0, _ = animate_sequence(scns0, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms0)
spln_frms1, _ = animate_sequence(scns1, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms1)
spln_frms2, _ = animate_sequence(scns2, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms2)

# DecapStream(spln_frms0, 0, 100).show(ms_between_frames = 100)
# DecapStream(spln_frms1, 0, 100).show(ms_between_frames = 100)
# DecapStream(spln_frms2, 0, 100).show(ms_between_frames = 100)

ls_frm_dim0 = spln_frms0[-1].dim(dim_factor)
ls_frm_dim1 = spln_frms1[-1].dim(dim_factor)
ls_frm_dim2 = spln_frms2[-1].dim(dim_factor)
frms0_ls, frms1_ls, frms2_ls = BGRFrames.from_frame(ls_frm_dim0, 25), BGRFrames.from_frame(ls_frm_dim1, 25), BGRFrames.from_frame(ls_frm_dim2, 25)

### pre_frames for spln_frms0, 1, 2
widen_by = 15
pre_frms0 = sht0.stream(cap).widen(widen_by).decap().frames[:widen_by].rescale(1.5) ### this is not needed, has its own pre_frms later!
pre_frms1 = sht1.stream(cap).widen(widen_by).decap().frames[:widen_by].rescale(1.5)
pre_frms2 = sht2.stream(cap).widen(widen_by).decap().frames[:widen_by].rescale(1.5)

### data for tables
myShot = sht0
def get_data_string(myShot):
    velo = myShot.get_initial_speed()
    spin = myShot.get_initial_spin()
    angl = myShot.get_launch_angle()

    veloStr = f'{int(round(velo, 0))} km/h'
    spinStr = f'{int(round(spin, 0))} RPM'
    anglStr = f'{round(angl, 1)} deg.'

    return f'Speed: {veloStr} \n\nSpin: {spinStr} \n\nLaunch Angle: {anglStr}'

### THE TABLE ###
str0, str1, str2 = vis_ut.get_shot_string(sht0), vis_ut.get_shot_string(sht1), vis_ut.get_shot_string(sht2)
dat0, dat1, dat2 = vis_ut.get_data_string(sht0), vis_ut.get_data_string(sht1), vis_ut.get_data_string(sht2)

tab_params = {  'index_offset': 1, 'index_col_width': 0.5, 'dimensions': (960, 540), 'skeleton_shrinkage': 0.2, 
                'automatic_vertical_offset_text_line_height': 0.4,
                'null_columns': 0
                }

tab_data0 = {'': {'Description': '', 'Data': ''}}
tab_data1 = {'1st Shot': {'Description': str0, 'Data': dat0}}
tab_data2 = {'1st Shot': {'Description': str0, 'Data': dat0}, '2nd Shot': {'Description': str1, 'Data': dat1}}
tab_data3 = {'1st Shot': {'Description': str0, 'Data': dat0}, '2nd Shot': {'Description': str1, 'Data': dat1}, '3rd Shot': {'Description': str2, 'Data': dat2}}

tab0 = Table(tab_data0, **tab_params, null_rows = 2)
tab1 = Table(tab_data1, **tab_params, null_rows = 2)
tab2 = Table(tab_data2, **tab_params, null_rows = 1)
tab3 = Table(tab_data3, **tab_params, null_rows = 0)

tab_frm0 = tab0.render()
tab_frm1 = tab1.render()
tab_frm2 = tab2.render()
tab_frm3 = tab3.render()

tab_frms0 = BGRFrames.from_frame(tab_frm0, 25)
tab_frms1 = BGRFrames.from_frame(tab_frm1, 25)
tab_frms2 = BGRFrames.from_frame(tab_frm2, 25)
tab_frms3 = BGRFrames.from_frame(tab_frm3, 25)

# tab_frm0.show()
# tab_frm1.show()
# tab_frm2.show()
# tab_frm3.show()

### THE RASTER ###
myRst = FramesRaster(2, 2)

myRst.add(spln_frms0, 0, 0)
myRst.add(tab_frms1, 1, 1)
rstFrms0 = myRst.render()
# DecapStream(rstFrms0, 0, 100).show(ms_between_frames = 100)

myRst.add(frms0_ls, 0, 0)
myRst.add(pre_frms1.join(spln_frms1), 0, 1)
myRst.add(tab_frms2, 1, 1)
rstFrms1 = myRst.render()
# DecapStream(rstFrms1, 0, 100).show(ms_between_frames = 100)

myRst.add(frms1_ls, 0, 1)
myRst.add(pre_frms2.join(spln_frms2), 1, 0)
myRst.add(tab_frms3, 1, 1)
rstFrms2 = myRst.render()
# DecapStream(rstFrms2, 0, 100).show(ms_between_frames = 100)

# rstFrms2[-1].show()

def ball_scene( ballPt, radius = 5, 
                ball_color = const.BGR['ball_yellow'], ball_dim_factor = 0.75, 
                line_color = const.BGR['bright_grey'], line_dim_factor = 1):
    # idea for rotation: two circles at 1/4 and 3/4 height of the ball 
    #   -> these rotate with the ball, and look like the lines on the ball
    #       (even though the lines on the ball are not circles.)
    #           like in these pictures: https://www.decathlon.de/p/tennisball-tb-jumbo-ball/_/R-p-3244

    line_lim_phi_1 = (0.25*np.pi, 0.3*np.pi)
    line_lim_phi_2 = (0.75*np.pi, 0.8*np.pi)
    
    myBall = Ball3D.from_point3D(ballPt, radius = radius, offset = 0).recolor(ball_color).dim(ball_dim_factor)
    secBall_1 = Ball3D.from_point3D(ballPt, radius = 1.001 * radius, offset = 0, limit_phi = line_lim_phi_1, rot_x = 0).recolor(line_color).dim(line_dim_factor)
    secBall_2 = Ball3D.from_point3D(ballPt, radius = 1.001 * radius, offset = 0, limit_phi = line_lim_phi_2, rot_x = 0).recolor(line_color).dim(line_dim_factor)
    ballScn = Scene([myBall, secBall_1, secBall_2])
    return ballScn

def animate_example_trajectory(ypf, zpf, rpf, ballScn, sim_frames = 62, sim_res = 5, staticCam = None):
    ### ypf, zpf, rpf are y displacement, z displacement, and rotation per frame
    currScn = ballScn
    out_frames = []
    prevScn = Scene([])
    for idx in range(sim_frames):
        
        y = ypf * np.exp(-0.01 * idx)
        z = zpf * np.exp(-0.05 * idx)
        r = rpf

        currBall = currScn[0]
        currPt = currBall.center_point()
        
        ### create the camera, or hold it at center
        if staticCam is None:
            if idx <= sim_frames // 2:
                currCam = Camera.from_to(currPt.right(100), currPt)
            else:
                currCam = currCam
        else:
            currCam = staticCam

        currView = View.make(currCam, currScn.combine(prevScn), resolution = sim_res)
        currFrm = currView.snap()
        out_frames.append(currFrm)

        new_shapes = []
        for shape in currScn:
            new_shapes.append(shape.front(y).up(z).rotate_x(r))
        currScn = Scene(new_shapes)
        
        if idx % 5 == 0:
            prevScn = prevScn.combine(currScn)

    outFrms = BGRFrames.from_single_frames(out_frames)

    return outFrms, prevScn, currCam

### parameters
sim_res = 25
sim_frames = 62

### SIMULATION ANIMATION ###
from ATMANpy.types.Shapes3D import Point3D, Ball3D

all_frames = []
script_lines = []

script_line = "Trajectories are detected by evaluating all the empirical information we have gathered so far within a physical simulation."

ypf, zpf, rpf = 3.5, 3, -0.1 * np.pi
ballPt = Point3D((100, 200, 200))

ballScn = ball_scene(ballPt, ball_dim_factor = 0.75, line_dim_factor = 1)
exFrms0, prevScn, currCam = animate_example_trajectory(ypf, zpf, rpf, ballScn, sim_res = sim_res, sim_frames = 62)
# DecapStream(exFrms0, 0, 100).show(ms_between_frames = 40)
# exFrms0[-1].show()

all_frames.append(exFrms0)
script_lines.append(script_line)

script_line = "The physical simulation takes into account all the forces acting on a ball in flight ..."
middle_idx = len(prevScn) // 3 // 2 * 3
middleScn = Scene(prevScn[middle_idx:middle_idx+3])

dimmed_prevScn = prevScn.dim(0.5)
dimmed_prevScn = dimmed_prevScn.drop(middle_idx).drop(middle_idx).drop(middle_idx)

dimFrm = View.make(currCam, dimmed_prevScn.combine(middleScn), resolution = sim_res).snap()
itplFrms = vs.interpolate_frames(exFrms0[-1], dimFrm, 25)
# DecapStream(itplFrms, 0, 100).show()

all_frames.append(itplFrms)
script_lines.append(script_line)

### LEGENDS
from ATMANpy.types.Overlays import Legend, LegendRow

grav_color = const.BGR['light_brown']
drag_color = const.BGR['strong_pink']
lift_color = const.BGR['marine_blue']
row_params = {'text_size': 2, 'text_thickness': 1, 'text_horizontal_offset': 1, 'text_vertical_offset': 0.3, 'shape_offset': 0.15}
leg_params = {'resolution': 5000, 'header_offset': 5.13, 'header_text': 'Forces', 'legend_dimensions': (600, 400), 'borders': False}

nRow = LegendRow()
gRow = LegendRow('Gravity', grav_color, **row_params)
dRow = LegendRow('Drag', drag_color, **row_params)
lRow = LegendRow('Lift', lift_color, **row_params)

l1 = Legend([gRow, nRow, nRow], **leg_params)
l2 = Legend([gRow, dRow, nRow], **leg_params)
l3 = Legend([gRow, dRow, lRow], **leg_params)

l1Frm = l1.render()
l2Frm = l2.render()
l3Frm = l3.render()
# l3Frm.show()

### ARROWS
from ATMANpy.types.ShapesTennis3D import Arrow
middleBall = middleScn[0]
middlePt = middleBall.center_point()
arr_len = 25
arr_params = {'hat_length': 3, 'thickness': 0.25, 'hat_angle': np.pi / 6, 'rot_angle': np.pi / 2}

script_line = '... like gravity ...'
gravArr = Arrow.make(middlePt, middlePt.down(arr_len), **arr_params, color = grav_color)
arrScn = dimmed_prevScn.combine(middleScn).put(gravArr)
arrFrm = View.make(currCam, arrScn, resolution = sim_res).snap()
arrFrm = overlay(arrFrm, l1Frm, top_left_point = Point2D((50, 50)))
# arrFrm.show()

arrFrms = BGRFrames.from_frame(arrFrm, 25)
all_frames.append(arrFrms)
script_lines.append(script_line)

script_line = '... drag ...'
y = ypf * np.exp(-0.01 * sim_frames / 2) * arr_len / 4
z = zpf * np.exp(-0.05 * sim_frames / 2) * arr_len / 4

dragArr = Arrow.make(middlePt, middlePt.back(y).down(z), **arr_params, color = drag_color)
arrScn = dimmed_prevScn.combine(middleScn).put(gravArr).put(dragArr)
arrFrm = View.make(currCam, arrScn, resolution = sim_res).snap()
arrFrm = overlay(arrFrm, l2Frm, top_left_point = Point2D((50, 50)))
# arrFrm.show()

arrFrms = BGRFrames.from_frame(arrFrm, 25)
all_frames.append(arrFrms)
script_lines.append(script_line)

script_line =   """ 
                    ... and lift. 
                    Lift is caused by the spin imparted on the ball: 
                    if we reverse the spin from backspin to topspin, 
                    the lift is reversed accordingly.
                """

### for lift: make the ball rotate, and adapt the arrow accordingly.
sim_frames = 350

currScn = middleScn
rs_per_frame = np.sin(np.linspace(0, 3 * np.pi / 2, num = sim_frames))

out_frames = []
for r in rs_per_frame:
    
    new_shapes = []
    for shape in currScn:
        new_shapes.append(shape.rotate_x(r))
    
    currScn = Scene(new_shapes)
    putScn = currScn.combine(dimmed_prevScn).put(gravArr).put(dragArr)

    up_force = r * y
    back_force = r * z

    liftArr = Arrow.make(middlePt, middlePt.back(back_force).up(up_force), **arr_params, color = lift_color)
    if liftArr[0].length() > 5:
        putScn = putScn.put(liftArr)

    currView = View.make(currCam, putScn, resolution = sim_res)
    currFrm = currView.snap()
    currFrm = overlay(currFrm, l3Frm, top_left_point = Point2D((50, 50)))

    out_frames.append(currFrm)

rotFrms = BGRFrames.from_single_frames(out_frames)
# DecapStream(rotFrms, 0, 100).show(ms_between_frames = 40)
# rotFrms[-1].show()

all_frames.append(rotFrms)
script_lines.append(script_line)

script_line =   """
                    ATMAN simulates around 4 to 5 million candidate trajectories.
                    In this schematic example, we show just a few.
                """
### -> for this line: 
    # 1) fade out arrows and legend, one by one
    # 2) show faintly other ball flight trajectories
# 1)
frms = 10
dropFrms = BGRFrames.from_frames([  BGRFrames.from_frame(currFrm, frms),
                                    BGRFrames.from_frame(View.make(currCam, putScn, resolution = sim_res).snap(), frms),
                                    BGRFrames.from_frame(View.make(currCam, putScn.drop(-1), resolution = sim_res).snap(), frms),
                                    BGRFrames.from_frame(View.make(currCam, putScn.drop(-1).drop(-1), resolution = sim_res).snap(), frms),
                                    BGRFrames.from_frame(View.make(currCam, putScn.drop(-1).drop(-1).drop(-1), resolution = sim_res).snap(), frms),
                                ])

fadeFrms = vs.interpolate_frames(dropFrms[-1], exFrms0[-1], num_frames = frms)

# DecapStream(dropFrms, 0, 100).show()
# dropFrms[-1].show()

# 2)
sim_frames = 62
exFrms0_ls = BGRFrames.from_frame(exFrms0[-1], sim_frames)
dist_1 = 10
dist_2 = 20
dist_3 = 30

dist_1 = 5
dist_2 = 10
dist_3 = 15

# 2a) inner ring
exPt = ballPt.up(dist_1)
exScn = ball_scene(exPt, ball_dim_factor = 0.6, line_dim_factor = 0.9)
ex_ypf, ex_zpf, ex_rpf = ypf - 0.1, zpf + 0.5, 0.5 * np.pi
exFrms_up, prevScn1, _ = animate_example_trajectory(ex_ypf, ex_zpf, ex_rpf, exScn, sim_frames = sim_frames, sim_res = sim_res, staticCam = currCam)

exPt = ballPt.down(dist_1)
exScn = ball_scene(exPt, ball_dim_factor = 0.6, line_dim_factor = 0.9)
ex_ypf, ex_zpf, ex_rpf = ypf + 0.1, zpf - 0.5, 0.5 * np.pi
exFrms_down, prevScn2, _ = animate_example_trajectory(ex_ypf, ex_zpf, ex_rpf, exScn, sim_frames = sim_frames, sim_res = sim_res, staticCam = currCam)

exFrms_inner = exFrms0_ls + exFrms_up + exFrms_down
# DecapStream(exFrms_inner, 0, 100).show()

exFrms_inner_ls = BGRFrames.from_frame(exFrms_inner[-1], sim_frames)

# 2b) outer ring
exPt = ballPt.up(dist_2)
exScn = ball_scene(exPt, ball_dim_factor = 0.45, line_dim_factor = 0.75)
ex_ypf, ex_zpf, ex_rpf = ypf - 0.3, zpf + 1.25, 0.5 * np.pi
exFrms_up, prevScn3, _ = animate_example_trajectory(ex_ypf, ex_zpf, ex_rpf, exScn, sim_frames = sim_frames, sim_res = sim_res, staticCam = currCam)

exPt = ballPt.down(dist_2)
exScn = ball_scene(exPt, ball_dim_factor = 0.45, line_dim_factor = 0.75)
ex_ypf, ex_zpf, ex_rpf = ypf + 0.3, zpf - 1.25, 0.5 * np.pi
exFrms_down, prevScn4, _ = animate_example_trajectory(ex_ypf, ex_zpf, ex_rpf, exScn, sim_frames = sim_frames, sim_res = sim_res, staticCam = currCam)

exFrms_outer = exFrms_inner_ls + exFrms_up + exFrms_down
# DecapStream(exFrms_outer, 0, 100).show()

exFrms_outer_ls = BGRFrames.from_frame(exFrms_outer[-1], sim_frames)

# 2c) even outer ring
exPt = ballPt.up(dist_3)
exScn = ball_scene(exPt, ball_dim_factor = 0.3, line_dim_factor = 0.6)
ex_ypf, ex_zpf, ex_rpf = ypf - 0.5, zpf + 2.25, 0.5 * np.pi
exFrms_up, prevScn5, _ = animate_example_trajectory(ex_ypf, ex_zpf, ex_rpf, exScn, sim_frames = sim_frames, sim_res = sim_res, staticCam = currCam)

exPt = ballPt.down(dist_3)
exScn = ball_scene(exPt, ball_dim_factor = 0.3, line_dim_factor = 0.6)
ex_ypf, ex_zpf, ex_rpf = ypf + 0.5, zpf - 2.25, 0.5 * np.pi
exFrms_down, prevScn6, _ = animate_example_trajectory(ex_ypf, ex_zpf, ex_rpf, exScn, sim_frames = sim_frames, sim_res = sim_res, staticCam = currCam)

exFrms_outer2 = exFrms_outer_ls + exFrms_up + exFrms_down
# DecapStream(exFrms_outer2, 0, 100).show()

jointFrms = BGRFrames.from_frames([dropFrms, fadeFrms, exFrms_inner, exFrms_outer, exFrms_outer2])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

### Focus on the one trajectory that is best, ostensibly.
def color_dim(my_ballScn, dim_factor = 0.5):
    rectScn = Scene([])
    for idx in range(len(my_ballScn)):
        if idx % 3 == 0:
            rectScn = rectScn.put(my_ballScn[idx].recolor(const.BGR['ball_yellow']))
        else:
            rectScn = rectScn.put(my_ballScn[idx].recolor(const.BGR['bright_grey']))
    return rectScn.dim(dim_factor)

cand_scenes = [prevScn, prevScn1, prevScn2, prevScn3, prevScn4, prevScn5, prevScn6]
choice_idx = 3
df1 = 0.25
df2 = 0.15

script_line =   """
                    The actual trajectory is then chosen among these possible trajectories as the one with which empirical evidence agrees with the most.
                    In this way, we get a trajectory that fits what ATMAN sees and hears, and also obeys the laws of physics.
                """
dimmed_candScn = Scene([])
for idx in range(len(cand_scenes)):
    curr_candScn = color_dim(cand_scenes[idx], dim_factor = df1) if idx != choice_idx else color_dim(cand_scenes[idx], dim_factor = 1)
    dimmed_candScn = dimmed_candScn.combine(curr_candScn)

postFrm = View.make(currCam, dimmed_candScn, resolution = sim_res).snap()
# postFrm.show()

itplFrms = vs.interpolate_frames(exFrms_outer2[-1], postFrm, num_frames = 100)
# DecapStream(itplFrms, 0, 100).show()

stopFrms = BGRFrames.from_frame(itplFrms[-1], 100)

# -> here, we move to the starting point of the chosen candidate trajectory, and show angle, speed and spin
choiceScn = cand_scenes[choice_idx]
choiceBall = choiceScn[0]
choicePt = choiceBall.get_point()
newCam = Camera.from_to(choicePt.right(70), choicePt)

very_dimmed_candScn = Scene([])
for idx in range(len(cand_scenes)):
    curr_candScn = color_dim(cand_scenes[idx], dim_factor = df2) if idx != choice_idx else color_dim(cand_scenes[idx], dim_factor = 1)
    very_dimmed_candScn = very_dimmed_candScn.combine(curr_candScn)

dimScns = vs.interpolate_scenes(dimmed_candScn, very_dimmed_candScn, 50)

movFrms, _ = vs.animate_sequence(dimScns, start_cam = currCam, stop_cam = newCam, argument_resolution = sim_res)
# DecapStream(movFrms, 0, 100).show()

jointFrms = BGRFrames.from_frames([itplFrms, stopFrms, movFrms])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "As a nice by-product..."
interFrms = BGRFrames.from_frame(jointFrms[-1], 25)
all_frames.append(interFrms)
script_lines.append(script_line)

script_line = "... the speed ..."

shape_color = const.BGR['intense_red']
lsScn = dimScns[-1]
nextBall = choiceScn[3]
nextPt = nextBall.get_point()
speedArr = Arrow.make(choicePt, nextPt, **arr_params, color = shape_color)
speedArr_pt = choicePt.annotate('Speed').recolor(shape_color).up(10).front(3.75)

arrScn = lsScn.put(speedArr)
diaFrm0 = View.make(newCam, arrScn, annotatedPoints = [speedArr_pt], resolution = sim_res).snap()
# diaFrm0.show()
diaFrms0 = BGRFrames.from_frame(diaFrm0, 25)

all_frames.append(diaFrms0)
script_lines.append(script_line)

script_line = "... spin ..."

from ATMANpy.types.Shapes3D import Circle3D
rotCrc = Circle3D(choicePt.coords, color = shape_color, radius = 6, fixed_dimension = 'x', limit_theta = (-np.pi / 2, np.pi / 2))

crcArr = rotCrc.rasterize(100)
lsPt0 = Point3D((tuple(crcArr[-2])))
lsPt1 = Point3D((tuple(crcArr[-1])))

rotArr = Arrow.make(lsPt0, lsPt1, hat_length = 1, thickness = 0, hat_angle = np.pi / 6, rot_angle = np.pi / 2, color = shape_color)
rotArr_pt = choicePt.annotate('Spin').recolor(shape_color).back(6.5).up(5)

rotScn = arrScn.put(rotCrc).put(rotArr)
diaFrm1 = View.make(newCam, rotScn, annotatedPoints = [speedArr_pt, rotArr_pt], resolution = sim_res).snap()
# diaFrm1.show()
diaFrms1 = BGRFrames.from_frame(diaFrm1, 25)

all_frames.append(diaFrms1)
script_lines.append(script_line)

script_line =   """ 
                    ... and launch angle of the ball can be directly read out 
                    as the initial conditions that uniquely describe the chosen trajectory.
                """

from ATMANpy.types.Shapes3D import Circle3D
angCrc = Circle3D(choicePt.coords, color = shape_color, radius = 10, fixed_dimension = 'x', limit_theta = (3 * np.pi / 4, np.pi))
angCrc_pt = choicePt.annotate('Angle').recolor(shape_color).front(5.5).up(2.5)

crcScn = rotScn.put(angCrc).put(rotCrc).put(rotArr)
diaFrm2 = View.make(newCam, crcScn, annotatedPoints = [speedArr_pt, angCrc_pt, rotArr_pt], resolution = sim_res).snap()
# diaFrm2.show()
diaFrms2 = BGRFrames.from_frame(diaFrm2, 25)

all_frames.append(diaFrms2)
script_lines.append(script_line)

script_line = ""
back_diaFrms = BGRFrames.from_frames([  BGRFrames.from_frame(diaFrm2, 10), 
                                        BGRFrames.from_frame(diaFrm1, 10),
                                        BGRFrames.from_frame(diaFrm0, 10),
                                        BGRFrames.from_frame(jointFrms[-1], 10)])
all_frames.append(back_diaFrms)
script_lines.append(script_line)

# fade to black
script_line = ""
zeroFrm = BGRFrame.make_black()
fadeFrms = vs.interpolate_frames(back_diaFrms[-1], zeroFrm, num_frames = 25)
all_frames.append(fadeFrms)
script_lines.append(script_line)

# vis_ut.show(all_frames)

##### REST OF THE ANIMATION ######
script_line =   """
                   Let's see the trajectory detection in action. 
                   Each of the rally's three shots occupies its own panel, with the panel in the lower right
                   showing the calculated speed, spin and angle of the shot. 
                """  # 10 seconds
pre_frames = 200
preFrms = stm0.widen(pre_frames).decap().frames[:pre_frames]
preArr_dimmed = np.stack([f.dim((idx / pre_frames) ** 2).array for idx, f in enumerate(preFrms)], axis = 0)
preFrms_dimmed = BGRFrames(preArr_dimmed).slow(2)

dimRst = FramesRaster(2, 2)
dimRst.add(preFrms_dimmed.rescale(1.5), 0, 0)
dimRst.add(tab_frms0, 1, 1)
rstFrms_init = dimRst.render()
# DecapStream(rstFrms_init, 0, 100).show()

all_frames.append(rstFrms_init)
script_lines.append(script_line)

script_line =   """
                    The serve is recognized as more of a kick serve than a flat serve; the landing location is very well detected, however. 
                    The serve is generally more challenging to recognize since it flies faster; 
                    therefore there are fewer frames for the ball to be tracked in, and the balls are blurry within the frames.
                    This would very likely be much less of an issue at image qualities higher than the 640-by-360, low-bitrate video analyzed here.
                """ ### 22 seconds
rstFrms_slow = rstFrms0.slow(2)
rstFrms_slow_add = BGRFrames.from_frame(rstFrms_slow[-1], 600)
rstFrms_slow_all = BGRFrames.from_frames([rstFrms_slow, rstFrms_slow_add])

all_frames.append(rstFrms_slow_all)
script_lines.append(script_line)

script_line =   """
                    Despite these limitations, the return is well recognized as a backhand slice at 77 km/h, 2500 RPM and a launch angle of 10.8 degrees.
                """ # 10 seconds
rstFrms_slow = rstFrms1.slow(2)
rstFrms_slow_pre = BGRFrames.from_frame(rstFrms_slow[0], 25)
rstFrms_slow_add = BGRFrames.from_frame(rstFrms_slow[-1], 200)

rstFrms_slow_all = BGRFrames.from_frames([rstFrms_slow_pre, rstFrms_slow, rstFrms_slow_add])

all_frames.append(rstFrms_slow_all)
script_lines.append(script_line)

script_line =   """
                    The server's subsequent forehand is tracked perfectly, flying out of the court at 111 km/h at a considerable topspin rate of 3750 RPM.
                """ # 11 seconds
rstFrms_slow = rstFrms2.slow(2)
rstFrms_slow_pre = BGRFrames.from_frame(rstFrms_slow[0], 25)
rstFrms_slow_add = BGRFrames.from_frame(rstFrms_slow[-1], 250)
rstFrms_slow_all = BGRFrames.from_frames([rstFrms_slow_pre, rstFrms_slow, rstFrms_slow_add])

all_frames.append(rstFrms_slow_all)
script_lines.append(script_line)

### remove raster elements, then remove trajectory, to prepare for zoom
from ATMANpy.types.Overlays import reverse_cascade, overlay

script_line = ""
finRst = myRst.to_frame_raster()
revRst, revFrms = reverse_cascade(finRst, [(1, 0)], frames_per_element = 15)
all_frames.append(revFrms)
script_lines.append(script_line)

script_line = ""
lsFrm = stm2.get(stm2.stop_frame).rescale(1.5)
y, x, _ = lsFrm.shape

blackFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))

tlPt = Point2D((0, 1080 - y))
overFrm = overlay(blackFrm, lsFrm, tlPt)
overFrms = BGRFrames.from_frame(overFrm, 10)

all_frames.append(overFrms)
script_lines.append(script_line)

### prep next points
nextPnt1 = allPnts[1]
nextPnt2 = allPnts[2]

stm_p1 = nextPnt1.stream(cap = cap)
stm_p2 = nextPnt2.stream(cap = cap)

### next few rallyes --> slowly zoom out at the end of the last shot.
script_line =   """
                    Let us now take a look at the next few rallyes to see more of the trajectory detection at work.
                """
fs_zoom, ls_zoom = stm2.stop_frame + 1, stm2.stop_frame + 101
zoomStm = Stream(cap, fs_zoom, ls_zoom - 1)
# zoomStm.show()

zoom_frames = []
for frm_num, frm, _ in zoomStm.generator():
    lmbd = (ls_zoom - frm_num) / (ls_zoom - fs_zoom)
    lmbd_rescale_factor = 1.5 * lmbd + 3 * (1 - lmbd)

    frm_resc = frm.rescale(lmbd_rescale_factor)
    y, x, _ = frm_resc.shape
    tlPt = Point2D((0, 1080 - y))
    
    frm_resc_over = overlay(blackFrm, frm_resc, tlPt)
    zoom_frames.append(frm_resc_over)

zoomFrms = BGRFrames.from_single_frames(zoom_frames)

### transition to beginning of trajectory
start, stop = ls_zoom, stm_p1.start_frame

transStm1 = Stream(cap, start, stop - 1)
transFrms = transStm1.decap().frames.rescale(3)
transFrms = vis_ut.speed_up_frames_progressively(transFrms, ramp_up_frames = 25)

zoom_and_transFrms = BGRFrames.from_frames([zoomFrms, transFrms])

all_frames.append(zoom_and_transFrms)
script_lines.append(script_line)

### show next points

script_line = ""

point_splns = nextPnt1.feat_spline_cumul(which_endpoint = 'penultimate').values()
point_splns = [[spln for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)] for splns in point_splns ]
scns = [Scene([spln.recolor(ball_color).resize(line_thickness) for spln in splns]) for splns in point_splns]

frms = stm_p1.decap().frames[:-1].rescale(3)

splnFrms1, _ = animate_sequence(scns, start_cam = myCam_full, stop_cam = myCam_full, cumulate_argument = False, animate_over = frms,
                                argument_resolution = 3)
# DecapStream(splnFrms1, 0, 100).show()

all_frames.append(splnFrms1)
script_lines.append(script_line)

### transition to the next trajectory
script_line = ""
start, stop = stm_p1.stop_frame, stm_p2.start_frame

transStm2 = Stream(cap, start, stop - 1)
transFrms = transStm2.decap().frames.rescale(3)
transFrms = vis_ut.speed_up_frames_progressively(transFrms, ramp_up_frames = 25)

all_frames.append(transFrms)
script_lines.append(script_line)

script_line = ""

point_splns = nextPnt2.feat_spline_cumul(which_endpoint = 'penultimate').values()
point_splns = [[spln for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)] for splns in point_splns ]
scns = [Scene([spln.recolor(ball_color).resize(line_thickness) for spln in splns]) for splns in point_splns]

frms = stm_p2.decap().frames[:-1].rescale(3)

splnFrms2, _ = animate_sequence(scns, start_cam = myCam_full, stop_cam = myCam_full, cumulate_argument = False, animate_over = frms,
                                argument_resolution = 3)
# DecapStream(splnFrms2, 0, 100).show()

all_frames.append(splnFrms2)
script_lines.append(script_line)

### finish up this traj -> not quite done!
script_line = ""
start, stop = stm_p2.stop_frame, stm_p2.stop_frame + 100 

transStm3 = Stream(cap, start, stop - 1)
transFrms = transStm3.decap().frames.rescale(3)

transFrms_dimmed = transFrms.dim_progressively(fs_dim_factor = 1, ls_dim_factor = 0)
# DecapStream(transFrms_dimmed, 0, 100).show()

all_frames.append(transFrms_dimmed)
script_lines.append(script_line)

vis_ut.show(all_frames)

from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

# b) Save segments
for idx, subFrms in enumerate(all_frames):
    subFrms.save(fullPath, f'seg_{idx}', save_as = 'mp4', lossless = True)

# c) Save script
for idx, script_line in enumerate(script_lines):
    with open(os.path.join(fullPath, f'script_{idx}.txt'), mode = 'w') as f:
        f.write(script_line)

# ### Check out trajectories for the 9 other examples:
# from ATMANpy.types.Abstract import NullPoint

# raster_frames = []
# for frm_idx, myPnt in enumerate(allPnts[:20]):
#     print(frm_idx)

#     if isinstance(myPnt, NullPoint):
#         stm = Stream.from_SF(segBFs[frm_idx], cap)
#         frms = stm.decap().frames
#         raster_frames.append(frms)
#         continue

#     point_splns = myPnt.feat_spline_cumul(which_endpoint = 'penultimate').values()

#     ### cull points in the net -> have a nan Point or two.
#     ### also cull None

#     point_splns = [[spln for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)] for splns in point_splns ]

#     scns = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns]) for splns in point_splns]

#     stm = myPnt.stream(cap = cap)
#     frms = stm.decap().frames[:-1]

#     myCam = Camera.calibrated_brest_behind()

#     frms_spln, _ = animate_sequence(scns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms)

#     raster_frames.append(frms_spln)

# for frm_idx, frms in enumerate(raster_frames[3:5]):
#     print(frm_idx)
#     DecapStream(frms, 0, 100).show()

# DecapStream(raster_frames[2].rescale(3), 0, 100).show(ms_between_frames = 100)

