
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import scipy

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Abstract import Points, Point, read_in_all_tables

from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player, Racquet, Mark, Court
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D, Lines3D
from ATMANpy.types.Overlays import Legend, overlay

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from ATMANpy.auxil.court import courtLines
from ATMANpy.auxil.court import onto_service_box_cam

import ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils as vis_ut

from ATMANpy.funcs.util_funcs import get_folder_struct

#### READ IN CONFIGS ####
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

name = ppl._name
graphPath = ut.get_folder_struct('graphs')['serve_under_pressure']
sub_graphPath = os.path.join(graphPath, 'Deuce')

### SAVE FOLDER 
DESCRIPTOR = 'transition_deuce_to_ad'
SAVE_FOLDER = 'serve_pressure'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

### PARAMETERS #####
RESOLUTION_MODIFIER = 3 # 2
RESOLUTION_BACKGROUND = 1 * RESOLUTION_MODIFIER
RESOLUTION_OBJECTS = 1 * RESOLUTION_MODIFIER
RESOLUTION_KDE = 2 * RESOLUTION_MODIFIER
MIX_COLORS_KDE = True
BANDWIDTH_KDE = 35
SCENE_FADE = 0.9
CONTOUR_PROP = 0.5
EPSILON = 0.002
EMPHASIS_DIM_FACTOR = 1.5
ARGUMENT_LUMINOSITY = 1.25
ARGUMENT_BLINKS_PER_SECOND = None
ADAPTIVE_RESOLUTION_EXPONENT = 1.5
BLINK_RATE = 1
prmDict = {'background_resolution': RESOLUTION_BACKGROUND,
            'argument_resolution': RESOLUTION_OBJECTS,
            'foreground_resolution': RESOLUTION_OBJECTS,
            'dummy_resolution': RESOLUTION_KDE,
            'adaptive_resolution_exponent': ADAPTIVE_RESOLUTION_EXPONENT,
            'argument_luminosity': ARGUMENT_LUMINOSITY,
            'argument_blinks_per_second': ARGUMENT_BLINKS_PER_SECOND,
            }

prmDict_back = {**prmDict}
prmDict_back['background_resolution'] /= 2 ### reduce the background resolution when the full background is in view, to lower the Moiree effect on the net.

### READ IN DATA
allTable = read_in_all_tables(confs)
allPnts = Points.from_datatable(allTable, split_by_match = False)
firstPnts = allPnts.filter('is_first_serve').enforce_direction('Bot')
deucePnts = firstPnts.filter('is_serve_deuce')
adPnts = firstPnts.filter('is_serve_ad')

## camera
deuce_boxCam = onto_service_box_cam(which = 'Deuce', side = 'Top', dist = 150).tilt(0.15)
ad_boxCam = onto_service_box_cam(which = 'Ad', side = 'Top', dist = 150).tilt(0.15)
courtScn = Scene.make_court()

## service line and region
srvLn, indDict = vis_ut.get_service_line_and_indicator('Top', 'Deuce') 
lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

## make courtlines with just service line and deuce service line -> this way, we can make it look more like a graph.
width = courtLines.see_dimensions()['otherline_width']
x_ax, y_ax = srvLn, courtLines['Top Center Line']
x_ax_3D = Region3D.from_line3D(Line3D.from_line2D(x_ax, z = 0), width = width, dim = 'y')
y_ax_3D = Region3D.from_line3D(Line3D.from_line2D(y_ax, z = 0), width = width, dim = 'x')
x_y_scene = Scene([x_ax_3D, y_ax_3D])
backView = View.make(deuce_boxCam, x_y_scene, resolution = RESOLUTION_BACKGROUND)

origPt = x_ax.weak_intersect(y_ax)
origPt_3D = Point3D.from_point2D(origPt, z = 0)
tPt = origPt_3D.left(5).front(10).annotate('Down the Middle')    
wPt = origPt_3D.left(100).front(10).annotate('Out Wide')

## player list
players = list(set([pnt.get_srv_player_name() for pnt in deucePnts]))

all_frames = []
script_lines = []

##################################
##### LAST DEUCE PLAYER ##########
##################################
player_name = 'Federico Gaio'

### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} deuce serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### MAKE KDES WITH SPLIT CATEGORIES
lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

low_kde_scene = vs.kde_hillmap(lowShots, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line, 
                               bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
mid_kde_scene = vs.kde_hillmap(midShots, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line, 
                               bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
high_kde_scene = vs.kde_hillmap(hiShots, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line, 
                                bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

mid_kde_scene.rastArrs[0] += [0, 60, 0]
high_kde_scene.rastArrs[0] += [0, 120, 0]

split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

loPt = origPt_3D.right(25).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
midPt = origPt_3D.back(70).right(35).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
hiPt = origPt_3D.back(150).right(37).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

######## Tests for similarity of distribution #########
diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
jntStr = vis_ut.make_stat_string(lowShots, hiShots)
statPt = diagPt.annotate(jntStr).back(110).left(40)

##### Player name and handedness ####
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness)

script_line = ''
myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = deuce_boxCam, num_frames = 10,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)
#myFrms[-1].show()

itpl_court_scenes = vs.interpolate_scenes(courtScn.transparify(0), courtScn, 25)

animDict['previous_annotated_points'] = None
prmDict_no_illum = {k: v for k, v in prmDict.items()}
prmDict_no_illum['argument_luminosity'] = 1
prmDict_no_illum['argument_resolution'] = prmDict_back['background_resolution']

script_line = ''
myFrms, animDict = vs.animate_sequence(itpl_court_scenes, background_scene = x_y_scene, start_cam = deuce_boxCam, stop_cam = deuce_boxCam,
                                       **prmDict_no_illum, cumulative_foreground_transparence = 0.9, cumulate_argument = False,
                                       **animDict)

all_frames.append(myFrms)
script_lines.append(script_line)

script_line = ''
myFrms, animDict = vs.animate_cam_motion(None, background_scene = courtScn, start_cam = deuce_boxCam, stop_cam = ad_boxCam, num_frames = 25,
                                         **prmDict_back)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Consider again lefty serves, this time hit to the ad side. It is typically argued that the lefty serve enjoys an advantage from the ad-side. with a slice out wide allowing the server to target the backhand side AND forcing the receiver wide, off the court. We begin with Hugo Gaston.'
myFrms, animDict = vs.animate_static(None, background_scene = courtScn, static_cam = ad_boxCam, num_frames = 25,
                                     **prmDict_back)
all_frames.append(myFrms)
script_lines.append(script_line)

#########################
##### HUGO GASTON #######
#########################
player_name = 'Hugo Gaston'

### COLLECT SHOTS FOR PLAYER
serveShots = adPnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_ad)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)


### MAKE BLINKING LEGEND FRAME ###
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness)

# script_line = ''
# myFrms = vs.blinking(overlay(myFrms[-1], legFrm), myFrms[-1], duration_in_frames = 50, blinks_per_second = 2)
# all_frames.append(myFrms)
# script_lines.append(script_line)

### MAKE KDE WITH ALL ON BASELINE
joint_kde_null_scene = vs.kde_hillmap(serveShots_sample, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
                                resolution = RESOLUTION_KDE, normalize_density = True, max_height = -0.5, one_color = const.BGR['strong_blue'])
joint_kde_scene = vs.kde_hillmap(serveShots_sample, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['strong_blue'])

joint_itpl_scenes = vs.interpolate_dummy_scenes(joint_kde_null_scene, joint_kde_scene, num_frames = 25)

script_line = ''
myFrms, animDict = vs.animate_sequence(joint_itpl_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = ad_boxCam, stop_cam = ad_boxCam,
                                    **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'In the aggregate, the serve out wide is indeed considerably more prevalent.'
myFrms, animDict = vs.animate_static(joint_kde_scene, background_scene = courtScn, static_cam = ad_boxCam, num_frames = 50,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### MAKE KDES WITH SPLIT CATEGORIES
lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

low_kde_scene = vs.kde_hillmap(lowShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
mid_kde_scene = vs.kde_hillmap(midShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
high_kde_scene = vs.kde_hillmap(hiShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

mid_kde_scene.rastArrs[0] += [0, 60, 0]
high_kde_scene.rastArrs[0] += [0, 120, 0]

joint_kde_scenes = joint_kde_scene.combine(joint_kde_scene).combine(joint_kde_scene)
split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

split_itpl_scenes = vs.interpolate_dummy_scenes(joint_kde_scenes, split_kde_scenes, num_frames = 25)


## make courtlines with just service line and ad service line -> this way, we can make it look more like a graph.
srvLn, indDict = vis_ut.get_service_line_and_indicator('Top', 'Ad') 
width = courtLines.see_dimensions()['otherline_width']
x_ax, y_ax = srvLn, courtLines['Top Center Line']
x_ax_3D = Region3D.from_line3D(Line3D.from_line2D(x_ax, z = 0), width = width, dim = 'y')
y_ax_3D = Region3D.from_line3D(Line3D.from_line2D(y_ax, z = 0), width = width, dim = 'x')
x_y_scene = Scene([x_ax_3D, y_ax_3D])
backView = View.make(ad_boxCam, x_y_scene, resolution = RESOLUTION_BACKGROUND)

origPt = x_ax.weak_intersect(y_ax)
origPt_3D = Point3D.from_point2D(origPt, z = 0)
tPt = origPt_3D.right(15).front(10).annotate('Down the Middle')    
wPt = origPt_3D.right(110).front(10).annotate('Out Wide')


loPt = origPt_3D.left(15).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
midPt = origPt_3D.back(70).left(20).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
hiPt = origPt_3D.back(150).left(25).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

######## Tests for similarity of distribution #########
diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
jntStr = vis_ut.make_stat_string(lowShots, hiShots)
statPt = diagPt.annotate(jntStr).back(110).left(40)

jntStr_comb = vis_ut.make_stat_string(lowShots, midShots.join(hiShots), lo_desc = 'Low Pressure', hi_desc = 'Mid + High Pressure')
statPt_comb = diagPt.annotate(jntStr_comb).back(110).left(40)


### FADE OUT COURT ###
court_fadeout_scenes = vs.interpolate_scenes(courtScn, courtScn.transparify(0), num_frames = 25)
prmDict_no_illum = {k: v for k, v in prmDict.items()}
prmDict_no_illum['argument_luminosity'] = 1
prmDict_no_illum['argument_resolution'] = prmDict_back['background_resolution']

script_line = ''
myFrms, animDict = vs.animate_sequence(court_fadeout_scenes, background_scene = x_y_scene, cumulate_argument = False, 
                                       start_cam = ad_boxCam, stop_cam = ad_boxCam,
                                       **prmDict_no_illum,
                                       **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### animate
script_line = ''
myFrms, animDict = vs.animate_sequence(split_itpl_scenes, background_scene = x_y_scene, cumulate_argument = False, 
                                        start_cam = ad_boxCam, stop_cam = ad_boxCam,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

jntFrms = BGRFrames.from_frames(all_frames)
DecapStream(jntFrms, 0, 100).show(ms_between_frames = 50)


### Saving the separate MultiFrame objects and the Joint Script
from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

# b) Save segments
for idx, subFrms in enumerate(all_frames):
    subFrms.save(fullPath, f'seg_{idx}', save_as = 'mp4', lossless = True)

# c) Save script
with open(os.path.join(fullPath, 'script.txt'), mode = 'w') as f:
    f.writelines([l + '\n' for l in script_lines])

    