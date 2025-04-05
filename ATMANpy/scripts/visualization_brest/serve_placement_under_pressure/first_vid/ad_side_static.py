
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
DESCRIPTOR = 'ad_side_static'
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


### READ IN DATA
allTable = read_in_all_tables(confs)
allPnts = Points.from_datatable(allTable, split_by_match = False)
firstPnts = allPnts.filter('is_first_serve').enforce_direction('Bot')
adPnts = firstPnts.filter('is_serve_ad')

## camera
boxCam = onto_service_box_cam(which = 'Ad', side = 'Top', dist = 150).tilt(0.15)

## service line and region
srvLn, indDict = vis_ut.get_service_line_and_indicator('Top', 'Ad') 
lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

## make courtlines with just service line and ad service line -> this way, we can make it look more like a graph.
width = courtLines.see_dimensions()['otherline_width']
x_ax, y_ax = srvLn, courtLines['Top Center Line']
x_ax_3D = Region3D.from_line3D(Line3D.from_line2D(x_ax, z = 0), width = width, dim = 'y')
y_ax_3D = Region3D.from_line3D(Line3D.from_line2D(y_ax, z = 0), width = width, dim = 'x')
x_y_scene = Scene([x_ax_3D, y_ax_3D])
backView = View.make(boxCam, x_y_scene, resolution = RESOLUTION_BACKGROUND)

origPt = x_ax.weak_intersect(y_ax)
origPt_3D = Point3D.from_point2D(origPt, z = 0)
tPt = origPt_3D.right(15).front(10).annotate('Down the Middle')    
wPt = origPt_3D.right(110).front(10).annotate('Out Wide')

## legend parameters
text_size = 1
thickness = 1
offset = 0.5
color = const.BGR['bright_grey']

## player list
players = list(set([pnt.get_srv_player_name() for pnt in adPnts]))

static_frames = {}

##################################
########### LIAM BROADY ##########
##################################
player_name = 'Liam Broady'

### COLLECT SHOTS FOR PLAYER
serveShots = adPnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_ad)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### MAKE KDES WITH SPLIT CATEGORIES
lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

low_kde_scene = vs.kde_hillmap(lowShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
mid_kde_scene = vs.kde_hillmap(midShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
high_kde_scene = vs.kde_hillmap(hiShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
								bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

mid_kde_scene.rastArrs[0] += [0, 60, 0]
high_kde_scene.rastArrs[0] += [0, 120, 0]

split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

loPt = origPt_3D.left(15).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
midPt = origPt_3D.back(70).left(20).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
hiPt = origPt_3D.back(150).left(25).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

######## Tests for similarity of distribution #########
diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
jntStr = vis_ut.make_stat_string(lowShots, hiShots)
statPt = diagPt.annotate(jntStr).back(110).left(40)

jntStr_comb = vis_ut.make_stat_string(lowShots, midShots.join(hiShots), lo_desc = 'Low Pressure', hi_desc = 'Mid + High Pressure')
statPt_comb = diagPt.annotate(jntStr_comb).back(110).left(40)

##### Player name and handedness ####
myLeg = Legend.make_empty(resolution = 100, header_offset = 3.25, legend_dimensions = (350, 200), header_thickness = 2)
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
l1 = {'text_string_shape': 'Player:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': player_name,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}        
l2 = {'text_string_shape': 'Handedness:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': handedness,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}
myLeg = myLeg.add(l1).add(l2)

legFrm = myLeg.render()

##### make still frames (with different elements added in)
frontView = View.make(boxCam, split_kde_scenes, annotatedPoints = [loPt, midPt, hiPt, tPt, wPt], resolution = RESOLUTION_KDE)
plainView = backView.combine(frontView)
plainFrm = plainView.snap()
plainFrm = overlay(plainFrm, legFrm)
plainFrm.show()

statsView = View.make(boxCam, Scene([]), annotatedPoints = [statPt], resolution = RESOLUTION_KDE)
statsFrm = statsView.snap()
fullFrm = plainFrm + statsFrm
fullFrm.show()

static_frames[player_name] = [plainFrm, fullFrm]

##################################
########### HUGO GASTON ##########
##################################
player_name = 'Hugo Gaston'

### COLLECT SHOTS FOR PLAYER
serveShots = adPnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_ad)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### MAKE KDES WITH SPLIT CATEGORIES
lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

low_kde_scene = vs.kde_hillmap(lowShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
mid_kde_scene = vs.kde_hillmap(midShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
high_kde_scene = vs.kde_hillmap(hiShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
								bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

mid_kde_scene.rastArrs[0] += [0, 60, 0]
high_kde_scene.rastArrs[0] += [0, 120, 0]

split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

loPt = origPt_3D.left(15).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
midPt = origPt_3D.back(70).left(20).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
hiPt = origPt_3D.back(150).left(25).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

######## Tests for similarity of distribution #########
diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
jntStr = vis_ut.make_stat_string(lowShots, hiShots)
statPt = diagPt.annotate(jntStr).back(110).left(40)

jntStr_comb = vis_ut.make_stat_string(lowShots, midShots.join(hiShots), lo_desc = 'Low Pressure', hi_desc = 'Mid + High Pressure')
statPt_comb = diagPt.annotate(jntStr_comb).back(110).left(40)

##### Player name and handedness ####
myLeg = Legend.make_empty(resolution = 100, header_offset = 3.25, legend_dimensions = (350, 200), header_thickness = 2)
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
l1 = {'text_string_shape': 'Player:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': player_name,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}        
l2 = {'text_string_shape': 'Handedness:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': handedness,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}
myLeg = myLeg.add(l1).add(l2)

legFrm = myLeg.render()

##### make still frames (with different elements added in)
frontView = View.make(boxCam, split_kde_scenes, annotatedPoints = [loPt, midPt, hiPt, tPt, wPt], resolution = RESOLUTION_KDE)
plainView = backView.combine(frontView)
plainFrm = plainView.snap()
plainFrm = overlay(plainFrm, legFrm)
plainFrm.show()

statsView = View.make(boxCam, Scene([]), annotatedPoints = [statPt], resolution = RESOLUTION_KDE)
statsFrm = statsView.snap()
fullFrm = plainFrm + statsFrm
fullFrm.show()

static_frames[player_name] = [plainFrm, fullFrm]

##################################
########### JOAO SOUSA ##########
##################################
player_name = 'Joao Sousa'

### COLLECT SHOTS FOR PLAYER
serveShots = adPnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_ad)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### MAKE KDES WITH SPLIT CATEGORIES
lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

low_kde_scene = vs.kde_hillmap(lowShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
mid_kde_scene = vs.kde_hillmap(midShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
high_kde_scene = vs.kde_hillmap(hiShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
								bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

mid_kde_scene.rastArrs[0] += [0, 60, 0]
high_kde_scene.rastArrs[0] += [0, 120, 0]

split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

loPt = origPt_3D.left(15).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
midPt = origPt_3D.back(70).left(20).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
hiPt = origPt_3D.back(150).left(25).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

######## Tests for similarity of distribution #########
diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
jntStr = vis_ut.make_stat_string(lowShots, hiShots)
statPt = diagPt.annotate(jntStr).back(110).left(40)

jntStr_comb = vis_ut.make_stat_string(lowShots, midShots.join(hiShots), lo_desc = 'Low Pressure', hi_desc = 'Mid + High Pressure')
statPt_comb = diagPt.annotate(jntStr_comb).back(110).left(40)

##### Player name and handedness ####
myLeg = Legend.make_empty(resolution = 100, header_offset = 3.25, legend_dimensions = (350, 200), header_thickness = 2)
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
l1 = {'text_string_shape': 'Player:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': player_name,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}        
l2 = {'text_string_shape': 'Handedness:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': handedness,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}
myLeg = myLeg.add(l1).add(l2)

legFrm = myLeg.render()

##### make still frames (with different elements added in)
frontView = View.make(boxCam, split_kde_scenes, annotatedPoints = [loPt, midPt, hiPt, tPt, wPt], resolution = RESOLUTION_KDE)
plainView = backView.combine(frontView)
plainFrm = plainView.snap()
plainFrm = overlay(plainFrm, legFrm)
plainFrm.show()

statsView = View.make(boxCam, Scene([]), annotatedPoints = [statPt], resolution = RESOLUTION_KDE)
statsFrm = statsView.snap()
fullFrm = plainFrm + statsFrm
fullFrm.show()

static_frames[player_name] = [plainFrm, fullFrm]

##################################
######### CARLOS TABERNER ########
##################################
player_name = 'Carlos Taberner'

### COLLECT SHOTS FOR PLAYER
serveShots = adPnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_ad)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### MAKE KDES WITH SPLIT CATEGORIES
lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

low_kde_scene = vs.kde_hillmap(lowShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
mid_kde_scene = vs.kde_hillmap(midShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
high_kde_scene = vs.kde_hillmap(hiShots, showReg = adlineReg, position_func = vis_ut.find_intersection_with_service_line, 
								bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

mid_kde_scene.rastArrs[0] += [0, 60, 0]
high_kde_scene.rastArrs[0] += [0, 120, 0]

split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

loPt = origPt_3D.left(15).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
midPt = origPt_3D.back(70).left(20).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
hiPt = origPt_3D.back(150).left(25).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

######## Tests for similarity of distribution #########
diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
jntStr = vis_ut.make_stat_string(lowShots, hiShots)
statPt = diagPt.annotate(jntStr).back(110).left(40)

jntStr_comb = vis_ut.make_stat_string(lowShots, midShots.join(hiShots), lo_desc = 'Low Pressure', hi_desc = 'Mid + High Pressure')
statPt_comb = diagPt.annotate(jntStr_comb).back(110).left(40)

##### Player name and handedness ####
myLeg = Legend.make_empty(resolution = 100, header_offset = 3.25, legend_dimensions = (350, 200), header_thickness = 2)
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
l1 = {'text_string_shape': 'Player:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': player_name,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}        
l2 = {'text_string_shape': 'Handedness:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': handedness,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}
myLeg = myLeg.add(l1).add(l2)

legFrm = myLeg.render()

##### make still frames (with different elements added in)
frontView = View.make(boxCam, split_kde_scenes, annotatedPoints = [loPt, midPt, hiPt, tPt, wPt], resolution = RESOLUTION_KDE)
plainView = backView.combine(frontView)
plainFrm = plainView.snap()
plainFrm = overlay(plainFrm, legFrm)
plainFrm.show()

statsView = View.make(boxCam, Scene([]), annotatedPoints = [statPt], resolution = RESOLUTION_KDE)
statsFrm = statsView.snap()
fullFrm = plainFrm + statsFrm
fullFrm.show()

static_frames[player_name] = [plainFrm, fullFrm]

#############################################
######## FROM STATIC FRAME TO FRAMES ########
#############################################

static_frames.keys()

all_frames = []
script_lines = []

## HUGO GASTON ##
frm1, frm2 = static_frames['Hugo Gaston']
frm1.show()

script_line = 'Given the previous caveats on sample size, it seems that left-handed Hugo Gaston indeed employs the serve out wide almost exclusively in points that matter. In low pressure situations, he mixes in a serve down the middle, possibly to establish a measure of unpredictability.'
myFrms = BGRFrames.from_frame(frm1, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

## LIAM BROADY ##
frm1, frm2 = static_frames['Liam Broady']
frm1.show()

script_line = 'For Liam Broady, we observe a similar picture, though the dependance on pressure is less pronounced.'
myFrms = BGRFrames.from_frame(frm1, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

## JOAO SOUSA ##
frm1, frm2 = static_frames['Joao Sousa']
frm1.show()

script_line = 'Do note that this behaviour is not limited to left-handed servers, however. A right-handed server can still deliver a flat serve out wide, though without the benefit of slice pulling the ball away from the receiver.'
myFrms = BGRFrames.from_frame(frm1, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'For example, Joao Sousa relies on the flat serve out wide on the ad side, especially in situations of great importance. If the point matters less, he delivers a serve down the middle about as often as out wide.'
myFrms = BGRFrames.from_frame(frm1, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Again, this might be to keep the receiver guessing, especially on later ad-side returns, which might bear more importance.'
myFrms = BGRFrames.from_frame(frm1, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Since for Joao Sousa we have four matches visually recognized by ATMAN (though they are all played against right-handers). it is easier to establish a statistically significant difference between serve placement in High Pressure and Low Pressure situations.'
myFrms = BGRFrames.from_frame(frm2, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

### CARLOS TABERNER ###
frm1, frm2 = static_frames['Carlos Taberner']
frm2.show()

script_line = 'On the other hand, Carlos Taberner exclusively serves down the middle in medium or high pressure situations. whereas he mixes his serves in low pressure situations.'
myFrms = BGRFrames.from_frame(frm1, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'However, Carlos Taberners only match in our sample came against left-handed Kenny De Schepper. so it might be that he is simply targetting the backhand on serve. and results might not generalize to right-handed receivers.'
myFrms = BGRFrames.from_frame(frm1, 100)
all_frames.append(myFrms)
script_lines.append(script_line)

jntFrms = BGRFrames.from_frames(all_frames)
DecapStream(jntFrms, 0, 100).show(ms_between_frames = 25)

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

    
    