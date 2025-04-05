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
from ATMANpy.types.Overlays import FramesRaster

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from ATMANpy.auxil.court import courtLines
from ATMANpy.auxil.court import onto_service_box_cam

import ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils as vis_ut

from ATMANpy.funcs.util_funcs import get_folder_struct

### PARAMETERS
SIZE = 3
RATIO = 3

### RASTER FUNCS

def make_simul(frms_list):
    ### MAKE SIMULTANEOUS RASTER
    myRst = FramesRaster(SIZE, SIZE)
    for idx, frms in enumerate(frms_list):
        div, rem = idx % SIZE, idx // SIZE
        frms = frms.rescale(RATIO / SIZE)
        myRst.add(frms, div, rem)

    return myRst

def make_cumul(frms_list):
    ### MAKE CUMULATIVE RASTERS
    cum_rasters = []
    from copy import deepcopy
    cumRst = FramesRaster(SIZE, SIZE)
    for idx, frms in enumerate(frms_list):
        div, rem = idx // SIZE, idx % SIZE
        frms = frms.rescale(RATIO / SIZE)
        cumRst.add(frms, div, rem, set_size = True)
        cum_rasters.append(deepcopy(cumRst))
    return cum_rasters

def render_rasters(raster_list, slow_by = 2):
    all_frames = []
    for rst in raster_list:
        myFrms = rst.render()
        all_frames.append(myFrms)
    
    jntFrms = BGRFrames.from_frames(all_frames)
    jntFrms = jntFrms.slow(slow_by)
    return jntFrms

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
DESCRIPTOR = 'deuce_side_static'
SAVE_FOLDER = 'serve_pressure'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

### READ IN DATA
allTable = read_in_all_tables(confs)
allPnts = Points.from_datatable(allTable, split_by_match = False)
firstPnts = allPnts.filter('is_first_serve').enforce_direction('Bot')
deucePnts = firstPnts.filter('is_serve_deuce')

###########################
###### HUGO GASTON ########
###########################
player_name = 'Hugo Gaston'

### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} deuce serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)
hiShots = hiShots.sort('get_total_match_score_pressure', descending = True)[:SIZE ** 2]

frms_list = vis_ut.collect_video_frames(hiShots, confs, widen_by = 3)

# ### MAKE SIMULTANEOUS RASTER
# myRst = make_simul(frms_list)
# simFrms = render_rasters([myRst])
# DecapStream(simFrms, 0, 100).show(ms_between_frames = 25)

### MAKE CUMULATIVE RASTERS
rasters = make_cumul(frms_list)
cumFrms = render_rasters(rasters)
DecapStream(cumFrms, 0, 100).show(ms_between_frames = 25)

###########################
###### LIAM BROADY ########
###########################
player_name = 'Liam Broady'

### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} deuce serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)
hiShots = hiShots.sort('get_total_match_score_pressure', descending = True)[:SIZE ** 2]

frms_list = vis_ut.collect_video_frames(hiShots, confs, widen_by = 3)

# ### MAKE SIMULTANEOUS RASTER
# myRst = make_simul(frms_list)
# simFrms = render_rasters([myRst])
# DecapStream(simFrms, 0, 100).show(ms_between_frames = 25)

### MAKE CUMULATIVE RASTERS
rasters = make_cumul(frms_list)
cumFrms = render_rasters(rasters)
DecapStream(cumFrms, 0, 100).show(ms_between_frames = 25)

###########################
### KENNY DE SCHEPPER #####
###########################
player_name = 'Kenny De Schepper'

### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} deuce serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)
hiShots = hiShots.sort('get_total_match_score_pressure', descending = True)[:SIZE ** 2]

frms_list = vis_ut.collect_video_frames(hiShots, confs, widen_by = 3)

# ### MAKE SIMULTANEOUS RASTER
# myRst = make_simul(frms_list)
# simFrms = render_rasters([myRst])
# DecapStream(simFrms, 0, 100).show(ms_between_frames = 25)

### MAKE CUMULATIVE RASTERS
rasters = make_cumul(frms_list)
cumFrms = render_rasters(rasters)
DecapStream(cumFrms, 0, 100).show(ms_between_frames = 25)

###########################
##### FEDERICO GAIO #######
###########################
player_name = 'Federico Gaio'

### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} deuce serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)
hiShots = hiShots.sort('get_total_match_score_pressure', descending = True)[:SIZE ** 2]

frms_list = vis_ut.collect_video_frames(hiShots, confs, widen_by = 3)

# ### MAKE SIMULTANEOUS RASTER
# myRst = make_simul(frms_list)
# simFrms = render_rasters([myRst])
# DecapStream(simFrms, 0, 100).show(ms_between_frames = 25)

### MAKE CUMULATIVE RASTERS
rasters = make_cumul(frms_list)
cumFrms = render_rasters(rasters)
DecapStream(cumFrms, 0, 100).show(ms_between_frames = 25)



