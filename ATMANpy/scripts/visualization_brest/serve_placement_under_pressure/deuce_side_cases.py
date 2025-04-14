
import os
import numpy as np
import pandas as pd
        
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame, MaskFrame
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Visual3D import Camera, View, Scene, DummyScene
from ATMANpy.types.Overlays import FramesRaster, overlay_center, overlay
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D, Point3D

from ATMANpy.types.Abstract import NullPoint, Points, Shots, read_in_data_tables, read_in_placeholder_tables, read_in_colinear_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const

import ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils as vis_ut


########################################
########### SAVE FOLDER ################
########################################

SAVE = True

DESCRIPTOR = 'deuce_side_cases'
SAVE_FOLDER = 'serve_pressure_new'
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

dataPnts = Points.from_datatable(replTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

firstPnts = dataPnts.filter('is_first_serve').enforce_direction('Top')
deucePnts = firstPnts.filter('is_serve_deuce')

### PARAMETERS
RESOLUTION_MODIFIER = 1 # 2
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

## camera
from ATMANpy.auxil.court import onto_service_box_cam
from ATMANpy.auxil.court import back_view_asymmetric_cam
from ATMANpy.types.ShapesTennis3D import Court

serve_from = 'Deuce'

boxCam = onto_service_box_cam(which = serve_from, side = 'Bot', dist = 150).tilt(0.15)

##### SIDE CALCULATATION ######
# How much more important is a point in the high pressure bucket on average than a point in the low pressure bucket?
hiShots = firstPnts.get_serves().filter('get_total_match_score_pressure', is_above = 0.08)
loShots = firstPnts.get_serves().filter('get_total_match_score_pressure', is_below = 0.04)

hi_avg = hiShots.get_total_match_score_pressure().mean()
lo_avg = loShots.get_total_match_score_pressure().mean()
ratio = hi_avg / lo_avg

def server_is(somePnt, player_name):
    return somePnt.get_serve().get_hit_player_name() == player_name

kde_kwargs = {'position_func': vis_ut.find_intersection_with_service_line, 'normalize_density': True, 
              'mix_colors': False, 'mix_alpha': False, 
              'resolution': RESOLUTION_KDE, 'bandwidth': BANDWIDTH_KDE}

sceneDict = {}

##################################
########## HOLGER RUNE ###########
##################################
player_name = 'Holger Rune'

### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.filter(server_is, player_name = player_name).get_serves()
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce, side = 'Bottom')
serveShots_net = serveShots.filter('is_in_net')
serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness, serve_from)
tlPt_leg = Point2D((50, 50))

splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, serve_from, **kde_kwargs)
_, _, annPnts_join = vis_ut.split_kde(serveShots_sample, 'Deuce', join_mid_and_high = True, **kde_kwargs)

sceneDict[player_name] = splitScn, xyScn, annPnts_join, legFrm

##################################
########### HUGO GASTON ##########
##################################
player_name = 'Hugo Gaston'
### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.filter(server_is, player_name = player_name).get_serves()
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce, side = 'Bottom')
serveShots_net = serveShots.filter('is_in_net')
serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness, serve_from)
tlPt_leg = Point2D((50, 50))

splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, serve_from, **kde_kwargs)

sceneDict[player_name] = splitScn, xyScn, annPnts, legFrm

##################################
########### LIAM BROADY ##########
##################################
player_name = 'Liam Broady'
### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.filter(server_is, player_name = player_name).get_serves()
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce, side = 'Bottom')
serveShots_net = serveShots.filter('is_in_net')
serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness, serve_from)
tlPt_leg = Point2D((50, 50))

splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, serve_from, **kde_kwargs)

sceneDict[player_name] = splitScn, xyScn, annPnts, legFrm

##################################
##### KENNY DE SCHEPPER ##########
##################################
player_name = 'Kenny De Schepper'
### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.filter(server_is, player_name = player_name).get_serves()
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce, side = 'Bottom')
serveShots_net = serveShots.filter('is_in_net')
serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness, serve_from)
tlPt_leg = Point2D((50, 50))

splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, serve_from, **kde_kwargs)

sceneDict[player_name] = splitScn, xyScn, annPnts, legFrm

##################################
######### FEDERICO GAIO ##########
##################################
player_name = 'Federico Gaio'
### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.filter(server_is, player_name = player_name).get_serves()
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce, side = 'Bottom')
serveShots_net = serveShots.filter('is_in_net')
serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness, serve_from)
tlPt_leg = Point2D((50, 50))

splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, serve_from, **kde_kwargs)

sceneDict[player_name] = splitScn, xyScn, annPnts, legFrm

#### ANIMATION ###

all_frames = []
script_lines = []

script_line = "Let's check back in on our lefties. "

splitScn0, xyScn0, annPnts0, legFrm0 = sceneDict['Holger Rune']
splitScn1, xyScn1, annPnts1, legFrm1 = sceneDict['Liam Broady']

itplFrms = vis_ut.morph_split_kdes(splitScn0, xyScn0, annPnts0, legFrm0,
                                    splitScn1, xyScn1, annPnts1, legFrm1, 
                                    boxCam, num_frames = 50, tlPt_leg = tlPt_leg, **prmDict)

all_frames.append(itplFrms)
script_lines.append(script_line)

script_line =   """
                    Liam Broady seems to rely primarily on a serve down the middle; 
                    This is typically a slice serve again, curving away from the returner to the backhand;
                    however, this time, the returner is drawn into the court, and not out of it.
                """

stillFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
                                annotated_points = annPnts1[:-1], num_frames = 50,
                                **prmDict)
stillFrms = overlay(stillFrms, legFrm1, tlPt_leg)

all_frames.append(stillFrms)
script_lines.append(script_line)

# script_line =   """
#                     At low leverages, there are some out wide serves mixed in, but evidence is ambivalent
#                 """
# statFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
#                                 annotated_points = annPnts1, num_frames = 50,
#                                 **prmDict)
# statFrms = overlay(statFrms, legFrm1, tlPt_leg)
# all_frames.append(statFrms)

script_line =   """
                    For Hugo Guston, the same pattern holds, with him seeming to mix in serves out wide in low leverage situations a bit more regularly, 
                    to protect the slice serve down the middle in high leverage situations.
                """

splitScn0, xyScn0, annPnts0, legFrm0 = sceneDict['Liam Broady']
splitScn1, xyScn1, annPnts1, legFrm1 = sceneDict['Hugo Gaston']

itplFrms = vis_ut.morph_split_kdes(splitScn0, xyScn0, annPnts0[:-1], legFrm0,
                                    splitScn1, xyScn1, annPnts1, legFrm1, 
                                    boxCam, num_frames = 50, tlPt_leg = tlPt_leg, **prmDict)

stillFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
                                annotated_points = annPnts1[:-1], num_frames = 50,
                                **prmDict)
stillFrms = overlay(stillFrms, legFrm1, tlPt_leg)

jointFrms = BGRFrames.from_frames([itplFrms, stillFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

# script_line =   """
#                     (Usual caveats about low sample size).
#                 """
# statFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
#                                 annotated_points = annPnts1, num_frames = 50,
#                                 **prmDict)
# statFrms = overlay(statFrms, legFrm1, tlPt_leg)
# all_frames.append(statFrms)

script_line =   """
                    Note that not all lefty servers redominantly employ the serve down the middle. 
                    Kenny De Schepper mixes in a flat serve out wide frequently, regardless of pressure.
                """
splitScn0, xyScn0, annPnts0, legFrm0 = sceneDict['Hugo Gaston']
splitScn1, xyScn1, annPnts1, legFrm1 = sceneDict['Kenny De Schepper']

itplFrms = vis_ut.morph_split_kdes(splitScn0, xyScn0, annPnts0[:-1], legFrm0,
                                    splitScn1, xyScn1, annPnts1, legFrm1, 
                                    boxCam, num_frames = 50, tlPt_leg = tlPt_leg, **prmDict)

stillFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
                                annotated_points = annPnts1[:-1], num_frames = 50,
                                **prmDict)
stillFrms = overlay(stillFrms, legFrm1, tlPt_leg)

jointFrms = BGRFrames.from_frames([itplFrms, stillFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

# statFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
#                                 annotated_points = annPnts1, num_frames = 50,
#                                 **prmDict)
# statFrms = overlay(statFrms, legFrm1, tlPt_leg)
# all_frames.append(statFrms)

# script_line = ""

# splitScn0, xyScn0, annPnts0, legFrm0 = sceneDict['Kenny de Schepper']
# splitScn1, xyScn1, annPnts1, legFrm1 = sceneDict['Federico Gaio']

# itplFrms = vis_ut.morph_split_kdes(splitScn0, xyScn0, annPnts0[:-1], legFrm0,
#                                     splitScn1, xyScn1, annPnts1, legFrm1, 
#                                     boxCam, num_frames = 50, tlPt_leg = tlPt_leg, **prmDict)
# all_frames.append(itplFrms)

# stillFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
#                                 annotated_points = annPnts1[:-1], num_frames = 50,
#                                 **prmDict)
# stillFrms = overlay(stillFrms, legFrm1, tlPt_leg)
# all_frames.append(stillFrms)

# statFrms, _ = vs.animate_static(splitScn1, background_scene = xyScn1, static_cam = boxCam,
#                                 annotated_points = annPnts1, num_frames = 50,
#                                 **prmDict)
# statFrms = overlay(statFrms, legFrm1, tlPt_leg)
# all_frames.append(statFrms)

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

