
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


import ATMANpy.auxil.constants as const
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Point, read_in_all_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.Visual3D import DummyScene

from statsmodels.discrete.discrete_model import Logit

#############################
#### READ IN CONFIGS ########
#############################
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import collect_video_frames
import ATMANpy.scripts.visualization_brest.return_chip_topspin.utils as vis_ut

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

###################################
######### READ IN DATA ############
###################################
allTable = read_in_all_tables(confs)
allPnts = Points.from_datatable(allTable, split_by_match = False)
firstPnts = allPnts.filter('is_first_serve')

### Remove aces, faults and net balls.
def get_returned_balls(myPnt):
    return len(myPnt) > 1
all_retPnts = firstPnts.filter(get_returned_balls)

############################################
####### FILTER OUT NON-DECISIVE POINTS #####
############################################
# store the winner codes, so we don't have to rerun this all the time
codes_path = os.path.join(get_folder_struct('ATMANpy')['scripts'], 'visualization_brest', 'return_chip_topspin')
codes_name = 'codes.txt'
codes_file = os.path.join(codes_path, codes_name)

if codes_name in os.listdir(codes_path):
    all_winSer = pd.read_csv(codes_file, index_col = 0).squeeze()
else:
    winner_codes = vis_ut.get_winners_from_score(all_retPnts, allPnts)
    all_winSer = pd.Series(winner_codes, name = 'winner_code')
    all_winSer.to_csv(codes_file)

dec_retPnts = Points([all_retPnts.pointList[idx] for idx in range(len(all_retPnts)) if all_winSer[idx] != 0])
dec_winSer = all_winSer[all_winSer != 0].reset_index(drop = True)

############################################
####### FILTER OUT PATHOLOGICAL BALLS ######
############################################
# a) serve speed too low!
# b) no bounce recorded!
# c) too many frames!
THRESHOLD_SPEED = 140
THRESHOLD_FRAMES = 35

retPnts = dec_retPnts.filter(vis_ut.is_not_pathological, threshold_speed = THRESHOLD_SPEED, threshold_frames = THRESHOLD_FRAMES)
pathPnts = dec_retPnts.filter(vis_ut.is_pathological, threshold_speed = THRESHOLD_SPEED, threshold_frames = THRESHOLD_FRAMES)

pathSer = pd.Series(dec_retPnts.apply(vis_ut.is_pathological))
winSer = dec_winSer[pathSer != 1].reset_index(drop = True)

##### ENFORCE DIRECTION
change_direction_array = retPnts.apply('is_top_to_bot', dtype = bool) ### for convenience, store whether we'll change the direction of the point.
indices = [(pt.get_match_index(), pt.get_point_index()) for pt in retPnts]
idxFrm = pd.DataFrame(indices, columns = ['match_idx', 'point_idx'])
idxFrm['is_top_to_bot'] = change_direction_array

retPnts = retPnts.enforce_direction('Top') ### Server is at the Bottom, Receiver at the Top. UPDATE: HAS BEEN SWITCHED!
indices_new = [(pt.get_match_index(), pt.get_point_index()) for pt in retPnts]
idxFrm_new = pd.DataFrame(indices, columns = ['match_idx', 'point_idx'])
idxFrm_new = idxFrm_new.merge(idxFrm, on = ['match_idx', 'point_idx'])

######################################
### PROBABILITY TO CHIP ESTIMATION ###
######################################
### Have to filter out chipped returns on the stretch -> compare only returns which are chipped by choice!
### But then, on the other hand -> if we try to match them to topspin returns, we won't find any, since on the stretch, most returns are blocked!
### -> automatically reduced!!

### Logistic Regression 
varDict = vis_ut.get_psm_var_dict(retPnts)
y = varDict.pop('is_chipped')
X = np.stack(list(varDict.values()), axis = 1)

xFrm = pd.DataFrame(X, columns = varDict.keys())
xFrm['const'] = 1
ySer = pd.Series(y, name = 'is_chipped').astype(bool)

logReg = Logit(ySer, xFrm)
logReg = logReg.fit()
logReg.summary()

#########################################
##### CREATE REGRESSORS AND WEIGHTS #####
#########################################
regFrm = vis_ut.make_regressor_frame(retPnts, logReg, winSer, idxFrm_new)

#### From here on out, we look at returns made
# -> filter!
in_col = 'return_is_fine' #'did_bounce_in'
retPnts_in = retPnts.filter(regFrm[in_col].values)
regFrm_in = regFrm[regFrm[in_col]]

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'descriptive'
SAVE_FOLDER = 'return_chip_topspin'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

########################################
############# PARAMETERS ###############
########################################

RESOLUTION_MODIFIER = 0.5 # 2
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

STATIC_FRAMES = 100 ### does not matter much once we record: the static frames will be fit to the narration
TRANSITION_FRAMES = 25 

INVISIBLE_ALPHA = 0.05 ### which alpha is accepted as invisible
FULL_ALPHA = 1 ### which alpha is accepted as fully visible --> FULL_ALPHA as parameter not needed except for FADE_IN, FADE_OUT_FACTOR calculation

FADE_IN_FACTOR = (FULL_ALPHA / INVISIBLE_ALPHA) ** (1 / TRANSITION_FRAMES)
FADE_OUT_FACTOR = 1 / FADE_IN_FACTOR ### easy to show that this must hold.

BALL_COLOR = const.BGR['ball_yellow']
CHIP_COLOR = const.BGR['marine_blue']
TOPS_COLOR = const.BGR['light_brown']

BANDWIDTH = 45

########################################
###### LOOK AT RETURNS MADE ############
########################################
# -> idea: 
    # -> take a video serve and show the chipped return, as an example
    # -> as the chipped return flies, the video interpolates to virtual space, to show the trajectory, and where the ball lands.
    # (or maybe: superimpose the ball feature (with spline?) on the real video, then freeze, then fade to virtual space)
    # as soon as we are in virtual space, mark out where the ball lands.
    # then, show many more marks, and fade into heatmap
    ### -> explain heatmaps, yada yada yada
    # then, back to trajectry arc, and move camera to side of the returning player.
    ### -> launch angles!

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)

from ATMANpy.types.Visual3D import Camera, Scene, View
from ATMANpy.types.Streams import StreamList
from ATMANpy.types.Features import SFList

### Shots
dirFrm = regFrm[(regFrm['is_top_to_bot'])]
low_pred_exFrm = dirFrm[dirFrm.is_chipped].sort_values('pred_is_chipped').iloc[:8]
idx = int(low_pred_exFrm.index[0])
myShots = retPnts[idx][:2]

retShots = retPnts_in.get_returns()
retShots_rectified = vis_ut.rectify_shots(retShots)
chip_retShots_rectified = retShots_rectified.filter('is_slice_visual')
tops_retShots_rectified = retShots_rectified.filter('is_slice_visual', negate = True)

### Real Stream
myDtm = myShots.stream(confs = confs).decap()
scaledFrms = myDtm.frames.rescale(3)

### Cams
brestCam = Camera.calibrated_brest_behind()
brestCam_resized = brestCam.resize(3)

from ATMANpy.auxil.court import cnt_2D, top_cnt_2D
from ATMANpy.types.Shapes3D import Point3D
p_orig, p_target = Point3D.from_point2D(cnt_2D, z = 100), Point3D.from_point2D(top_cnt_2D, z = 0)
heatCam = Camera.from_to(p_orig, p_target, f = 1000, sensor_dims = (1920, 1080)).zoom(1)

### Static Scenes
courtScn = Scene.make_court()

### Legends
from ATMANpy.types.Overlays import Legend, LegendRow, overlay
legRows = [LegendRow('Topspin', const.BGR['light_brown'], text_size = 2),
           LegendRow('Chip', const.BGR['marine_blue'], text_size = 2)]
myLeg_chip_topspin = Legend(legRows, header_text = '', header_offset = 2, header_thickness = 1,
                            legend_dimensions = (300, 200), resolution = 100)

legFrm = myLeg_chip_topspin.render()

# ballSF = myShots.feat_ball(which_endpoint = 'penultimate')
# ballSF = myShots.feat_spline(which_endpoint = 'penultimate')
# topSF = myShots.feat_top_player()
# botSF = myShots.feat_bot_player()
# jntSF = ballSF.merge(topSF.rename('Top')).merge(botSF.rename('Bot'))

# _, topPly = topSF.last()
# _, botPly = botSF.last()
# plyScn = Scene([topPly, botPly])

all_frames = []

####### FLIGHT SUPERIMPOSED OVER BALL
serveSF = myShots[0].feat_spline_cumul(which_endpoint = 'penultimate')
returnSF = myShots[1].feat_spline_cumul(which_endpoint = 'penultimate')

serve_scenes = [Scene([spLn.recolor(BALL_COLOR) for spLn in splines if spLn is not None]) for splines in serveSF.values()]
return_scenes = [Scene([spLn.recolor(BALL_COLOR) for spLn in splines if spLn is not None]) for splines in returnSF.values()]
spline_scenes = [*serve_scenes, *return_scenes]

myFrms, animDict = vs.animate_sequence(spline_scenes, start_cam = brestCam_resized, stop_cam = brestCam_resized, background_scene = None, 
                                       cumulate_argument = False, animate_over = scaledFrms, **prmDict)
all_frames.append(myFrms)

myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
all_frames.append(myFrms)

####### MORPH TRAJECTORY TO VIRTUAL 
lastFrm = myFrms[-1]
splnScn = spline_scenes[-1]
staticScn = courtScn.combine(splnScn)
virtualFrm = View.make(brestCam_resized, staticScn).snap()
myFrms = vs.interpolate_frames(lastFrm, virtualFrm, num_frames = 100)
all_frames.append(myFrms)

myFrms, animDict = vs.animate_cam_motion(splnScn, start_cam = brestCam_resized, stop_cam = heatCam, background_scene = courtScn,
                                         num_frames = TRANSITION_FRAMES, interpolate_func = vs.interpolate_camera_slerp, **prmDict)
all_frames.append(myFrms)

myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
all_frames.append(myFrms)

##### MAKE ALL BALL MARKS ####
from ATMANpy.types.ShapesTennis3D import Mark

# a) the visualized mark
bouncePt = Point3D.from_point2D(Point2D(myShots[1].get_first_bounce_position()), z = 0)
bounceMk = Mark.make(bouncePt, size = 3, color = const.BGR['ball_yellow'])

# b) the undifferentiated, unrectified marks
markScn_mono = vs.scatter_markmap(retShots, one_color = const.BGR['ball_yellow'])
markScn_mono_trans = markScn_mono.realpha(INVISIBLE_ALPHA)

# c) the undifferentiated, rectified marks
markScn_mono_rect = vs.scatter_markmap(retShots_rectified, one_color = const.BGR['ball_yellow'])

# d1) the differentiated, rectified marks
markScn_poly_rect = vs.scatter_markmap(retShots_rectified, color_func_or_arr = 'is_slice_visual', 
                                        zero_color = const.BGR['light_brown'], one_color = const.BGR['marine_blue'])

# d2) the differentiated, rectified marks, separated
chip_markScn = vs.scatter_markmap(chip_retShots_rectified, one_color = const.BGR['marine_blue'])
tops_markScn = vs.scatter_markmap(tops_retShots_rectified, one_color = const.BGR['light_brown'])

####### FADE OUT FLIGHT LINE
mark_scenes = [bounceMk for _ in range(TRANSITION_FRAMES)]
myFrms, animDict = vs.animate_sequence(mark_scenes, start_cam = heatCam, stop_cam = heatCam, cumulative_foreground_transparence = FADE_OUT_FACTOR,
                                       background_scene = courtScn, **animDict, **prmDict)
all_frames.append(myFrms)

myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
all_frames.append(myFrms)

####### FADE IN OTHER MARKS
animDict['previous_foreground_scene'] = markScn_mono_trans
myFrms, animDict = vs.animate_sequence(mark_scenes, background_scene = courtScn, cumulate_argument = False,
                                       cumulative_foreground_transparence = FADE_IN_FACTOR, start_cam = heatCam, stop_cam = heatCam,
                                       **animDict, **prmDict)
all_frames.append(myFrms)

myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
all_frames.append(myFrms)

#### Switch Over Returns to Lefties so we have forehand/backhand
itpl_markScns = vs.interpolate_scenes(markScn_mono, markScn_mono_rect, num_frames = 25)
myFrms, animDict = vs.animate_sequence(itpl_markScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, **prmDict)
all_frames.append(myFrms)

from ATMANpy.auxil.court import top_bsl_deu_2D, top_bsl_ad_2D
fhPt = Point3D.from_point2D(top_bsl_deu_2D, z = 0).front(10).annotate('Forehand')
bhPt = Point3D.from_point2D(top_bsl_ad_2D, z = 0).front(10).left(25).annotate('Backhand')

myFrms, animDict = vs.animate_static(markScn_mono_rect, static_cam = heatCam, background_scene = courtScn, annotated_points = [fhPt, bhPt], **prmDict)
all_frames.append(myFrms)

itpl_markScns = vs.interpolate_scenes(markScn_mono_rect, markScn_poly_rect, num_frames = 25)
myFrms, animDict = vs.animate_sequence(itpl_markScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, annotated_points = [fhPt, bhPt], 
                                       **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)

myFrm = overlay(myFrms[-1], legFrm)
myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
all_frames.append(myFrms)

### fade out chip marks
mark_scenes = [tops_markScn for _ in range(TRANSITION_FRAMES)]
myFrms, animDict = vs.animate_sequence(mark_scenes, start_cam = heatCam, stop_cam = heatCam, background_scene = courtScn,
                                       **animDict, **prmDict, cumulative_foreground_transparence = FADE_OUT_FACTOR)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)

myFrm = overlay(myFrms[-1], legFrm)
myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
all_frames.append(myFrms)


### Topspin Heatmaps
topsScn = vs.kde_heatmap(tops_retShots_rectified, 
                         position_func = 'get_first_bounce_position', 
                         normalize_density = True,
                         zero_color = TOPS_COLOR, one_color = TOPS_COLOR, 
                         bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)
topsScn_zero = topsScn.realpha(0)
itplScns = vs.interpolate_dummy_scenes(topsScn_zero, topsScn, STATIC_FRAMES)

myFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam,
                                       **animDict, **prmDict, cumulative_foreground_transparence = FADE_OUT_FACTOR)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)

myFrm = overlay(myFrms[-1], legFrm)
myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
all_frames.append(myFrms)

# Heatmap for Weighted, 'Choice' Chips 
chipFrm = regFrm_in[regFrm_in.is_chipped == 1]
topspin_preds = chipFrm.pred_is_topspin.values
chipScn = vs.kde_heatmap(chip_retShots_rectified, 
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = topspin_preds,
                        normalize_density = True,
                        zero_color = CHIP_COLOR, one_color = CHIP_COLOR,
                        bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

itplScns = vs.interpolate_dummy_scenes(topsScn, chipScn, STATIC_FRAMES)

myFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, annotated_points = [fhPt, bhPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)

myFrm = overlay(myFrms[-1], legFrm)
myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
all_frames.append(myFrms)

show(all_frames)

### TODO: try the contourmaps (50% of balls land within...) or even hillmaps. Contourmaps can be superimposed more easily with backhand and forehand.

### DIFFERENCE BETWEEN TOPSPIN AND CHIP PLACEMENT
topsScn_not_norm = vs.kde_heatmap(tops_retShots_rectified, 
                                    position_func = 'get_first_bounce_position', 
                                    normalize_density = False,
                                    zero_color = TOPS_COLOR, one_color = TOPS_COLOR, 
                                    bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

chipScn_not_norm = vs.kde_heatmap(chip_retShots_rectified, 
                                    position_func = 'get_first_bounce_position', 
                                    weight_func_or_arr = topspin_preds,
                                    normalize_density = False,
                                    zero_color = CHIP_COLOR, one_color = CHIP_COLOR,
                                    bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

tops_kde = topsScn_not_norm.alphaArrs[0]
chip_kde = chipScn_not_norm.alphaArrs[0]

diff_kde = tops_kde - chip_kde

rastArr = topsScn_not_norm.rastArrs[0]
bgrArr = np.where(diff_kde > 0, TOPS_COLOR, CHIP_COLOR)

diff_kde_plus, diff_kde_minus = diff_kde.clip(min = 0), -diff_kde.clip(max = 0)

diff_alpha_plus = diff_kde_plus / diff_kde_plus.max()
diff_alpha_minus = diff_kde_minus / diff_kde_minus.max()

alphaArr = np.where(diff_kde > 0, diff_alpha_plus, diff_alpha_minus)

diffScn = DummyScene([rastArr], [bgrArr], [alphaArr])

myFrms, animDict = vs.animate_static(diffScn, background_scene = courtScn, static_cam = heatCam, **prmDict, annotated_points = [fhPt, bhPt])
# myFrms[-1].show() ### BEAUTIFUL!

### LAUNCH ANGLE

# fade trajectory back in
itplScns = vs.interpolate_scenes(splnScn.realpha(0), splnScn, num_frames = 25)
myFrms, animDict = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, background_scene = courtScn, 
                                       **prmDict, **animDict, annotated_points = [fhPt, bhPt], cumulative_foreground_transparence = 0.9)
DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)

# make the lateral cam and intermediate cam over net
bPt = splnScn[0].leftPt ### origin of the shot
aPt = bPt.left(100)
sideretCam = Camera.from_to(aPt, bPt, f = 1000)

from ATMANpy.auxil.court import cnt_2D
cnt = Point3D.from_point2D(cnt_2D, z = 100)
intmCam = Camera.from_to(cnt, cnt.right(100).down(20))

# player shape
_, player = myShots[1].feat_top_player().first()

# cam motion -> made so that the motion shows also the entire trajectory!
myFrms1, animDict = vs.animate_cam_motion(splnScn, num_frames = 50,  annotated_points = [fhPt, bhPt],
                                         background_scene = courtScn, start_cam = heatCam, stop_cam = intmCam)

myFrms2, animDict =  vs.animate_cam_motion(splnScn, num_frames = 50,  annotated_points = [fhPt, bhPt],
                                         background_scene = courtScn, start_cam = intmCam, stop_cam = sideretCam)
myFrms = BGRFrames.from_frames([myFrms1, myFrms2])

DecapStream(myFrms, 0, 100).show(ms_between_frames = 50)
myFrms[-1].show()

# get topspin and chip dists, with axes
kwargDict = {'bandwidth': 2,
             'resolution': 100,
             'axis_length': 40,
             'axis_origin_point': bPt,
             'x_unit_vector': Point3D((0, -1, 0)), 
             'y_unit_vector': Point3D((0, 0, 1)),
             'normalize_density': False, 
             'stretch_density': 4.5}

weightArr = regFrm_in.loc[regFrm_in.is_chipped == 1, 'weight_topspin'].values
weightArr_preds = 1 - regFrm_in.loc[regFrm_in.is_chipped == 1, 'pred_is_chipped'].values

tops_angleScn, tops_modes = vs.kde_angle_distribution(tops_retShots_rectified, **kwargDict, color = TOPS_COLOR)
chip_angleScn, chip_modes = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict, weight_func_or_arr = weightArr_preds, color = CHIP_COLOR)

xLn, yLn = tops_angleScn.shapes[1], tops_angleScn.shapes[2]

axScn = Scene([xLn, yLn])
court_and_axScn = courtScn.combine(axScn)

myFrms, animDict = vs.animate_static(splnScn, static_cam = sideretCam, background_scene = court_and_axScn)
myFrms[-1].show()

from ATMANpy.types.Shapes3D import Circle3D
ang = myShots[1].get_launch_angle()
limit_theta = ((180 - ang) * const.ANGLE_TO_RAD, 180 * const.ANGLE_TO_RAD)
degCrc = Circle3D.from_point3D(bPt, fixed_dimension = 'x', radius = 25, limit_theta = limit_theta)

myFrms, animDict = vs.animate_static(splnScn.put(degCrc), static_cam = sideretCam, background_scene = court_and_axScn)
myFrms[-1].show()

anglePt = bPt.front(24).up(1).annotate(f'{ang:.1f} Degrees')

myFrms, animDict = vs.animate_static(splnScn.put(degCrc), static_cam = sideretCam, background_scene = court_and_axScn, annotated_points = [anglePt])
myFrms[-1].show()

from ATMANpy.types.Abstract import Shots

### build a distribution of launch angles on choice chips.
sorted_chip_retShots_rectified = chip_retShots_rectified.sort(weightArr_preds, descending = True)

acceleration_rate = 0.1
initial_frames = 25
splnScn_curr = splnScn
prev_kde = None
shotList = []
weightList = []
frms = []
idx = 0
for shot, weight in zip(sorted_chip_retShots_rectified[:25], weightArr_preds[:25]):
    if not 0 < shot.get_launch_angle() <= 75:
        continue

    ### calculate num_frames
    num_frames = max(2, int(initial_frames * (1 - acceleration_rate) ** idx)) ### need at least 2 frames to have clean interpolation

    ### build new kde
    shotList.append(shot)
    weightList.append(weight)
    kde, _ = vs.kde_angle_distribution(Shots(shotList), **kwargDict, weight_func_or_arr = np.array(weightList), 
                                       color = CHIP_COLOR)
    kde = Scene([kde[0]])

    ### if necessary: create previous KDE
    if prev_kde is None:
        els = len(kde[0].coords[0])
        x, y, z = bPt.coords
        prev_pt = Point3D((np.full(els, fill_value = x), np.full(els, fill_value = y), np.full(els, fill_value = z)), color = CHIP_COLOR)
        prev_kde = Scene([prev_pt])
    
    ### Animate the current trajectory
    anchTraj = shot.shotTraj.reanchor(bPt)
    _, splines_current = anchTraj.feat_spline_cumul().last()
    splnScn_curr = Scene([spLn.recolor(BALL_COLOR) for spLn in splines_current if spLn is not None])

    myFrms, _ = vs.animate_static(splnScn_curr, background_scene = court_and_axScn, previous_foreground_scene = prev_kde, **prmDict,
                                  static_cam = sideretCam, num_frames = num_frames)
    frms.append(myFrms)

    ### Animate the new KDE
    itpl_kdes = vs.interpolate_scenes(prev_kde, kde, num_frames)
    myFrms, _ = vs.animate_sequence(itpl_kdes, background_scene = court_and_axScn, previous_foreground_scene = splnScn_curr, **prmDict,
                                    start_cam = sideretCam, stop_cam = sideretCam, cumulative_foreground_transparence = 1)
    frms.append(myFrms)

    prev_kde = kde
    idx += 1

jntFrms = BGRFrames.from_frames(frms)
DecapStream(jntFrms, 0, 100).show()

### Show entire distribution
myFrms, _ = vs.animate_static(chip_angleScn, background_scene = court_and_axScn, **prmDict,
                                static_cam = sideretCam, num_frames = num_frames)
myFrms[-1].show()

myFrms, _ = vs.animate_static(tops_angleScn, background_scene = court_and_axScn, **prmDict,
                                static_cam = sideretCam, num_frames = num_frames)
myFrms[-1].show()


shot0 = myShots[1]
kde1, _ = vs.kde_angle_distribution(Shots([shot0]), **kwargDict)

myFrms, animDict = vs.animate_static(kde1, static_cam = sideretCam, previous_foreground_scene = axScn, background_scene = courtScn)
myFrms[-1].show()

shot1 = sorted_chip_retShots_rectified[1]
traj1 = shot1.shotTraj.reanchor(bPt)

_, my_splines = traj1.feat_spline_cumul().last()
trajScn = Scene([spLn.recolor(BALL_COLOR) for spLn in my_splines if spLn is not None])

myFrms, animDict = vs.animate_static(trajScn, static_cam = sideretCam, **animDict, background_scene = courtScn)
myFrms[-1].show()



myFrms[-1].show()
myShots[1]



self.shotList[*list(np.argsort(weightArr_preds))]

sorted(self.shotList, key = weightArr_preds)


bPt_up = bPt.up(100)
bPt_in = bPt.front(100)

from ATMANpy.types.Shapes3D import Line3D
xLn, yLn = Line3D(bPt, bPt_up), Line3D(bPt, bPt_in)
axScn = itplScns[-1].put(xLn).put(yLn)

myFrms, animDict = vs.animate_static(axScn, static_cam = sideretCam, background_scene = courtScn)
myFrms[-1].show()


# launch KDEs
kwargDict = {'bandwidth': 2.5,
             'resolution': 10,
             'axis_length': 100,
             'axis_origin_point': bPt,
             'x_unit_vector': Point3D((0, -1, 0)), 
             'y_unit_vector': Point3D((0, 0, 1))}

specDicts = [{'value_func': None},
             {'value_func': 'get_launch_angle'},
             {'value_func': 'get_launch_angle', 'weight_func_or_arr': weightArr},
             {'value_func': 'get_launch_angle', 'weight_func_or_arr': weightArr_preds},
                ]
backView = View.make(sideretCam, courtScn)

#### TOPSPIN SHOTS IN PLAY
angleScn, _ = vs.kde_angle_distribution(tops_retShots_rectified, **kwargDict)
angleView = View.make(sideretCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()


##### CHIPPED SHOTS IN PLAY
for specDict in specDicts:
    angleScn, _ = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict, **specDict)
    angleView = View.make(sideretCam, angleScn)
    angleFrm = backView.combine(angleView).snap()
    angleFrm.show()
