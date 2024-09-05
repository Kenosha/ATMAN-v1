
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
staticScn = courtScn.combine(spline_scenes[-1])
virtualFrm = View.make(brestCam_resized, staticScn).snap()
myFrms = vs.interpolate_frames(lastFrm, virtualFrm, num_frames = 100)
all_frames.append(myFrms)

myFrms, animDict = vs.animate_cam_motion(spline_scenes[-1], start_cam = brestCam_resized, stop_cam = heatCam, background_scene = courtScn,
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
markScn_mono_trans = markScn_mono.transparify(0.01)

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
myFrms, animDict = vs.animate_sequence(mark_scenes, start_cam = heatCam, stop_cam = heatCam, cumulative_foreground_transparence = 0.95,
                                       background_scene = courtScn, **animDict, **prmDict)
all_frames.append(myFrms)

myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
all_frames.append(myFrms)

####### FADE IN OTHER MARKS
animDict['previous_foreground_scene'] = markScn_mono_trans
myFrms, animDict = vs.animate_sequence(mark_scenes, background_scene = courtScn, cumulate_argument = False,
                                       cumulative_foreground_transparence = 1.2, start_cam = heatCam, stop_cam = heatCam,
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
myFrms, animDict = vs.animate_sequence(itpl_markScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, annotated_points = [fhPt, bhPt])
all_frames.append(myFrms)

myFrm = overlay(myFrms[-1], legFrm)
myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
all_frames.append(myFrms)

### fade out chip marks
mark_scenes = [tops_markScn for _ in range(TRANSITION_FRAMES)]
myFrms, animDict = vs.animate_sequence(mark_scenes, start_cam = heatCam, stop_cam = heatCam, background_scene = courtScn,
                                       **animDict, cumulative_foreground_transparence = 0.95)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)

### Topspin Heatmaps
topsScn = vs.kde_heatmap(tops_retShots_rectified, one_color = TOPS_COLOR, 
                         position_func = 'get_first_bounce_position', 
                         normalize_density = True,
                         bandwidth = BANDWIDTH)
topsScn_zero = topsScn.realpha(0)
itplScns = vs.interpolate_dummy_scenes(topsScn_zero, topsScn, STATIC_FRAMES)

myFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam,
                                       **animDict, cumulative_foreground_transparence = 0.95)
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
                        one_color = CHIP_COLOR,
                        normalize_density = True,
                        bandwidth = BANDWIDTH)

itplScns = vs.interpolate_dummy_scenes(topsScn, chipScn, STATIC_FRAMES)

myFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam,
                                       **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)

myFrm = overlay(myFrms[-1], legFrm)
myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
all_frames.append(myFrms)


show(all_frames)

##################################
##### Heatmap of Returns Made ####
##################################
from ATMANpy.types.Visual3D import View, Scene, Camera
import ATMANpy.funcs.vis_funcs as vs
from ATMANpy.auxil.court import back_view_asymmetric_cam
myCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Bot')
retShots = retPnts_in.get_returns()
showReg = vis_ut.botReg.scale(1.25)
courtScn = Scene.make_court()
backView = View.make(myCam, courtScn, resolution = 1)

chipFrm = regFrm_in[regFrm_in.is_chipped == 1]
topspin_preds = chipFrm.pred_is_topspin.values
topspin_odds = chipFrm.pred_is_topspin.values / chipFrm.pred_is_chipped.values

retShots_rectified = vis_ut.rectify_shots(retShots) ### left handed server points are mirrored along the middle, so that all returns going to the (bottom) left are to the backhand

chip_retShots = retShots.filter('is_slice_visual')
tops_retShots = retShots.filter('is_slice_visual', negate = True)

chip_retShots_rectified = retShots_rectified.filter('is_slice_visual')
tops_retShots_rectified = retShots_rectified.filter('is_slice_visual', negate = True)

BANDWIDTH = 45

### Topspin Heatmaps
topsScn = vs.kde_heatmap(tops_retShots, 
                         showReg = showReg, 
                         position_func = 'get_first_bounce_position', 
                         bandwidth = BANDWIDTH)

kdeView = View.make(myCam, topsScn, resolution = 1)
topsFrm = backView.combine(kdeView).snap()
topsFrm.show()

topsScn = vs.kde_heatmap(tops_retShots_rectified, 
                         showReg = showReg, 
                         position_func = 'get_first_bounce_position', 
                         bandwidth = BANDWIDTH)

kdeView = View.make(myCam, topsScn, resolution = 1)
topsFrm_rect = backView.combine(kdeView).snap()
topsFrm_rect.show()

### Chip Heatmaps
# Heatmap for All Chips
chipScn = vs.kde_heatmap(chip_retShots,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        normalize_density = True,
                        bandwidth = BANDWIDTH)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

# Heatmap for Weighted, 'Choice' Chips 
chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = topspin_preds,
                        normalize_density = True,
                        bandwidth = BANDWIDTH)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm_rect = backView.combine(kdeView).snap()
chipFrm_rect.show()

# Heatmap for 'Forced' Chips
chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = 1 - topspin_preds,
                        zero_color = (0, 0, 0), one_color = const.BGR['intense_red'],
                        normalize_density = True,
                        bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm_rect = backView.combine(kdeView).snap()
chipFrm_rect.show()

######################################
####### COMPARING CLUSTERS ###########
######################################

# ### Contour Maps with Pseudo-Clusters
# topsScn, tops_comps = vs.kde_contourmap(tops_retShots_rectified,  showReg = showReg,
#                                         position_func = 'get_first_bounce_position', 
#                                         normalize_density = True,
#                                         bandwidth = 45,
#                                         display_at_height = False,
#                                         n_components = 1)

# kdeView = View.make(myCam, topsScn, resolution = 1)
# topsFrm = backView.combine(kdeView).snap()
# topsFrm.show()

# chipScn, chip_comps = vs.kde_contourmap(chip_retShots_rectified,  showReg = showReg,
#                                         position_func = 'get_first_bounce_position', 
#                                         weight_func_or_arr = topspin_preds,
#                                         normalize_density = True,
#                                         bandwidth = 45, 
#                                         display_at_height = False,
#                                         n_components = 2)

# kdeView = View.make(myCam, chipScn, resolution = 1)
# chipFrm = backView.combine(kdeView).snap()
# chipFrm.show()

# ellScn = vs.ellipses_from_cluster_data(chip_comps)
# ellView = View.make(myCam, ellScn, resolution = 1)
# showFrm = backView.combine(kdeView).combine(ellView).snap()
# showFrm.show()

# # Heatmap for Weighted, 'Choice' Chips 
# chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
#                         position_func = 'get_first_bounce_position', 
#                         weight_func_or_arr = topspin_preds,
#                         normalize_density = True,
#                         bandwidth = BANDWIDTH)

# kdeView = View.make(myCam, chipScn, resolution = 1)
# chipFrm_rect = backView.combine(kdeView).snap()
# chipFrm_rect.show()

# #### Cluster the Actual Data
# ellScn, cluster_data = vs.gmm_clustering_grid_search(chip_retShots_rectified, 
#                                                      weight_func_or_arr = topspin_preds, weight_sampling_factor = 5)
# ellView = View.make(myCam, ellScn, resolution = 1)
# showFrm = backView.combine(kdeView).combine(ellView).snap()
# showFrm.show()

# ellScn, cluster_data = vs.gmm_clustering_grid_search(chip_retShots_rectified, 
#                                                      weight_func_or_arr = topspin_preds, weight_sampling_factor = 5,
#                                                      take_which = 2, score_func = 'bic')
# ellView = View.make(myCam, ellScn, resolution = 1)
# showFrm = backView.combine(kdeView).combine(ellView).snap()
# showFrm.show()

################################
#### RETURN DEPTH VS WIN PROB ##
################################

WEIGHT_VARIABLE = 'weight_topspin'
ENDOG_COL = 'ret_wins_weighted'

regFrm_in['ret_dist_to_bh_sq'] = regFrm_in['ret_dist_to_bh'] ** 2
regFrm_in['ret_dist_to_bsl_sq'] = regFrm_in['ret_dist_to_bsl'] ** 2
regFrm_in['ret_dist_to_bsl_bh'] = regFrm_in['ret_dist_to_bsl'] * regFrm_in['ret_dist_to_bh']

project_results_path = get_folder_struct('results_video')['return_chip_topspin']
save_path = maybe_make_folder(project_results_path, 'regression' + make_timestamp())

### Unweighted for a Base Case
regDict = {'base': ['is_chipped']}
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = None)

### Weighted with topspin odds (weight_topspin)
regDict = {'base_odds': ['is_chipped'],
           'dist_bsl_odds': ['is_chipped', 'ret_dist_to_bsl']}
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)

### Weighted with topspin odds and len weights (weight_len_pred)
regDict = { 'base_weighted': ['is_chipped'],
            'dist_bsl_weighted': ['is_chipped', 'ret_dist_to_bsl'],
            'dist_bsl_bh_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl'],
            'dist_bsl_bh_sq_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bsl_bh'],
            'dist_bsl_bh_parab_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl',  'ret_dist_to_bsl_bh'],
            'dist_bsl_bh_sq_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq', 'ret_dist_to_bsl_bh'],
           } 
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)

### Get exog, endog data from model
regressors = ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                                        endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)
X, y = vis_ut.retrieve_X_y(glmReg_win, regressors)

### find average distance to baseline for a topspin ball
mean_topspin_dist_to_bsl = X.loc[X.is_chipped == 0, 'ret_dist_to_bsl'].mean()
mean_topspin_dist_to_bh = X.loc[X.is_chipped == 0, 'ret_dist_to_bh'].mean()
avg_topspin_pred = glmReg_win.predict([0, mean_topspin_dist_to_bsl, mean_topspin_dist_to_bh, 1])
print(avg_topspin_pred)

### Change the chipScn to reflect win probability over topspin average
chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = topspin_preds,
                        normalize_density = True,
                        bandwidth = 45)

bhPt = vis_ut.botReg['Bottom'].leftPt
rastArr = chipScn.rastArrs[0]
rel_rastArr = vis_ut.get_relative_raster_array(rastArr, bhPt)

ones = np.ones((len(rel_rastArr), 1))
regressor_array = np.concatenate([ones, rel_rastArr], axis = 1)
regressor_frame = pd.DataFrame(regressor_array, columns = regressors)
regressor_frame['const'] = 1

colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regressor_frame, relative_to = avg_topspin_pred)
chipScn.colArrs[0] = colArr
chipScn.alphaArrs[0] = (chipScn.alphaArrs[0] * alArr) ** 0.5

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()


### Get exog, endog data from model
regressors = ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                                        endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)
X, y = vis_ut.retrieve_X_y(glmReg_win, regressors)

### find average distance to baseline for a topspin ball
mean_topspin_dist_to_bsl = X.loc[X.is_chipped == 0, 'ret_dist_to_bsl'].mean()
mean_topspin_dist_to_bh = X.loc[X.is_chipped == 0, 'ret_dist_to_bh'].mean()
avg_topspin_pred = glmReg_win.predict([0, mean_topspin_dist_to_bsl, mean_topspin_dist_to_bh, 
                                       mean_topspin_dist_to_bsl ** 2, mean_topspin_dist_to_bh ** 2, 1])
print(avg_topspin_pred)

### Change the chipScn to reflect win probability over topspin average
chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = topspin_preds,
                        normalize_density = True,
                        bandwidth = 45)

bhPt = vis_ut.botReg['Bottom'].leftPt
rastArr = chipScn.rastArrs[0]
rel_rastArr = vis_ut.get_relative_raster_array(rastArr, bhPt)

ones = np.ones((len(rel_rastArr), 1))
regressor_array = np.concatenate([ones, rel_rastArr, rel_rastArr ** 2], axis = 1)
regressor_frame = pd.DataFrame(regressor_array, columns = regressors)
regressor_frame['const'] = 1

colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regressor_frame, relative_to = avg_topspin_pred)
chipScn.colArrs[0] = colArr
chipScn.alphaArrs[0] = (chipScn.alphaArrs[0] * alArr) ** 0.5

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

#### LOOK AT BOUNCES AND OUTCOMES DESCRIPTIVELY ####
colArr = (regFrm_in.loc[regFrm_in.is_chipped == 1, 'winner_code'] + 1).values
markScn = vs.scatter_markmap(chip_retShots_rectified, color_func_or_arr = colArr, 
                             zero_color = const.BGR['leafy_green'], one_color = const.BGR['intense_red'])
markFrm = View.make(myCam, markScn, resolution = 1).snap()
markFrm.show()

colArr = (regFrm_in.loc[regFrm_in.is_chipped == 0, 'winner_code'] + 1).values
markScn = vs.scatter_markmap(tops_retShots_rectified, color_func_or_arr = colArr, 
                             zero_color = const.BGR['leafy_green'], one_color = const.BGR['intense_red'])
markFrm = View.make(myCam, markScn, resolution = 1).snap()
markFrm.show()

# X, y, w = vis_ut.get_X_y_w(regFrm, [*list(varDict.keys()), 'is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True, weight_col = WEIGHT_VARIABLE)
# glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
# glmReg_win.summary()

# X, y, w = vis_ut.get_X_y_w(regFrm, ['is_fore', 'speed', 'dist', 'is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True, weight_col = WEIGHT_VARIABLE)
# glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
# glmReg_win.summary()
#### NO NEED TO CONTROL FOR SERVE -> THE ONLY THING THAT COUNTS ONCE THE RETURN IS MADE, IS THE RETURN ITSELF! (SHOT SEQUENCE IS MARKOV OF ORDER 1)

### TODO: fix outliers in ret_dist_to_bsl -> some have return distances to baseline implying a bounce inside the own court...
### TODO: find bimodality in ret_dist_to_bsl -> very short returns (preferably shallow) are uncomfortable to exploit, as the server
### TODO: look at launch angle -> are they bimodally distributed for chips? for topspin? effect on return quality?
### TODO: look at speed -> does controlling for it reduce the impact of is_chipped?

###########################################
######## A LOOK AT LAUNCH ANGLES ##########
###########################################

# import matplotlib.pyplot as plt
# import os
# fig, axes = plt.subplots(2, 1)

# pd.Series(chip_retShots_rectified.get_launch_angle()).hist(ax = axes[0])
# pd.Series(tops_retShots_rectified.get_launch_angle()).hist(ax = axes[1])

# save_where = get_folder_struct('graphs')['chip_returns']
# fig.savefig(os.path.join(save_where, 'angle.png'))
from ATMANpy.types.Shapes3D import Point3D, Line3D, Lines3D
someLn = vis_ut.courtLines['Top Baseline']
someLn3D = Line3D.from_line2D(someLn, z = 0)

kwargDict = {'bandwidth': 2.5,
             'resolution': 10,
             'axis_length': 100,
             'axis_origin_point': someLn3D.leftPt,
             'x_unit_vector': Point3D((1, 0, 0)), 
             'y_unit_vector': Point3D((0, 0, 1))}

weightArr = regFrm_in.loc[regFrm_in.is_chipped == 1, 'weight_topspin'].values
weightArr_preds = 1 - regFrm_in.loc[regFrm_in.is_chipped == 1, 'pred_is_chipped'].values
specDicts = [{'value_func': None},
             {'value_func': 'get_launch_angle'},
             {'value_func': 'get_launch_angle', 'weight_func_or_arr': weightArr},
             {'value_func': 'get_launch_angle', 'weight_func_or_arr': weightArr_preds},
                ]

##### CHIPPED SHOTS IN PLAY
for specDict in specDicts:
    angleScn, _ = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict, **specDict)
    angleView = View.make(myCam, angleScn)
    angleFrm = backView.combine(angleView).snap()
    angleFrm.show()

#### TOPSPIN SHOTS IN PLAY
angleScn, _ = vs.kde_angle_distribution(tops_retShots_rectified, **kwargDict)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

### VISUALIZE MAPPING OF LAUNCH ANGLE ####

# a) map from unit circle to angle kde
unitScn, _ = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict, **specDicts[0])
angleScn, chip_modes = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict, **specDicts[3])
itplScns = vs.interpolate_scenes(unitScn, angleScn, 25)
myFrms, _ = vs.animate_sequence(itplScns)
DecapStream(myFrms, 0, 100).show()

chip_mode_lo, chip_mode_hi = chip_modes[[0, 5]]
mode_angles = [chip_mode_lo, chip_mode_hi]
modeLns = vis_ut.make_mode_lines(mode_angles, line_len = kwargDict['axis_length'], 
                                 x_ax = kwargDict['x_unit_vector'], y_ax = kwargDict['y_unit_vector'],
                                 orig = kwargDict['axis_origin_point'])
angleScn_with_line = angleScn.put(modeLns)
angleView = View.make(myCam, angleScn_with_line)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

angleScn, tops_modes = vs.kde_angle_distribution(tops_retShots_rectified, **kwargDict)
itplScns = vs.interpolate_scenes(unitScn, angleScn, 25)
myFrms, _ = vs.animate_sequence(itplScns)
DecapStream(myFrms, 0, 100).show()

tops_mode = tops_modes[0]
mode_angles = [tops_mode]
modeLns = vis_ut.make_mode_lines(mode_angles, line_len = kwargDict['axis_length'], 
                                 x_ax = kwargDict['x_unit_vector'], y_ax = kwargDict['y_unit_vector'],
                                 orig = kwargDict['axis_origin_point'])

angleScn_with_line = angleScn.put(modeLns)
angleView = View.make(myCam, angleScn_with_line)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

# b) map from line to circle 
pass

##################################################
### DOES LAUNCH ANGLE CORRELATE WITH PLACEMENT ###
##################################################

thresh_val = chip_modes[[0, 4]].mean()

regFrm_in['launch_angle'] = retPnts_in.get_returns().apply('get_launch_angle')
weightArr_preds_hi = 1 - regFrm_in.loc[(regFrm_in.is_chipped == 1) & (regFrm_in.launch_angle > thresh_val) , 'pred_is_chipped'].values
weightArr_preds_lo = 1 - regFrm_in.loc[(regFrm_in.is_chipped == 1) & (regFrm_in.launch_angle < thresh_val) , 'pred_is_chipped'].values

hi_chip_retShots_rectified = chip_retShots_rectified.filter('get_launch_angle', is_above = thresh_val)
lo_chip_retShots_rectified = chip_retShots_rectified.filter('get_launch_angle', is_below = thresh_val)

chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
                            position_func = 'get_first_bounce_position', 
                            weight_func_or_arr = weightArr_preds,
                            normalize_density = True,
                            bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

### Chip Heatmaps, Hi Launch
chipScn_hi = vs.kde_heatmap(hi_chip_retShots_rectified,  showReg = showReg,
                            position_func = 'get_first_bounce_position', 
                            weight_func_or_arr = weightArr_preds_hi,
                            normalize_density = True,
                            bandwidth = 45)

kdeView = View.make(myCam, chipScn_hi, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

### Chip Heatmaps, Low Launch
chipScn_lo = vs.kde_heatmap(lo_chip_retShots_rectified,  showReg = showReg,
                            position_func = 'get_first_bounce_position', 
                            weight_func_or_arr = weightArr_preds_lo,
                            normalize_density = True,
                            bandwidth = 45)

kdeView = View.make(myCam, chipScn_lo, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

### Combination
alpha_delta = chipScn_hi.alphaArrs[0] - chipScn_lo.alphaArrs[0]
minimum, maximum = alpha_delta.min(), alpha_delta.max()

alpha_renorm = np.where(alpha_delta < 0, - alpha_delta / minimum, alpha_delta / maximum)
alpha_renorm = (alpha_renorm + 1 / 2)
alpha_renorm = (alpha_renorm * chipScn.alphaArrs[0]) ** 0.5

colArr = const.BGR['light_brown'] * alpha_renorm + const.BGR['strong_blue'] * (1 - alpha_renorm)

from ATMANpy.types.Visual3D import DummyScene
deltaScn = DummyScene(chipScn_lo.rastArrs, [colArr], [alpha_renorm])
kdeView = View.make(myCam, deltaScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

### TODO: a little regression, maybe?

###########################################
#### BACKHAND INTERACTED /W SERVE SIDE ####
###########################################

regFrm_in['ret_dist_to_bh_serve_fore'] = regFrm_in['ret_dist_to_bh'] * regFrm_in['serve_fore']
regFrm_in['ret_dist_to_bh_serve_back'] = regFrm_in['ret_dist_to_bh'] * (1 - regFrm_in['serve_fore'])

regFrm_in['ret_dist_to_bh_serve_fore_sq'] = regFrm_in['ret_dist_to_bh_serve_fore'] ** 2
regFrm_in['ret_dist_to_bh_serve_back_sq'] = regFrm_in['ret_dist_to_bh_serve_back'] ** 2

### Weighted with topspin odds and len weights (weight_len_pred)
regDict = { 'dist_bsl_bh_side_weighted': [  'is_chipped', 
                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                            'ret_dist_to_bsl', 
                                            'launch_angle'],
            'dist_bsl_bh_side_sq_weighted': [   'is_chipped', 
                                                'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                'ret_dist_to_bsl', 
                                                'launch_angle'],
            

            }
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name,
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)

### -> not bad! on a serve from the backhand side, the model prefers returns farther from the backhand of the server (to the forehand)
### on a serve from the forehand side, the model prefers a serve that is basically hit back at the server directly
     ### e.g. in dist_bsl_bh_side_sq_weighted: 0.273 - 2 * 0.0193 * x = 0 <-> x = 7 meters (from the backhand)
### however, on the aggregate, we still see a bias for the forehand side on the return, which is hard to explain

##########################################
### LAUNCH ANGLE IN A SPECIFICATION II ###
##########################################
regFrm_in['ret_dist_to_bsl'].min(), regFrm_in['ret_dist_to_bsl'].max()
regFrm_in['ret_dist_to_bh'].min(), regFrm_in['ret_dist_to_bh'].max()

regFrm_in['ret_dist_to_bsl_x_launch_angle'] = regFrm_in['ret_dist_to_bsl'] * regFrm_in['launch_angle']

### Weighted with topspin odds and len weights (weight_len_pred)
regDict = { 'dist_bsl_bh_launch_weighted': [    'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                'launch_angle'],
            'dist_bsl_bh_launch_inter_weighted': [  'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                    'launch_angle', 'ret_dist_to_bsl_x_launch_angle'], 
            'dist_bsl_bh_launch_inter_speed_weighted':  [   'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle',
                                                            'ret_speed', 'ret_spin'], 
            'dist_bsl_bh_parab_launch_inter_speed_weighted':  [ 'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bsl_bh',
                                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle',
                                                                'ret_speed', 'ret_spin'], 
            'dist_bsl_bh_halfsq_parab_launch_inter_speed_weighted':  [  'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                                        'ret_dist_to_bsl_bh', 'ret_dist_to_bh_sq',
                                                                        'launch_angle', 'ret_dist_to_bsl_x_launch_angle',
                                                                        'ret_speed', 'ret_spin'], 
            'dist_bsl_bh_sq_launch_inter_weighted': [   'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                        'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq',
                                                        'launch_angle', 'ret_dist_to_bsl_x_launch_angle'],
            'dist_bsl_bh_halfsq_launch_inter_weighted':  [  'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl',
                                                            'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle'
                                                            ], 
            'dist_bsl_bh_sq_launch_inter_speed_weighted':  ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                            'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle',
                                                            'ret_speed', 'ret_spin'], 
            'dist_bsl_bh_side_sq_launch_inter_weighted': [  'is_chipped', 
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl', 'ret_dist_to_bsl_x_launch_angle'
                                                            ],
          
          }
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)


###################################
## CHIP RETURN QUALITY LOCATIONS ##
###################################

regressors = regDict['dist_bsl_bh_launch_inter_weighted']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                                        endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)
X, y = vis_ut.retrieve_X_y(glmReg_win, regressors)

avg_tops_pred = glmReg_win.predict(X[X.is_chipped == 0].mean()).values[0]
avg_chip_pred = glmReg_win.predict(X[X.is_chipped == 1].mean()).values[0]
print(avg_tops_pred)
print(avg_chip_pred)

bhPt = vis_ut.botReg['Bottom'].leftPt
rastArr_2D = vs.meshgrid_region(vis_ut.botReg, resolution = 1)
num_x, num_y, num_coords = rastArr_2D.shape
rastArr_2D = rastArr_2D.reshape((num_x * num_y, num_coords))

rel_rastArr = vis_ut.get_relative_raster_array(rastArr_2D, bhPt)

ones = np.ones((len(rel_rastArr), 1))
launch_angle = chip_modes[0]
angs = launch_angle * ones
dist_x_angs = rel_rastArr[:, [1]] * launch_angle

regressor_array = np.concatenate([ones, rel_rastArr, angs, dist_x_angs], axis = 1)
regressor_frame = pd.DataFrame(regressor_array, columns = regressors)
regressor_frame['const'] = 1

colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regressor_frame, relative_to = avg_topspin_pred)
zArr = np.zeros((num_x * num_y, 1))
rastArr_3D = np.concatenate([rastArr_2D, zArr], axis = 1)
rastScn = DummyScene([rastArr_3D], [colArr], [alArr])

kdeView = View.make(myCam, rastScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

######################################
### RETURNS IN AND OUT OF POSITION ###
######################################
regFrm_in['ret_dist_to_opt_sq'] = regFrm_in['ret_dist_to_opt'] ** 2

regFrm_in['ret_dist_to_opt'].min(), regFrm_in['ret_dist_to_opt'].max()

### check out extreme positioning cases
extreme_data = regFrm_in[regFrm_in['ret_dist_to_opt'] == regFrm_in['ret_dist_to_opt'].max()]
extreme_idx = int(extreme_data.index[0])
retPnts_in[extreme_idx].stream(confs = confs).show(ms_between_frames = 100)
### -> max is very interesting: the return puts the returner way out of position, and he's lucky to win the point.
### -> min indicates that maybe we should take the position of the receiver not at the return, but a bit later
    # since he might be carries sideways by the impetus of the shot.

regFrm_in['ret_dist_to_opt_x_launch_angle'] = regFrm_in['ret_dist_to_opt'] * regFrm_in['launch_angle']

### Weighted with topspin odds and len weights (weight_len_pred)
regDict = { 'dist_opt_weighted': ['is_chipped', 'ret_dist_to_opt'],
            'dist_bsl_opt_weighted': ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_opt'],
            'dist_bsl_bh_opt_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt'],
            'dist_bsl_bh_opt_inter_weighted': [ 'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle'],
            'dist_bsl_bh_opt_inter2_weighted': [ 'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
            'dist_bsl_bh_opt_halfsq_inter2_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                'ret_dist_to_bh_sq',
                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
            'dist_bsl_bh_opt_chars_halfsq_inter2_weighted': [ 'ret_speed', 'ret_spin', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                        'ret_dist_to_bh_sq',
                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
            'dist_bsl_bh_opt_chars_parab_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                            'ret_dist_to_bsl_bh',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
            'dist_bsl_opt_inter2_weighted':  [ 'is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
           } 
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)


### Optimal positioning -> squared specification.
regDict = { 'dist_opt_sq_weighted': ['is_chipped', 'ret_dist_to_opt', 'ret_dist_to_opt_sq'],
            'dist_bsl_opt_sq_weighted': ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq'],
            'dist_bsl_bh_opt_sq_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq'],
            'dist_bsl_bh_opt_sq_inter2_weighted': [ 'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                                    'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
            }
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)


#### Interact BH with IS_CHIPPED ###############
regFrm_in['ret_dist_to_bh_x_is_chipped'] = regFrm_in['ret_dist_to_bh'] * regFrm_in['is_chipped']
regDict = {'dist_bh_inter': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bh_x_is_chipped']}
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)


################################################
### RETURNS IN AND OUT OF POSITION WITH SIDE ###
################################################
regDict = {'dist_bsl_bh_side_sq_opt_inter2_weighted':  ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                        'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                        'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                        'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
            'dist_bsl_bh_side_sq_opt_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],
            'dist_bsl_bh_side_sq_opt_speed_inter2_weighted': ['ret_speed', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_x_launch_angle'],}
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                               endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)

### --> ret_dist_to_opt_x_launch_angle > 0 means a higher launch angle gives time to get in position!!!
    # check this with a small regression --> see below: launch angle and time

##################################
##### LAUNCH ANGLE AND TIME ######
##################################
player_speed_kph = 15 ### this is taken from https://tennismash.com/2016/11/24/gig-djokovic-fastest-tennis-player-world/#:~:text=As%20many%20might%20expect%2C%20Novak,top%20speed%20of%2014.89kph.
player_speed_mps = 15 * const.KPH_TO_MPS
seconds_needed_to_recover = 0.5 ### can't immediately start running after contact
 
regressors = ['launch_angle', 'ret_speed', 'ret_spin']
y, X = regFrm_in['ret_time'], regFrm_in[regressors]
X['const'] = 1
X = X.astype(float)
olsReg_fit = sm.OLS(y, X).fit()
olsReg_fit.summary()

### margins
seconds_per_degree = olsReg_fit._results.params[0]
seconds_per_degree * player_speed_mps

### simple means
means = X.groupby(regFrm_in['is_chipped'], as_index = False).mean()
fitSer = olsReg_fit.predict(means) ### simple means
delta = fitSer[1] - fitSer[0]
delta * player_speed_mps

### means using launch angle modes
dist_to_chip_lo = np.abs(regFrm_in['launch_angle'] - chip_mode_lo)
dist_to_chip_hi = np.abs(regFrm_in['launch_angle'] - chip_mode_hi)
dist_to_tops = np.abs(regFrm_in['launch_angle'] - tops_mode)
distFrm = pd.concat([dist_to_chip_lo, dist_to_chip_hi, dist_to_tops], axis = 1)
distFrm.columns = ['chip_lo', 'chip_hi', 'tops']

weightFrm = 1 / distFrm ** 2
weightFrm['chip_lo'] = weightFrm['chip_lo'] * regFrm_in['is_chipped']
weightFrm['chip_hi'] = weightFrm['chip_hi'] * regFrm_in['is_chipped']
weightFrm['tops'] = weightFrm['tops'] * (1 - regFrm_in['is_chipped'])

weightFrm_norm = weightFrm / weightFrm.sum()

means_chip_lo = X.mul(weightFrm_norm['chip_lo'], axis = 0).sum()
means_chip_hi = X.mul(weightFrm_norm['chip_hi'], axis = 0).sum()
means_tops = X.mul(weightFrm_norm['tops'], axis = 0).sum()

weighted_mean_valFrm = pd.concat([means_chip_lo, means_chip_hi, means_tops], axis = 1)
weighted_mean_valFrm.columns = ['chip_lo', 'chip_hi', 'tops']

timeSer = olsReg_fit.predict(weighted_mean_valFrm.T)
timeSer_running = timeSer - seconds_needed_to_recover
distSer = timeSer_running * player_speed_mps

###################################
## CHIP RETURN QUALITY LOCATIONS ##
###################################

regressors = regDict['dist_bsl_bh_side_sq_opt_inter2_weighted']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                                        endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)
X, y = vis_ut.retrieve_X_y(glmReg_win, regressors)

meanFrm = X.groupby(regFrm_in['is_chipped'].values).mean()
chipSer = meanFrm.loc[True]
topsSer = meanFrm.loc[False]

avg_tops_pred = glmReg_win.predict(topsSer).values[0]
avg_chip_pred = glmReg_win.predict(chipSer).values[0]
print(avg_tops_pred)
print(avg_chip_pred)

bhPt = vis_ut.botReg['Bottom'].leftPt
rastArr_2D = vs.meshgrid_region(vis_ut.botReg, resolution = 1)
num_x, num_y, num_coords = rastArr_2D.shape
rastArr_2D = rastArr_2D.reshape((num_x * num_y, num_coords))

rel_rastArr = vis_ut.get_relative_raster_array(rastArr_2D, bhPt)

###### A) Deuce-Side, Serve out Wide
# -> TODO: technically, for the average positions, we'd have to take lefty ad serves and flip them, and then calculate positions.
relPnts = retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_deuce')
ret_player_pos = relPnts.get_returns().get_player_hit_position().mean(axis = 0)
ret_ball_pos = relPnts.get_returns().get_ball_hit_position().mean(axis = 0)[:2]
dist_in_meters = vis_ut.get_ret_dist_to_opt_with_points(rastArr_2D, ret_ball_pos, ret_player_pos)

### make a regressor for each column
N = len(rel_rastArr)
xDict = {}
xDict['is_chipped'] = np.ones(N)
xDict['ret_dist_to_bsl'] = rel_rastArr[:, 1]
xDict['ret_dist_to_opt'] = dist_in_meters
xDict['ret_dist_to_bh_serve_fore'] = rel_rastArr[:, 0] ### Deuce side is 'forehand' side
xDict['ret_dist_to_bh_serve_back'] = np.zeros(N)
xDict['ret_dist_to_bh_serve_fore_sq'] = rel_rastArr[:, 0] ** 2
xDict['ret_dist_to_bh_serve_back_sq'] = np.zeros(N)
xDict['launch_angle'] = np.ones(N) * chip_mode_lo
xDict['ret_dist_to_bsl_x_launch_angle'] = rel_rastArr[:, 1] * chip_mode_lo
xDict['ret_dist_to_opt_x_launch_angle'] = dist_in_meters * chip_mode_lo
xDict['const'] = np.ones(N)

regressor_frame = pd.DataFrame(xDict)
colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regressor_frame, relative_to = avg_tops_pred, 
                                                                  alpha_exponent = 0.5)
zArr = np.zeros((num_x * num_y, 1))
rastArr_3D = np.concatenate([rastArr_2D, zArr], axis = 1)
rastScn = DummyScene([rastArr_3D], [colArr], [alArr])

kdeView = View.make(myCam, rastScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

#### WITH SPEED ###
regressors = regDict['dist_bsl_bh_side_sq_opt_speed_inter2_weighted']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, 
                                        endog_col = ENDOG_COL, weight_col = WEIGHT_VARIABLE)
X, y = vis_ut.retrieve_X_y(glmReg_win, regressors)

meanFrm = X.groupby(regFrm_in['is_chipped'].values).mean()
chipSer = meanFrm.loc[True]
topsSer = meanFrm.loc[False]

avg_tops_pred = glmReg_win.predict(topsSer).values[0]
avg_chip_pred = glmReg_win.predict(chipSer).values[0]
print(avg_tops_pred)
print(avg_chip_pred)

bhPt = vis_ut.botReg['Bottom'].leftPt
rastArr_2D = vs.meshgrid_region(vis_ut.botReg, resolution = 1)
num_x, num_y, num_coords = rastArr_2D.shape
rastArr_2D = rastArr_2D.reshape((num_x * num_y, num_coords))

rel_rastArr = vis_ut.get_relative_raster_array(rastArr_2D, bhPt)

tops_speed, tops_angle = weighted_mean_valFrm.loc[['ret_speed', 'launch_angle'], 'tops']
chip_lo_speed, chip_lo_angle = weighted_mean_valFrm.loc[['ret_speed', 'launch_angle'], 'chip_lo']
chip_hi_speed, chip_hi_angle = weighted_mean_valFrm.loc[['ret_speed', 'launch_angle'], 'chip_hi']

### do all 8 options in a loop
# -> TODO: technically, for the average positions, we'd have to take lefty ad serves and flip them, and then calculate positions.
relPnts_options = {'deuce_wide': retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_deuce'),
                   'deuce_middle': retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_deuce'),
                   'ad_middle': retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad'),
                   'ad_wide': retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad')}
angle_options = {'lo': (chip_lo_speed, chip_lo_angle), 
                 'hi': (chip_hi_speed, chip_hi_angle)}

for rel_opt, relPnts in relPnts_options.items():
    side, width = rel_opt.split('_')
    for ang_opt, speed_and_angle in angle_options.items():
        print(rel_opt, ang_opt)
        
        ret_speed, ret_launch_angle = speed_and_angle
        
        relPnts = retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_deuce')
        ret_player_pos = relPnts.get_returns().get_player_hit_position().mean(axis = 0)
        ret_ball_pos = relPnts.get_returns().get_ball_hit_position().mean(axis = 0)[:2]
        dist_in_meters = vis_ut.get_ret_dist_to_opt_with_points(rastArr_2D, ret_ball_pos, ret_player_pos)

        ### make a regressor for each column
        N = len(rel_rastArr)
        xDict = {}
        xDict['ret_speed'] = ret_speed
        xDict['ret_dist_to_bsl'] = rel_rastArr[:, 1]
        xDict['ret_dist_to_opt'] = dist_in_meters

        if side == 'deuce': ### Deuce side is rectified 'forehand' side
            xDict['ret_dist_to_bh_serve_fore'] = rel_rastArr[:, 0] 
            xDict['ret_dist_to_bh_serve_back'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_fore_sq'] = rel_rastArr[:, 0] ** 2
            xDict['ret_dist_to_bh_serve_back_sq'] = np.zeros(N)
        else: ### Ad side is rectified 'backhand' side
            xDict['ret_dist_to_bh_serve_fore'] = np.zeros(N) 
            xDict['ret_dist_to_bh_serve_back'] = rel_rastArr[:, 0]
            xDict['ret_dist_to_bh_serve_fore_sq'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_sq'] = rel_rastArr[:, 0] ** 2

        xDict['launch_angle'] = np.ones(N) * ret_launch_angle
        xDict['ret_dist_to_bsl_x_launch_angle'] = rel_rastArr[:, 1] * ret_launch_angle
        xDict['ret_dist_to_opt_x_launch_angle'] = dist_in_meters * ret_launch_angle
        xDict['const'] = np.ones(N)

        regressor_frame = pd.DataFrame(xDict)
        colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regressor_frame, relative_to = avg_tops_pred, 
                                                                        alpha_exponent = 0.5)
        zArr = np.zeros((num_x * num_y, 1))
        rastArr_3D = np.concatenate([rastArr_2D, zArr], axis = 1)
        rastScn = DummyScene([rastArr_3D], [colArr], [alArr])

        kdeView = View.make(myCam, rastScn, resolution = 1)
        chipFrm = backView.combine(kdeView).snap()
        chipFrm.show()


##################################
##### LAUNCH ANGLE AND TIME ######
##################################
player_speed_kph = 15 ### this is taken from https://tennismash.com/2016/11/24/gig-djokovic-fastest-tennis-player-world/#:~:text=As%20many%20might%20expect%2C%20Novak,top%20speed%20of%2014.89kph.
player_speed_mps = 15 * const.KPH_TO_MPS
seconds_needed_to_recover = 0.5 ### can't immediately start running after contact
 
regressors = ['launch_angle', 'ret_speed', 'ret_spin']
y, X = regFrm_in['ret_time'], regFrm_in[regressors]
X['const'] = 1
X = X.astype(float)
olsReg_fit = sm.OLS(y, X).fit()
olsReg_fit.summary()

### margins
seconds_per_degree = olsReg_fit._results.params[0]
seconds_per_degree * player_speed_mps

### simple means
means = X.groupby(regFrm_in['is_chipped'], as_index = False).mean()
fitSer = olsReg_fit.predict(means) ### simple means
delta = fitSer[1] - fitSer[0]
delta * player_speed_mps

### means using launch angle modes
dist_to_chip_lo = np.abs(regFrm_in['launch_angle'] - chip_mode_lo)
dist_to_chip_hi = np.abs(regFrm_in['launch_angle'] - chip_mode_hi)
dist_to_tops = np.abs(regFrm_in['launch_angle'] - tops_mode)
distFrm = pd.concat([dist_to_chip_lo, dist_to_chip_hi, dist_to_tops], axis = 1)
distFrm.columns = ['chip_lo', 'chip_hi', 'tops']

weightFrm = 1 / distFrm ** 2
weightFrm['chip_lo'] = weightFrm['chip_lo'] * regFrm_in['is_chipped']
weightFrm['chip_hi'] = weightFrm['chip_hi'] * regFrm_in['is_chipped']
weightFrm['tops'] = weightFrm['tops'] * (1 - regFrm_in['is_chipped'])

weightFrm_norm = weightFrm / weightFrm.sum()

means_chip_lo = X.mul(weightFrm_norm['chip_lo'], axis = 0).sum()
means_chip_hi = X.mul(weightFrm_norm['chip_hi'], axis = 0).sum()
means_tops = X.mul(weightFrm_norm['tops'], axis = 0).sum()

weighted_mean_valFrm = pd.concat([means_chip_lo, means_chip_hi, means_tops], axis = 1)
weighted_mean_valFrm.columns = ['chip_lo', 'chip_hi', 'tops']

timeSer = olsReg_fit.predict(weighted_mean_valFrm.T)
timeSer_running = timeSer - seconds_needed_to_recover
distSer = timeSer_running * player_speed_mps


### TODO: for heatmaps: maybe use topspin_odds, but zero color is also red -> this way, the KDE does not appear so falsely sparse!

################################
####### A LOOK AT INTENTION ####
################################

# idea: chip location leans heavily to the backhand side of the server -> harder to attack!
# --> can we see this when comparing LH servers and righthanded servers?
regFrm_in['is_right_handed_server'] = retPnts_in.apply(vis_ut.server_is_right_handed)
regFrm_in['is_left_handed_server'] = retPnts_in.apply(vis_ut.server_is_left_handed)

chipPnts_right = retPnts_in.filter(vis_ut.server_is_right_handed).filter(vis_ut.ret_is_chipped)
chipPnts_left = retPnts_in.filter(vis_ut.server_is_left_handed).filter(vis_ut.ret_is_chipped)

chipShots_right = chipPnts_right.get_returns()
chipShots_left = chipPnts_left.get_returns()

chipFrm_left = regFrm_in[(regFrm_in.is_chipped == 1) & (regFrm_in.is_left_handed_server == 1)]
topspin_preds_left = 1 - chipFrm_left.pred_is_chipped.values
topspin_odds_left = (1 - chipFrm_left.pred_is_chipped.values) / chipFrm_left.pred_is_chipped.values

chipFrm_right = regFrm_in[(regFrm_in.is_chipped == 1) & (regFrm_in.is_right_handed_server == 1)]
topspin_preds_right = 1 - chipFrm_right.pred_is_chipped.values
topspin_odds_right = (1 - chipFrm_right.pred_is_chipped.values) / chipFrm_right.pred_is_chipped.values

chipScn = vs.kde_heatmap(chipShots_right,  showReg = showReg,
                                  position_func = 'get_first_bounce_position', 
                                  weight_func_or_arr = topspin_preds_right,
                                  normalize_density = True,
                                  bandwidth = BANDWIDTH)

kdeView = View.make(myCam, chipScn, resolution = 1)
rightFrm = backView.combine(kdeView).snap()
rightFrm.show()

chipScn = vs.kde_heatmap(chipShots_left,  showReg = showReg,
                                  position_func = 'get_first_bounce_position', 
                                  weight_func_or_arr = topspin_preds_left,
                                  normalize_density = True,
                                  bandwidth = BANDWIDTH)

kdeView = View.make(myCam, chipScn, resolution = 1)
leftFrm = backView.combine(kdeView).snap()
leftFrm.show()

########################################
#### return dependence on serve side ###
########################################

adPnts = retPnts_in.filter('is_serve_ad')
deucePnts = retPnts_in.filter('is_serve_deuce')

regFrm_in['is_serve_ad'] = retPnts_in.apply('is_serve_ad')
regFrm_in['is_serve_deuce'] = retPnts_in.apply('is_serve_deuce')

chipShots_ad = adPnts.filter(ret_is_chipped).get_returns()
chipShots_deuce = deucePnts.filter(ret_is_chipped).get_returns()

chipFrm_ad = regFrm_in[(regFrm_in.is_chipped == 1) & (regFrm_in.is_serve_ad == 1)]
chipFrm_deuce = regFrm_in[(regFrm_in.is_chipped == 1) & (regFrm_in.is_serve_deuce == 1)]

topspin_preds_ad = 1 - chipFrm_ad.pred_is_chipped.values
topspin_preds_deuce = 1 - chipFrm_deuce.pred_is_chipped.values

chipScn = vs.kde_heatmap(chipShots_ad,  showReg = showReg,
                                  position_func = 'get_first_bounce_position', 
                                  weight_func_or_arr = topspin_preds_ad,
                                  normalize_density = True,
                                  bandwidth = 45)

backView = View.make(myCam, courtScn, resolution = 1)
kdeView = View.make(myCam, chipScn, resolution = 1)

backView.combine(kdeView).snap().show()

chipScn = vs.kde_heatmap(chipShots_deuce,  showReg = showReg,
                                  position_func = 'get_first_bounce_position', 
                                  weight_func_or_arr = topspin_preds_deuce,
                                  normalize_density = True,
                                  bandwidth = 45)

backView = View.make(myCam, courtScn, resolution = 1)
kdeView = View.make(myCam, chipScn, resolution = 1)

backView.combine(kdeView).snap().show()




