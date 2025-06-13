
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


import ATMANpy.auxil.constants as const
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Shots, read_in_data_tables, read_in_placeholder_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.Visual3D import DummyScene
from ATMANpy.types.Shapes3D import Point3D, Line3D, Circle3D, Circles3D
from ATMANpy.types.ShapesTennis3D import Player
from ATMANpy.types.Overlays import FrameRaster, FramesRaster, Table

from statsmodels.discrete.discrete_model import Logit

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)

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
dataTable = read_in_data_tables(confs, before_y_m_d = (2025, 1, 1))
nullTable = read_in_placeholder_tables(confs, before_y_m_d = (2025, 1, 1))

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

firstPnts = dataPnts.filter('is_first_serve')

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
DECAY_RATE = 0.9
regFrm = vis_ut.make_regressor_frame(retPnts, logReg, winSer, idxFrm_new, decay_rate = DECAY_RATE)

#### From here on out, we look at returns made
# -> filter!
in_col = 'return_is_fine' #'did_bounce_in'
retPnts_in = retPnts.filter(regFrm[in_col].values)
regFrm_in = regFrm[regFrm[in_col]]

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'inference'
SAVE_FOLDER = 'return_chip_topspin'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

########################################
######## INSTRUCTIONS ##################
########################################
SAVE_REGS = True

########################################
############# PARAMETERS ###############
########################################

def make_prm_dict(resolution_modifier = 1, argument_luminosity = 1, adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None):
    resolution_background = 1 * resolution_modifier
    resolution_objects = 1 * resolution_modifier
    resolution_kde = 2 * resolution_modifier

    return {'background_resolution': resolution_background,
            'argument_resolution': resolution_objects,
            'foreground_resolution': resolution_objects,
            'dummy_resolution': resolution_kde,
            'adaptive_resolution_exponent': adaptive_resolution_exponent,
            'argument_luminosity': argument_luminosity,
            'argument_blinks_per_second': argument_blinks_per_second,
            }


RESOLUTION_MODIFIER_NORMAL = 35
RESOLUTION_MODIFIER_HEATMAP = 0.5
prmDict = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_NORMAL, argument_luminosity = 1.25, 
                        adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
prmDict_nolum = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_NORMAL, argument_luminosity = 1, 
                              adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
prmDict_heat = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_HEATMAP, argument_luminosity = 1, 
                             adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
RESOLUTION_KDE = prmDict_heat['dummy_resolution']

MIX_COLORS_KDE = True
BANDWIDTH_KDE = 35
SCENE_FADE = 0.9
CONTOUR_PROP = 0.5
EPSILON = 0.002
EMPHASIS_DIM_FACTOR = 1.5
BLINK_RATE = 1

prmDict_nolum = {**prmDict}
prmDict_nolum['argument_luminosity'] = 1

STATIC_FRAMES = 100 ### does not matter much once we record: the static frames will be fit to the narration
TRANSITION_FRAMES = 25 

INVISIBLE_ALPHA = 0.05 ### which alpha is accepted as invisible
FULL_ALPHA = 1 ### which alpha is accepted as fully visible --> FULL_ALPHA as parameter not needed except for FADE_IN, FADE_OUT_FACTOR calculation

FADE_IN_FACTOR = (FULL_ALPHA / INVISIBLE_ALPHA) ** (1 / TRANSITION_FRAMES)
FADE_OUT_FACTOR = 1 / FADE_IN_FACTOR ### easy to show that this must hold.

BALL_COLOR = const.BGR['ball_yellow']
CHIP_LO_COLOR = const.BGR['marine_blue']
CHIP_HI_COLOR = const.BGR['bright_blue']
TOPS_COLOR = const.BGR['light_brown']

FRAME_THICKNESS = 2
FRAME_COLOR = const.BGR['bright_grey']

### Cam and View
from ATMANpy.types.Visual3D import View, Scene, Camera
from ATMANpy.auxil.court import back_view_asymmetric_cam
myCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Bot')
myCam = Camera.calibrated_brest_behind().resize(3)
courtScn = Scene.make_court()

from ATMANpy.auxil.court import cnt_2D, top_cnt_2D
from ATMANpy.types.Shapes3D import Point3D
p_orig, p_target = Point3D.from_point2D(cnt_2D, z = 100), Point3D.from_point2D(top_cnt_2D, z = 20)
heatCam = Camera.from_to(p_orig, p_target, f = 1000, sensor_dims = (960, 540)).zoom(0.5)

halfCam = myCam.resize(0.5)

# annotated points
from ATMANpy.auxil.court import top_bsl_deu_2D, top_bsl_ad_2D
fhPt_annot = Point3D.from_point2D(top_bsl_deu_2D, z = 0).front(10).annotate('Forehand')
bhPt_annot = Point3D.from_point2D(top_bsl_ad_2D, z = 0).front(10).left(35).annotate('Backhand')

fhPt_annot_half = fhPt_annot.left(50).front(25)
bhPt_annot_half = bhPt_annot.left(15).front(25)

fhPt_annot_heat = fhPt_annot_half.right(10)
bhPt_annot_heat = bhPt_annot_half.right(30)

prmDict_annot = {**prmDict_nolum}
prmDict_annot['previous_annotated_points'] = [fhPt_annot_half, bhPt_annot_half]

prmDict_annot_heat = {**prmDict}
prmDict_annot_heat['previous_annotated_points'] = [fhPt_annot_heat, bhPt_annot_heat]
prmDict_annot_heat['argument_luminosity'] = 1.5

backView_half = View.make(halfCam, courtScn, resolution = 1, annotatedPoints = [fhPt_annot, bhPt_annot])
backView_heat = View.make(heatCam, courtScn, resolution = 1, annotatedPoints = [fhPt_annot_heat, bhPt_annot_heat])

# backView_half.snap().show()
# backView_heat.snap().show()

### Points over which to render
aPnt = retPnts_in.get(26, 40)
bPnt = retPnts_in.get(3, 141)
vis_ut.is_serve_down_middle(aPnt)
vis_ut.is_serve_out_wide(bPnt)
### b -> is out wide! can use it as initial example

def get_spline_frames(myPnt):
    serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').values()
    splnScns = [Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in splns if spln is not None]) for splns in serve_splines]
    realFrms = myPnt[0].stream(confs = confs).decap().frames
    myFrms, _ = vs.animate_sequence(splnScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = realFrms.rescale(3), **prmDict)
    return myFrms

def get_wide_frames(myPnt, pre_frames = 150):
    preDtm = myPnt.stream(confs = confs).widen(pre_frames).decap()
    preFrms = preDtm.frames[:pre_frames]
    preArr_dimmed = np.stack([f.dim((idx / pre_frames) ** 2).array for idx, f in enumerate(preFrms)], axis = 0)
    preFrms_dimmed = BGRFrames(preArr_dimmed).rescale(3)
    return preFrms_dimmed

def get_spline_and_player_scene(myPnt, player_color = const.BGR['smokey_grey']):
    _, serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').last()
    outScn = Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in serve_splines if spln is not None])
    
    x0, y0 = myPnt[0].get_player_hit_position()
    x1, y1 = myPnt[1].get_player_hit_position()

    srvPly = Player.make((x0, y0), color = player_color)
    recPly = Player.make((x1, y1), color = player_color)

    return outScn.put(srvPly).put(recPly) 

def is_return_topspin(myPnt):
    ### filter for topspin returns so that we can introduce counterfactual.
    return myPnt.get_return().is_topspin_visual()

realDict = {}
wideDict = {}
sceneDict = {}

pre_frames = 100

# a) deuce, out wide
myFrms = get_spline_frames(bPnt)
preFrms = get_wide_frames(bPnt, pre_frames = pre_frames)
myScn = get_spline_and_player_scene(bPnt)
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['deuce_wide'] = myFrms
wideDict['deuce_wide'] = preFrms
sceneDict['deuce_wide'] = myScn

# b) deuce, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_deuce').filter(is_return_topspin)
idx = 39
myFrms = get_spline_frames(relPnts[idx])
preFrms = get_wide_frames(relPnts[idx], pre_frames = pre_frames)
myScn = get_spline_and_player_scene(relPnts[idx])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['deuce_middle'] = myFrms
wideDict['deuce_middle'] = preFrms
sceneDict['deuce_middle'] = myScn

# c) ad, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad').filter(is_return_topspin)
idx = 40
myFrms = get_spline_frames(relPnts[idx]) ### not nakashima again.
preFrms = get_wide_frames(relPnts[idx], pre_frames = pre_frames) 
myScn = get_spline_and_player_scene(relPnts[idx])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['ad_middle'] = myFrms
wideDict['ad_middle'] = preFrms
sceneDict['ad_middle'] = myScn

# d) ad, out wide
idx = 96
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[idx])
preFrms = get_wide_frames(relPnts[idx], pre_frames = pre_frames) 
myScn = get_spline_and_player_scene(relPnts[idx])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['ad_wide'] = myFrms
wideDict['ad_wide'] = preFrms
sceneDict['ad_wide'] = myScn

### Get area of confidence for prediction
# kdeScn = vs.kde_heatmap(retPnts_in.get_returns(), showReg = vis_ut.topReg, resolution = RESOLUTION_KDE, normalize_density = True, bandwidth = BANDWIDTH_KDE)
# backView.combine(View.make(myCam, kdeScn)).snap().show()
# -> just simply use everything behind the service line.
showReg = CourtLines.make().get_deep_box(side = 'Top')

### Meshgrid
bhPt = vis_ut.topReg['Top'].rightPt
rastArr_2D = vs.meshgrid_region(showReg, resolution = RESOLUTION_KDE)

num_x, num_y, num_coords = rastArr_2D.shape
N = num_x * num_y

rastArr_2D = rastArr_2D.reshape((N, num_coords))
rel_rastArr = vis_ut.get_relative_raster_array(rastArr_2D, bhPt, x_flip = True)

zArr = np.zeros_like(rastArr_2D[:, [0]])
rastArr_3D = np.concatenate([rastArr_2D, zArr], axis = 1)
### TODO: add player positions OR bring heatmaps onto video of proper examples!

#######################
#### TARGET AREAS #####
#######################

# idea -> for each point in the showReg, assume it is used as a target. 
# then, build a kernel around it -> this is the inaccuracy that is factored in
# then, calculate how much of that kernel is within the play area, and how much is without.
# for each point, the ratio of "within area" over "kernel area" is the target quality metric.
# then, weight the win probability with the target quality metric at each point
# from this, subtract the probability of winning with an average topspin shot minus the probability of making the topspin shot 
# (the above diagrams implicitly assume that this latter probability is 1)

shareArr = vis_ut.target_quality_measure(showReg, bandwidth = 30)


shareArr_norm = (shareArr - shareArr.min()) / (shareArr.max() - shareArr.min())
checkScn = DummyScene([rastArr_3D], [np.ones_like(rastArr_3D) * 255], [shareArr_norm[:, np.newaxis]])
# View.make(myCam, checkScn).snap().show()

### SOME PARAMETERS

### adapt these with reference to descriptive
MODE_CHIP_LO = 11.75
MODE_CHIP_HI = 18.75 
MODE_TOPS = 7.28
# DIST_CHIP_LO = 6.25
# DIST_CHIP_HI = 8.08
# DIST_TOPS = 3.33

angle_filter = np.abs(regFrm_in['launch_angle'] - MODE_CHIP_LO) < np.abs(regFrm_in['launch_angle'] - MODE_CHIP_HI) 
closest_mode = np.where(~regFrm_in.is_chipped, 'tp', np.where(angle_filter, 'lo', 'hi'))

SPEED_CHIP_HI, SPEED_CHIP_LO, SPEED_TOPS = regFrm_in.groupby(closest_mode).ret_speed.median()
assert SPEED_TOPS > SPEED_CHIP_LO > SPEED_CHIP_HI

RPM_CHIP_HI, RPM_CHIP_LO, RPM_TOPS = regFrm_in.groupby(closest_mode).ret_spin.median()

modeDict = {'lo': MODE_CHIP_LO, 'hi': MODE_CHIP_HI, 'tp': MODE_TOPS}
speedDict = {'lo': SPEED_CHIP_LO, 'hi': SPEED_CHIP_HI, 'tp': SPEED_TOPS}
rpmDict = {'lo': RPM_CHIP_LO, 'hi': RPM_CHIP_HI, 'tp': RPM_TOPS}

### rendering parameters
ALPHA_EXPONENT = 0.5
RENDER_ON_REAL = True
DIM_FACTOR = 0.5

#### Make return player positions for each serve case
serve_dist_dict = {}
server_pos_dict = {}
return_pos_dict = {}

retPnts_in_left = retPnts_in.filter(vis_ut.server_is_left_handed)
retPnts_in_right = retPnts_in.filter(vis_ut.server_is_right_handed)
retPnts_in_corr = retPnts_in_right.join(retPnts_in_left.flip(across_middle = True, across_net = False))

def get_three_pos(relPnts):
    serve_player_pos = relPnts.get_serves().get_player_hit_position().mean(axis = 0)
    ret_player_pos = relPnts.get_returns().get_player_hit_position().mean(axis = 0)
    ret_ball_pos = relPnts.get_returns().get_ball_hit_position().mean(axis = 0)[:2]
    return serve_player_pos, ret_player_pos, ret_ball_pos

# a) Deuce-Side, Serve Out Wide
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_deuce')
serve_player_pos, ret_player_pos, ret_ball_pos = get_three_pos(relPnts)
dist_in_meters = vis_ut.get_ret_dist_to_opt_with_points(rastArr_2D, ret_ball_pos, ret_player_pos, server_side = 'Top')
serve_dist_dict['deuce_wide'] = dist_in_meters
server_pos_dict['deuce_wide'] = serve_player_pos
return_pos_dict['deuce_wide'] = ret_player_pos

# b) Deuce-Side, Serve Down Middle
relPnts =  retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_deuce')
serve_player_pos, ret_player_pos, ret_ball_pos = get_three_pos(relPnts)
dist_in_meters = vis_ut.get_ret_dist_to_opt_with_points(rastArr_2D, ret_ball_pos, ret_player_pos, server_side = 'Top')
serve_dist_dict['deuce_middle'] = dist_in_meters
server_pos_dict['deuce_middle'] = serve_player_pos
return_pos_dict['deuce_middle'] = ret_player_pos

# c) Ad-Side, Serve Down Middle
relPnts =  retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad')
serve_player_pos, ret_player_pos, ret_ball_pos = get_three_pos(relPnts)
dist_in_meters = vis_ut.get_ret_dist_to_opt_with_points(rastArr_2D, ret_ball_pos, ret_player_pos, server_side = 'Top')
serve_dist_dict['ad_middle'] = dist_in_meters
server_pos_dict['ad_middle'] = serve_player_pos
return_pos_dict['ad_middle'] = ret_player_pos

# d) Ad-Side, Serve out Wide
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad')
serve_player_pos, ret_player_pos, ret_ball_pos = get_three_pos(relPnts)
dist_in_meters = vis_ut.get_ret_dist_to_opt_with_points(rastArr_2D, ret_ball_pos, ret_player_pos, server_side = 'Top')
serve_dist_dict['ad_wide'] = dist_in_meters
server_pos_dict['ad_wide'] = serve_player_pos
return_pos_dict['ad_wide'] = ret_player_pos

### Construct Meshgridded regressor dataframe for each case in question
# These are always the same!
xDict = {}
xDict['const'] = np.ones(N)
xDict['ret_dist_to_bsl'] = rel_rastArr[:, 1]
xDict['ret_dist_to_bh'] = rel_rastArr[:, 0]
xDict['ret_dist_to_bsl_bh'] = xDict['ret_dist_to_bsl'] * xDict['ret_dist_to_bh']
xDict['ret_dist_to_bsl_sq'] = xDict['ret_dist_to_bsl'] ** 2
xDict['ret_dist_to_bh_sq'] = xDict['ret_dist_to_bh'] ** 2

regressors_base = pd.DataFrame(xDict)

# these depend on the launch angle
regressors_angle = {}
for chip_name in ['tp', 'lo', 'hi']:
    is_chipped = chip_name != 'tp'

    angle = modeDict[chip_name]
    speed = speedDict[chip_name]
    rpm = rpmDict[chip_name]

    xDict = {}
    
    xDict['launch_angle'] = np.ones(N) * angle
    xDict['ret_dist_to_bh_x_launch_angle'] = rel_rastArr[:, 0] * angle
    xDict['ret_dist_to_bsl_x_launch_angle'] = rel_rastArr[:, 1] * angle
    xDict['ret_dist_to_bh_sq_x_launch_angle'] = rel_rastArr[:, 0] ** 2 * angle
    xDict['ret_dist_to_bsl_sq_x_launch_angle'] = rel_rastArr[:, 1] ** 2 * angle

    xDict['is_chipped'] = np.ones(N) * is_chipped
    xDict['ret_dist_to_bh_x_is_chipped'] = rel_rastArr[:, 0] * is_chipped    
    xDict['ret_dist_to_bsl_x_is_chipped'] =  rel_rastArr[:, 1] * is_chipped
    xDict['ret_dist_to_bh_sq_x_is_chipped'] = rel_rastArr[:, 0] ** 2 * is_chipped
    xDict['ret_dist_to_bsl_sq_x_is_chipped'] = rel_rastArr[:, 1] ** 2 * is_chipped
    xDict['ret_dist_to_bsl_bh_x_is_chipped'] = rel_rastArr[:, 1] * rel_rastArr[:, 0] * is_chipped

    xDict['ret_speed'] = np.ones(N) * speed
    xDict['ret_dist_to_bh_x_ret_speed'] = rel_rastArr[:, 0] * speed    
    xDict['ret_dist_to_bsl_x_ret_speed'] =  rel_rastArr[:, 1] * speed
    xDict['ret_dist_to_bh_sq_x_ret_speed'] = rel_rastArr[:, 0] ** 2 * speed
    xDict['ret_dist_to_bsl_sq_x_ret_speed'] = rel_rastArr[:, 1] ** 2 * speed

    xDict['ret_spin'] = np.ones(N) * rpm
    xDict['ret_dist_to_bh_x_ret_spin'] = rel_rastArr[:, 0] * rpm    
    xDict['ret_dist_to_bsl_x_ret_spin'] =  rel_rastArr[:, 1] * rpm
    xDict['ret_dist_to_bh_sq_x_ret_spin'] = rel_rastArr[:, 0] ** 2 * rpm
    xDict['ret_dist_to_bsl_sq_x_ret_spin'] = rel_rastArr[:, 1] ** 2 * rpm

    regressors_angle[chip_name] = pd.DataFrame(xDict)

# these depend on the serve case
regressors_case = {}
for serve_case, dist in serve_dist_dict.items():
    side, where = serve_case.split('_')
    xDict = {}
    xDict['ret_dist_to_opt'] = dist
    xDict['ret_dist_to_opt_sq'] = dist ** 2

    if side == 'deuce': ### Deuce side is 'forehand' side
        xDict['serve_fore'] = np.ones(N)
        xDict['ret_dist_to_bh_serve_fore'] = rel_rastArr[:, 0] 
        xDict['ret_dist_to_bh_serve_back'] = np.zeros(N)
        xDict['ret_dist_to_bh_serve_fore_sq'] = rel_rastArr[:, 0] ** 2
        xDict['ret_dist_to_bh_serve_back_sq'] = np.zeros(N)
        xDict['ret_dist_to_bh_serve_fore_bsl'] = rel_rastArr[:, 0] * rel_rastArr[:, 1]
        xDict['ret_dist_to_bh_serve_back_bsl'] = np.zeros(N)

    if side == 'ad':
        xDict['serve_fore'] = np.zeros(N)
        xDict['ret_dist_to_bh_serve_fore'] = np.zeros(N) 
        xDict['ret_dist_to_bh_serve_back'] = rel_rastArr[:, 0]
        xDict['ret_dist_to_bh_serve_fore_sq'] = np.zeros(N)
        xDict['ret_dist_to_bh_serve_back_sq'] = rel_rastArr[:, 0] ** 2
        xDict['ret_dist_to_bh_serve_fore_bsl'] = np.zeros(N)
        xDict['ret_dist_to_bh_serve_back_bsl'] = rel_rastArr[:, 0] * rel_rastArr[:, 1]

    regressors_case[serve_case] = pd.DataFrame(xDict)

# these depend on serve case AND angle
regressors_angle_case = {}
for chip_name in ['tp', 'lo', 'hi']:
    is_chipped = chip_name != 'tp'

    angle = modeDict[chip_name]
    speed = speedDict[chip_name]
    rpm = rpmDict[chip_name]

    for serve_case, dist in serve_dist_dict.items():
        side, where = serve_case.split('_')
        xDict = {}
        
        xDict['ret_dist_to_opt_x_launch_angle'] = dist * angle
        xDict['ret_dist_to_opt_sq_x_launch_angle'] = (dist ** 2) * angle

        xDict['ret_dist_to_opt_x_is_chipped'] = dist * is_chipped
        xDict['ret_dist_to_opt_sq_x_is_chipped'] = dist ** 2 * is_chipped

        xDict['ret_dist_to_opt_x_ret_speed'] = dist * speed
        xDict['ret_dist_to_opt_sq_x_ret_speed'] = dist ** 2 * speed

        xDict['ret_dist_to_opt_x_ret_spin'] = dist * rpm
        xDict['ret_dist_to_opt_sq_x_ret_spin'] = dist ** 2 * rpm

        
        if side == 'deuce': ### Deuce side is 'forehand' side
            xDict['ret_dist_to_bh_serve_fore_x_launch_angle'] = rel_rastArr[:, 0] * angle
            xDict['ret_dist_to_bh_serve_back_x_launch_angle'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_fore_sq_x_launch_angle'] = rel_rastArr[:, 0] ** 2 * angle
            xDict['ret_dist_to_bh_serve_back_sq_x_launch_angle'] = np.zeros(N)
            
            xDict['ret_dist_to_bh_serve_fore_x_is_chipped'] = rel_rastArr[:, 0] * is_chipped
            xDict['ret_dist_to_bh_serve_back_x_is_chipped'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_fore_sq_x_is_chipped'] = rel_rastArr[:, 0] ** 2 * is_chipped
            xDict['ret_dist_to_bh_serve_back_sq_x_is_chipped'] = np.zeros(N) 

            xDict['ret_dist_to_bh_serve_fore_bsl_x_is_chipped'] = rel_rastArr[:, 0] * rel_rastArr[:, 1] * is_chipped
            xDict['ret_dist_to_bh_serve_back_bsl_x_is_chipped'] = np.zeros(N)
            
            xDict['ret_dist_to_bh_serve_fore_x_ret_speed'] = rel_rastArr[:, 0] * speed
            xDict['ret_dist_to_bh_serve_back_x_ret_speed'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_fore_sq_x_ret_speed'] = rel_rastArr[:, 0] ** 2 * speed
            xDict['ret_dist_to_bh_serve_back_sq_x_ret_speed'] = np.zeros(N) 

            xDict['ret_dist_to_bh_serve_fore_bsl_x_ret_speed'] = rel_rastArr[:, 0] * rel_rastArr[:, 1] * speed
            xDict['ret_dist_to_bh_serve_back_bsl_x_ret_speed'] = np.zeros(N)
            
        if side == 'ad':
            xDict['ret_dist_to_bh_serve_fore_x_launch_angle'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_x_launch_angle'] = rel_rastArr[:, 0] * angle
            xDict['ret_dist_to_bh_serve_fore_sq_x_launch_angle'] = np.zeros(N) 
            xDict['ret_dist_to_bh_serve_back_sq_x_launch_angle'] = rel_rastArr[:, 0] ** 2 * angle

            xDict['ret_dist_to_bh_serve_fore_x_is_chipped'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_x_is_chipped'] = rel_rastArr[:, 0] * is_chipped
            xDict['ret_dist_to_bh_serve_fore_sq_x_is_chipped'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_sq_x_is_chipped'] = rel_rastArr[:, 0] ** 2 * is_chipped

            xDict['ret_dist_to_bh_serve_fore_bsl_x_is_chipped'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_bsl_x_is_chipped'] = rel_rastArr[:, 0] * rel_rastArr[:, 1] * is_chipped
        
            xDict['ret_dist_to_bh_serve_fore_x_ret_speed'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_x_ret_speed'] = rel_rastArr[:, 0] * speed
            xDict['ret_dist_to_bh_serve_fore_sq_x_ret_speed'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_sq_x_ret_speed'] = rel_rastArr[:, 0] ** 2 * speed

            xDict['ret_dist_to_bh_serve_fore_bsl_x_ret_speed'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_bsl_x_ret_speed'] = rel_rastArr[:, 0] * rel_rastArr[:, 1] * speed
        

        regressors_angle_case[chip_name + '_' + serve_case] = pd.DataFrame(xDict)

def get_reg_frame(angle, side, where):
    ### make regressor frame
    regFrm_base = regressors_base
    regFrm_angle = regressors_angle[angle]
    regFrm_case = regressors_case[side + '_' + where]
    regFrm_angle_case = regressors_angle_case[angle + '_' + side + '_' + where]
    
    regressor_frame = pd.concat([regFrm_base, regFrm_angle, regFrm_case, regFrm_angle_case], axis = 1)
    regressor_frame = regressor_frame[[*regressors, 'const']]
    return regressor_frame

regDict = {'chip_inter_opt_launch_even_more': ['is_chipped', 'launch_angle', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                    'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                    'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                    'ret_dist_to_bh_serve_fore_bsl', 'ret_dist_to_bh_serve_back_bsl',
                                    'ret_dist_to_bh_serve_fore_x_is_chipped', 'ret_dist_to_bh_serve_back_x_is_chipped',
                                    'ret_dist_to_bh_serve_fore_sq_x_is_chipped', 'ret_dist_to_bh_serve_back_sq_x_is_chipped',
                                    'ret_dist_to_opt_x_is_chipped', 'ret_dist_to_opt_sq_x_is_chipped',
                                    'ret_dist_to_bh_serve_fore_bsl_x_is_chipped', 'ret_dist_to_bh_serve_back_bsl_x_is_chipped',
                                    'ret_dist_to_bsl_x_launch_angle', 
                                    'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle',
                                    'ret_dist_to_bh_serve_fore_x_launch_angle', 'ret_dist_to_bh_serve_back_x_launch_angle',
                                    'ret_dist_to_bh_serve_fore_sq_x_launch_angle', 'ret_dist_to_bh_serve_back_sq_x_launch_angle'],
        'nochip_launch_speed_rpm_inter3': ['launch_angle', 'ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                    'ret_dist_to_bh', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_bh',
                                    'ret_dist_to_bsl_x_launch_angle', 
                                    'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle',
                                    'ret_dist_to_bh_x_launch_angle', 'ret_dist_to_bh_sq_x_launch_angle',
                                    'ret_dist_to_bsl_x_ret_speed', 
                                    'ret_dist_to_opt_x_ret_speed', 'ret_dist_to_opt_sq_x_ret_speed',
                                    'ret_dist_to_bh_x_ret_speed', 'ret_dist_to_bh_sq_x_ret_speed',
                                    'ret_dist_to_bh_x_is_chipped', 'ret_dist_to_bh_sq_x_is_chipped',
                                    'ret_dist_to_bsl_x_is_chipped',
                                    'ret_dist_to_opt_x_is_chipped', 'ret_dist_to_opt_sq_x_is_chipped']}

###### ESTIMATE MODEL ######
reg_name = 'nochip_launch_speed_rpm_inter3'
regressors = regDict[reg_name]

endog = 'ret_wins_weighted'
weight = 'weight_bal_pred'

glmReg = vis_ut.make_and_save_model(regFrm_in, regressors, save_path = None, save_name = None, 
                                                endog_col = endog, weight_col = weight)
    
########## SETUP #########
myRst = FramesRaster(2, 2, borders = None)
all_frames = []
script_lines = []
dim_factor = 0.25
annot_color = const.BGR['bright_grey']
bandwidth_field = 50

postRst = FramesRaster(2, 2, borders = None)
backFrm = View.make(halfCam, courtScn).snap()

comparison_prob_dict = {'tp': 0.55, ### this way, target areas will have roughly the same size:
                        'lo': 0.55,      ### we then annotate their quality with annotation point
                        'hi': 0.5}

color_dict = {'tp': TOPS_COLOR,
              'lo': CHIP_LO_COLOR,
              'hi': CHIP_HI_COLOR}

### Legends
from ATMANpy.types.Overlays import Legend, LegendRow, overlay

from ATMANpy.funcs.util_funcs import interpolate_tuples
CHIP_COLOR = interpolate_tuples(CHIP_LO_COLOR, CHIP_HI_COLOR)

from ATMANpy.types.Overlays import Legend, LegendRow, overlay
tpRow = LegendRow('... Topspin', shape_color = TOPS_COLOR, text_size = 2)
loRow = LegendRow('... Low Chip', shape_color = CHIP_LO_COLOR, text_size = 2)
hiRow = LegendRow('... High Chip', shape_color = CHIP_HI_COLOR, text_size = 2)
cpRow = LegendRow('... (Choice) Chip', shape_color = CHIP_COLOR, text_size = 2)

leg_dims = (300, 300)
# leg_dims = (500, 500)
leg_params = {'borders': True, 'header_text': 'Return is ...', 'resolution': 200, 'render_height': 300,
              'legend_dimensions': leg_dims, 'header_offset': 2, 'header_thickness': 1}

leg1 = Legend([tpRow, LegendRow(None), LegendRow(None)], **leg_params)
leg1Frm = leg1.render()
# leg1Frm.show()

leg2 = Legend([tpRow, loRow, LegendRow(None)], **leg_params)
leg2Frm = leg2.render()
# leg2Frm.show()

leg3 = Legend([tpRow, loRow, hiRow], **leg_params)
leg3Frm = leg3.render()
# leg3Frm.show()

top_left_coords = (1920 - leg_dims[0]) // 2, (1080 - leg_dims[1]) // 2
tlPt = Point2D(top_left_coords)

def consolidate_dummy_scene(myScn):
    myScn.rastArrs
    myScn.colArrs
    myScn.alphaArrs

    premul_alpha = np.zeros_like(myScn.alphaArrs[0])
    premul_color = np.zeros_like(myScn.colArrs[0])

    for rastArr, colArr, alphaArr in zip(myScn.rastArrs, myScn.colArrs, myScn.alphaArrs):
        premul_alpha = premul_alpha * (1 - alphaArr) + alphaArr
        premul_color = premul_color * (1 - alphaArr) + colArr * alphaArr

    rastArr = myScn.rastArrs[0]

    return DummyScene([rastArr], [premul_color], [premul_alpha])

### Little Flag
from ATMANpy.types.Overlays import make_text_in
def make_flag(basePt, height_of_flag = 10, width_of_flag = 20, color = const.BGR['intense_red'], alpha = 1):
    from ATMANpy.types.Shapes3D import Region3D, Line3D
    leftPt = basePt.up(height_of_flag / 2)
    rightPt = leftPt.right(width_of_flag)

    flagLn = Line3D(leftPt, rightPt)
    return Region3D.from_line3D(flagLn, width = height_of_flag, dim = 'z', left_right = "+x", top_bottom = '+z',
                                color = color, alpha = alpha)

def put_desc(someFrms, descStr, side, height, color = const.BGR['leafy_green'], thickness = 2):
    assert side in ['left', 'right']
    assert height in ['up', 'down']
    
    if side == 'left':
        x = 50
    else:
        x = 730

    if height == 'up':
        y = 70
    else:
        y = 470
    
    where = (x, y)

    textFrms = someFrms.put(descStr, where = where, thickness = thickness, color = color)

    return textFrms

#### S+1/S+2 Table #######

GRAPH_RESOLUTION = 50

DIMENSIONS = (1920 // 3, 1080 // 3)
x1, y1 = DIMENSIONS
x2, y2 = x1 * 2, y1 * 2

jntFrm, jntFrm_decayed, shrFrm, diffFrm_chip, diffFrm_top, avgFrm = vis_ut.chip_topspin_return_avgs_by_length(regFrm, decay_rate = DECAY_RATE)

cond_shrFrm = shrFrm.loc[3:] / (1 - shrFrm.loc[2])

combFrm = avgFrm * shrFrm

chip_shareFrm = shrFrm['chip_smart']
chip_avgFrm = avgFrm['chip_smart']

shares_2_6 = chip_shareFrm.loc[2:6].values
shares_7_8 = chip_shareFrm.loc[7:8].sum()
shares_9_10 = chip_shareFrm.loc[9:10].sum()
shares_11 = chip_shareFrm.loc[11:].sum()

share_ys = [*shares_2_6, shares_7_8, shares_9_10, shares_11]

avg_2_6 = chip_avgFrm.loc[2:6].values
avg_7_8 = (chip_shareFrm * chip_avgFrm).loc[7:8].sum() / shares_7_8
avg_9_10 = (chip_shareFrm * chip_avgFrm).loc[9:10].sum() / shares_9_10
avg_11 = (chip_shareFrm * chip_avgFrm).loc[11:].sum() / shares_11

avg_ys = [*avg_2_6, avg_7_8, avg_9_10, avg_11]

shares_2_6 = chip_shareFrm.loc[2:6].values
shares_7   = chip_shareFrm.loc[7:].sum()

shr_ys = [*shares_2_6, shares_7]
shr_ys = np.array(shr_ys)

avg_2_6 = chip_avgFrm.loc[2:6].values
avg_7 = (chip_shareFrm * chip_avgFrm).loc[7:].sum() / shares_7

avg_ys = [*avg_2_6, avg_7]
avg_ys = 1 - np.array(avg_ys) ### from perspective of server

xs = [1, 2, 3, 4, 5, 6]
x_labels = ['2', '3', '4', '5', '6', '7+']

from ATMANpy.types.Overlays import LineChart, ScatterChart, Graph, Legend, LegendRow

graph_kwargs = {'color': const.BGR['bright_grey'], 'y_label_format': '>4.0%', 'y_label': None, 'offset': 0.3, 
                'make_default_x_labels': False,
                'y_label_dist': 2.5, 'dimensions': (x2, y2), 'x_label_dist': 1.15, 'x_label_left_shift': 3.75, 
                'x_tick_label_left_shift': 0.3,
                'label_size': 2, 'tick_label_size': 2, 
                'header_text': 'Server Success against Choice Chipped Returns', 'header_size': 2, 'header_horizontal_offset': 6,
                'x_label': 'Shots in Rally', 'resolution': GRAPH_RESOLUTION}

avg_color = const.BGR['leafy_green']
shr_color = const.BGR['bright_blue']

nullChart = LineChart([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0], color = const.BGR['bright_grey'], thickness = 0)
shrChart = LineChart(xs, shr_ys, color = shr_color, x_labels = x_labels, annotate_line_heights = True, thickness = 0.125)
avgChart = LineChart(xs, avg_ys, color = avg_color, x_labels = x_labels, annotate_line_heights = True, thickness = 0.125)

n = 30
x_idx = 1
v_xs = np.ones(n) * xs[x_idx]
v_ys = np.linspace(0, avg_ys[x_idx], num = n)
vChart1 = ScatterChart(v_xs, v_ys, color = const.BGR['intense_red'])

n = 40
x_idx = 3
v_xs = np.ones(n) * xs[x_idx]
v_ys = np.linspace(0, avg_ys[x_idx], num = n)
vChart2 = ScatterChart(v_xs, v_ys, color = const.BGR['intense_red'])

### make legend
avgRow = LegendRow(text = 'Server Win%', shape_color = avg_color, text_size = 2, shape_offset = 0.25, text_vertical_offset = 0.3)
shrRow = LegendRow(text = 'Frequency', shape_color = shr_color, text_size = 2, shape_offset = 0.25, text_vertical_offset = 0.3)
nulRow = LegendRow(text = None)

leg1 = Legend([avgRow, nulRow], legend_dimensions = (300, 150), borders = False, header_text = '', resolution = 2000)
leg2 = Legend([avgRow, shrRow], legend_dimensions = (300, 150), borders = False, header_text = '', resolution = 2000)
legFrm1 = leg1.render()
legFrm2 = leg2.render()

tlPt_graph = Point2D((900, 100))

myGraph = Graph([nullChart, avgChart], **graph_kwargs)
avg_graphFrm0 = myGraph.render()
avg_graphFrm0 = overlay(avg_graphFrm0, legFrm1, top_left_point = tlPt_graph)
# avg_graphFrm0.show()

myGraph.add(vChart1)
avg_graphFrm1 = myGraph.render()
avg_graphFrm1 = avg_graphFrm1.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
avg_graphFrm1 = overlay(avg_graphFrm1, legFrm1, top_left_point = tlPt_graph)
# avg_graphFrm1.show()

myGraph.add(vChart2)
avg_graphFrm2 = myGraph.render()
avg_graphFrm2 = avg_graphFrm2.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
avg_graphFrm2 = avg_graphFrm2.put('Serve+2', color = const.BGR['intense_red'], where = (716, 425), size = 1)
avg_graphFrm2 = overlay(avg_graphFrm2, legFrm1, top_left_point = tlPt_graph)
# avg_graphFrm2.show()

myGraph.add(shrChart)
shr_graphFrm = myGraph.render()
shr_graphFrm = shr_graphFrm.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
shr_graphFrm = shr_graphFrm.put('Serve+2', color = const.BGR['intense_red'], where = (716, 425), size = 1)
shr_graphFrm = overlay(shr_graphFrm, legFrm2, top_left_point = tlPt_graph)
# shr_graphFrm.show()

###### DEUCE WIDE ######
side, where = 'deuce', 'wide'
i, j = 1, 1

descStr = 'Forehand Side Serve\nOut Wide'
desc_side = 'right'
desc_height = 'down'

# frames to render on
realFrms = realDict[side + '_' + where]
realFrms_resc = realFrms.rescale(0.5)
realFrms_resc_last = BGRFrames.from_frame(realFrms_resc[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg,
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

    # ### simple predictions for maxima
    # pred_share_winSer = glmReg.predict(regressor_frame) * shareArr
    # max = pred_share_winSer.max()
    # max_idx = pred_share_winSer.argmax()
    # x, y = rastArr_2D[max_idx]
    # maxPt = Point3D((x, y, 0))
    # pvalDict[angle] = (maxPt, max)

    # ### color array for heatmap
    # colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
    #                                                                 alpha_exponent = ALPHA_EXPONENT)
    # rastScn0 = DummyScene([rastArr_3D], [colArr], [alArr])
    # heatDict[angle] = rastScn0

    # ### zone array for best target
    # colArr, alArr = vis_ut.get_field_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
    #                                                         factor_array = shareArr, color = color)
    # rastScn1 = DummyScene([rastArr_3D], [colArr], [alArr])
    # zoneDict[angle] = rastScn1


script_line =   """
                    Take for example this very situation here: Alexandre Mueller is pulled slightly out wide by 
                    Luca van Assche's serve from the deuce side.
                """

realFrms_resc_last_long = BGRFrames.from_frame(realFrms_resc_last[0], 210)
myRst.add(realFrms_resc_last_long.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
statFrms = myRst.render()
# rastFrms[-1].show()

# c) dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)

jointFrms = BGRFrames.from_frames([statFrms, rastFrms])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    The orange area shows our model's target zone for the returning player on a topspin shot; 
                    A typical topspin shot that lands within the target zone 
                    improves the returner's chance of winning the point to above 50%.

                    In calculating this target zone, the model not only takes into account the probability of winning 
                    the point if a ball lands anywhere in the court, but also the deviation of the landing location from the
                    intended target. Therefore, areas too close to the base- and sidelines are excluded as targets.

                    In three words, the target zone basically says: aim here, and if you make it, you'll win the point
                    more often than you lose it. 
                    
                    Our model believes that it is best to hit the topspin return cross-court 
                    when a right-handed server serves out wide from the deuce side of the court.
                    All the examples following will be depicted true to a right-handed server.
                """

zoneScn = zoneDict['tp']

myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    In case of a left-handed server, simply flip the entire situation, and the target zone, across the middle:
                    the forehand side is then on the screen's right, and the exact same cross-court return would be optimal, 
                    according to our model, on an ad-serve out wide.
                """

myFrms_no_annot, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_nolum, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrm_no_annot = myFrms_no_annot[-1]

courtLines = CourtLines.make()
along_line = courtLines['Top Service Line']

self = myFrm_no_annot
def flip(self, along_line):
    from ATMANpy.funcs.util_funcs import cartesian_product
    from ATMANpy.types.Frames import BGRFrame
    from scipy.interpolate import NearestNDInterpolator

    origArr = np.array(along_line.leftPt.coords) ### define origin

    lineVec = along_line.to_vector() ### line as vector from origin
    perpVec = lineVec.find_perpendicular() ### perpendicular vector -> normal direction

    M = np.array([lineVec.coords, (-perpVec).coords]) ### this array, when inverted, and pre-multiplied with the positions of a pixel
                                                        # yields the distances, along lineVec and perpVec,
                                                        # of a pixel to the line along which to flip
    M_inv = np.linalg.inv(M)
    
    idxArr = cartesian_product(np.arange(self.shape[0]), np.arange(self.shape[1])) ### positions of the pixels...
    idxArr_shift = idxArr - origArr                                                 ## ... adjusted to our origin

    dist_coeffs = np.einsum('ij, nj -> ni', M_inv, idxArr_shift)                    #### find distance coeffs -> the relevant one is along perpVec
    shiftArr = dist_coeffs[:, 1, np.newaxis] * np.array(perpVec.coords)[np.newaxis, :] # ... at position = 1

    flip_idxArr = idxArr + 2 * shiftArr     ### to flip across line, shift the position along the perpendicular vector, twice!
    flip_idxArr = flip_idxArr.astype(int)

    oldArr = self.array.copy()
    newArr = np.full_like(oldArr.astype(float), fill_value = np.nan)
    newArr[flip_idxArr[:, 0], flip_idxArr[:, 1]] = oldArr[idxArr[:, 0], idxArr[:, 1]] ### reassign pixels according to their new position

    ### fill black stripes
    newArr = np.where(np.isnan(newArr), np.roll(newArr, 1, axis = 1), newArr) ### very crude
    newArr = newArr.astype('uint8')
    return BGRFrame(newArr)

myFrm_no_annot_flip = flip(myFrm_no_annot, courtLines['Top Service Line'])
# myFrm_no_annot_flip.show()

from ATMANpy.types.Lines import Region
p1, p2, p3, p4 = Point2D((0, 370)), Point2D((960, 370)), Point2D((960, 540)), Point2D((0, 540))
keepReg1 = Region.from_points(p1, p2, p3, p4)
# myFrm_no_annot_flip.put(keepReg1).show()
# myFrm_no_annot.put(keepReg1).show()

p1, p2, p3, p4 = Point2D((0, 0)), Point2D((0, 50)), Point2D((170, 0)), Point2D((170, 50))
keepReg2 = Region.from_points(p1, p2, p3, p4)
# myFrm_no_annot_flip.put(keepReg2).show()
# myFrm_no_annot.put(keepReg2).show()

p1, p2, p3, p4 = Point2D((790, 0)), Point2D((790, 50)), Point2D((960, 0)), Point2D((960, 50))
keepReg3 = Region.from_points(p1, p2, p3, p4)
# myFrm_no_annot_flip.put(keepReg3).show()
# myFrm_no_annot.put(keepReg3).show()

from ATMANpy.types.Frames import MaskFrame
blackMask = MaskFrame.false(myFrm_no_annot_flip)
keepMask = blackMask.within_region(keepReg1) | blackMask.within_region(keepReg2) | blackMask.within_region(keepReg3)

flipFrm = myFrm_no_annot_flip.mask(~keepMask) + myFrm_no_annot.mask(keepMask)
# flipFrm.show()

fhPt_inv = fhPt_annot_heat.retext(bhPt_annot_heat.text).left(50)
bhPt_inv = bhPt_annot_heat.retext(fhPt_annot_heat.text).left(50)

flipFrm_annot = View.make(halfCam, Scene.make_empty(), annotatedPoints = [fhPt_inv, bhPt_inv]).snap(over_array = flipFrm.array)
# flipFrm_annot.show()

myFrms_annot, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrm_annot = myFrms_annot[-1]

myFrm_annot_put = myFrm_annot.put('Righty Server', size = 2, where = (700, 70))
flipFrm_annot_put = flipFrm_annot.put('Lefty Server', size = 2, where = (714, 70))

blinkFrms = vs.blinking(myFrm_annot_put, flipFrm_annot_put, blinks_per_second = 0.5, duration_in_frames = 400)
# DecapStream(blinkFrms, 0, 100).show()

myRst.add(blinkFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)
# DecapStream(rastFrms, 0, 100).show()

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   ""
myFrms_annot = BGRFrames.from_frame(myFrm_annot, 25)
myFrms_annot = put_desc(myFrms_annot, descStr, desc_side, desc_height)
myRst.add(myFrms_annot.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line =   """
                    As for the low chip, the model likes the cross-court return again, 
                    slightly preferring width over depth, compared to the topspin target zone.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg2Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']))
script_line =   """
                    For a high-arching chip, shown here in light-blue, 
                    the target zone is considerably smaller than either the topspin return or the low chip.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

# lsFrm = myFrms[-1]
# myRst.add(lsFrm, i, j)

script_line =   """
                    Let's take a closer look at the target zones. 
                    The flags indicate the highest returner win probability spot within the target zone.
                    For example, if a typical topspin return lands exactly at the base of the orange flag, 
                    the point will be won by the returner 57% of the time, on aggregate.

                    Though the target zones for the topspin and the low chip return are almost of equal size,
                    the low chip return has a higher returner win probability at the peak. 
                    The high chip, having the smallest target zone and lowest peak probability, is discouraged by our model in this situation.
                """

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

# postFrms = BGRFrames.from_frames([transFrms, moveFrms])
# postRst.add(postFrms, i, j)

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']
print(f'{side} + {where} ---> Topspin: {tp_max:.0%}, Low Max: {lo_max:.0%}, High Chip: {hi_max:.0%}')

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 25), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = TRANSITION_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)

postFrms = BGRFrames.from_frames([transFrms, moveFrms, lineFrms, statFrms])
postFrms = put_desc(postFrms, descStr, desc_side, desc_height)
myRst.add(postFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = postFrms[-1]
myRst.add(lsFrm.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)

# postRst.add(postFrms, i, j)
# postFrms[-1].show()

###### DEUCE MIDDLE ######
side, where = 'deuce', 'middle'
i, j = 0, 1

descStr = 'Forehand Side Serve\nDown The Middle'
desc_side = 'right'
desc_height = 'up'

# frames to render on
realFrms = realDict[side + '_' + where]
realFrms_resc = realFrms.rescale(0.5)
realFrms_resc_last = BGRFrames.from_frame(realFrms_resc[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

preFrms = wideDict[side + '_' + where]
preFrms_resc = preFrms.rescale(0.5)

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg, 
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Consider now a serve DOWN THE MIDDLE from the server's forehand side.
                """
rastFrm = myRst.render()[0]
rastFrms0 = BGRFrames.from_frame(rastFrm, 10)
rastFrms0 = overlay(rastFrms0, leg3Frm, top_left_point = tlPt)

preFrms_resc = put_desc(preFrms_resc, descStr, desc_side, desc_height)
myRst.add(preFrms_resc.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms1 = myRst.render()
rastFrms1 = overlay(rastFrms1, leg3Frm, top_left_point = tlPt)

realFrms_resc = put_desc(realFrms_resc, descStr, desc_side, desc_height)
myRst.add(realFrms_resc.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms2 = myRst.render()
rastFrms2 = overlay(rastFrms2, leg3Frm, top_left_point = tlPt)

jointFrms = BGRFrames.from_frames([rastFrms0, rastFrms1, rastFrms2])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp']
script_line =   """
                    For the topspin return, the model indicates a very generous target zone, 
                    ranging from the server's feet to his backhand side.
                """

# c) dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms0 = myRst.render()
rastFrms0 = overlay(rastFrms0, leg3Frm, top_left_point = tlPt)

myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms1 = myRst.render()
rastFrms1 = overlay(rastFrms1, leg3Frm, top_left_point = tlPt)

jointFrms = BGRFrames.from_frames([rastFrms0, rastFrms1])

all_frames.append(jointFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line = "On the other hand, the model still recommends to play the low chip cross-court to the forehand."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']))
script_line = "Again, the high chip has the smallest target zone, requiring depth and width on the server's forehand side."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

# lsFrm = myFrms[-1]
# myRst.add(lsFrm, i, j)

script_line =   """
                    The topspin return is the safe bet, having not only the largest target zone, but
                    also a solid peak win probability of 60%. 
                    The best possible outcome is a perfect low chip, deep into the forehand side,
                    but win probabilities fall off fast around the peak, making for a smaller target zone.
                    The high chip appears to be the inferior choice on aggregate.
                """

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

# postFrms = BGRFrames.from_frames([transFrms, moveFrms])
# postRst.add(postFrms, i, j)

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']
print(f'{side} + {where} ---> Topspin: {tp_max:.0%}, Low Max: {lo_max:.0%}, High Chip: {hi_max:.0%}')

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 25), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = TRANSITION_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)

postFrms = BGRFrames.from_frames([transFrms, moveFrms, lineFrms, statFrms])
postFrms = put_desc(postFrms, descStr, desc_side, desc_height)
myRst.add(postFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = postFrms[-1]
myRst.add(lsFrm.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)

# postRst.add(postFrms, i, j)

# postFrms[-1].show()

# statFrms[-1].show()

###### AD MIDDLE ######
side, where = 'ad', 'middle'
i, j = 0, 0

descStr = 'Backhand Side Serve\nDown The Middle'
desc_side = 'left'
desc_height = 'up'

# frames to render on
realFrms = realDict[side + '_' + where]
realFrms_resc = realFrms.rescale(0.5)
realFrms_resc_last = BGRFrames.from_frame(realFrms_resc[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

preFrms = wideDict[side + '_' + where]
preFrms_resc = preFrms.rescale(0.5)

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg, 
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Up next: a serve down the middle, from the server's BACKHAND side.
                """
rastFrm = myRst.render()[0]
rastFrms0 = BGRFrames.from_frame(rastFrm, 10)
rastFrms0 = overlay(rastFrms0, leg3Frm, top_left_point = tlPt)

preFrms_resc = put_desc(preFrms_resc, descStr, desc_side, desc_height)
myRst.add(preFrms_resc.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms1 = myRst.render()
rastFrms1 = overlay(rastFrms1, leg3Frm, top_left_point = tlPt)

realFrms_resc = put_desc(realFrms_resc, descStr, desc_side, desc_height)
myRst.add(realFrms_resc.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms2 = myRst.render()
rastFrms2 = overlay(rastFrms2, leg3Frm, top_left_point = tlPt)

jointFrms = BGRFrames.from_frames([rastFrms0, rastFrms1, rastFrms2])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

# c) dim and fade in target area
script_line = ""
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp']
script_line =   """
                    Similarly to the forehand-side serve down the middle, the backhand-side serve down the middle should be returned
                    anywhere from down the line to the server's feet. 
                    However, presumably because now the return lands on the server's FOREHAND side, the model requires greater depth
                    to bring the returner's chances to above 50%.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line = "The recipe for a good low chip is bizarrely consistent: target the server's forehand with good width."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']))
script_line = "No change also in the high-chip strategy; the target zone remains woefully small, deep on the forehand side."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

# lsFrm = myFrms[-1]
# myRst.add(lsFrm, i, j)

script_line =   """
                    A closer look at the target zones reveals a similar picture as in the other cases:
                    The low chip return is potentially high-risk high-reward, having a small target zone but high peak, 
                    compared to the topspin return.
                """

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

# postFrms = BGRFrames.from_frames([transFrms, moveFrms])
# postRst.add(postFrms, i, j)

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 25), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = TRANSITION_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)

postFrms = BGRFrames.from_frames([transFrms, moveFrms, lineFrms, statFrms])
postFrms = put_desc(postFrms, descStr, desc_side, desc_height)
myRst.add(postFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = postFrms[-1]
myRst.add(lsFrm.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)

# postFrms[-1].show()

###### AD WIDE ######
side, where = 'ad', 'wide'
i, j = 1, 0

descStr = 'Backhand Side Serve\nOut Wide'
desc_side = 'left'
desc_height = 'down'

# frames to render on
realFrms = realDict[side + '_' + where]
realFrms_resc = realFrms.rescale(0.5)
realFrms_resc_last = BGRFrames.from_frame(realFrms_resc[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

preFrms = wideDict[side + '_' + where]
preFrms_resc = preFrms.rescale(0.5)

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg, 
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Finally, we have a serve out wide from the server's backhand half of the court.
                """
rastFrm = myRst.render()[0]
rastFrms0 = BGRFrames.from_frame(rastFrm, 10)
rastFrms0 = overlay(rastFrms0, leg3Frm, top_left_point = tlPt)

preFrms_resc = put_desc(preFrms_resc, descStr, desc_side, desc_height)
myRst.add(preFrms_resc.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms1 = myRst.render()
rastFrms1 = overlay(rastFrms1, leg3Frm, top_left_point = tlPt)

realFrms_resc = put_desc(realFrms_resc, descStr, desc_side, desc_height)
myRst.add(realFrms_resc.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms2 = myRst.render()
rastFrms2 = overlay(rastFrms2, leg3Frm, top_left_point = tlPt)

jointFrms = BGRFrames.from_frames([rastFrms0, rastFrms1, rastFrms2])

all_frames.append(jointFrms)
script_lines.append(script_line)

# c) dim and fade in target area
script_line = ""
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp']
script_line =   """
                    In this situation, the model recommends playing the topspin return strictly cross-court. 
                    Recall from the earlier discussion that the returner is afforded very little time to reposition on 
                    the topspin return. 
                    On wide serves, the returning player is already off the court, 
                    The model therefore sees it as imperative to send the ball cross-court to reduce the distance the returning player 
                    needs to cover to get back in position.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line =   """
                    The low-chip target zone is on the forehand side again, though decidedly smaller in this case.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['hi']).combine(zoneDict['lo']))
script_line =   """
                    In this isolated case, the high-chip target zone is actually larger than the low-chip zone:
                    though the returner would play himself out of position going down-the-line, the model takes into account
                    that the high-chip affords the returner with plenty of time to recover to almost anywhere along the baseline; 
                    as we saw earlier, this is especially useful on the down-the-line return from out wide.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myFrms = put_desc(myFrms, descStr, desc_side, desc_height)
myRst.add(myFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    Nevertheless, the topspin return cross-court is unambiguously the best option here, 
                    pairing the largest zone with the highest peak probability.
                """

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)


# postFrms = BGRFrames.from_frames([transFrms, moveFrms])
# postRst.add(postFrms, i, j)

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 10), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = TRANSITION_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)

postFrms = BGRFrames.from_frames([transFrms, moveFrms, lineFrms, statFrms])
postFrms = put_desc(postFrms, descStr, desc_side, desc_height)
myRst.add(postFrms.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = postFrms[-1]
myRst.add(lsFrm.framed(FRAME_THICKNESS, FRAME_COLOR), i, j)

# postFrms[-1].show()

# vis_ut.show(all_frames)

####### POST-STUFF #######

# script_line = ""
# postFrms = postRst.render()
# postFrms = overlay(postFrms, leg3Frm, top_left_point = tlPt)
# all_frames.append(postFrms)
# script_lines.append(script_line)

# script_line =   """
#                     Interestingly, though the low chip zones are typically smaller than the topspin target zones, 
#                     their peak probability is higher in three out of four cases, indicating that a low chip is higher risk, but also higher reward stategy according to our model, 
#                     as opposed to the topspin return, which allows for more room for error, but affords a slightly lower returner advantage beyond 50%.
#                     The high-chip return consistently has the lowest peak win probability, and, barring one exception, the smallest target zone;
#                     our model therefore would discourage its use. 
#                 """
# statFrms_post = BGRFrames.from_frame(postFrms[-1], STATIC_FRAMES)
# statFrms_post = overlay(statFrms_post, leg3Frm, top_left_point = tlPt)
# all_frames.append(statFrms_post)
# script_lines.append(script_line)

script_line =   """
                    So in all cases except the backhand-side serve out wide, the low chip is a high-risk high-reward alternative to the topspin return.

                    The remaining mystery is why the model wants the chip to be played to the forehand in all situations.
                    Typically, the serving player can redirect the ball and inject pace more easily. So why chip to the forehand?
                 """
                    # The remaining mystery is the model's consistent infatuation with chip returns to the 
                    # server's forehand side. Typically, these would be easier to attack; a player can redirect balls more easily on the forehand,
                    # and inject more pace on a floating ball; 
                    # why then does the model want the chip to be played to the forehand?
                    
                    # We have alluded to one potential factor earlier: serve + 1 winners are not as promising on chip returns
                    # as serve + 2 combinations; The returner may therefore bait the serving player 
                    # into premature aggression with a well-placed chip to the forehand.

                    # But what explains the aversion to the backhand side, on the chip return? 

rastFrms = myRst.render()
statFrms_post = BGRFrames.from_frame(rastFrms[-1], STATIC_FRAMES)
statFrms_post = overlay(statFrms_post, leg3Frm, top_left_point = tlPt)
all_frames.append(statFrms_post)
script_lines.append(script_line)

# show(all_frames)
# postFrms[-1].show()

#### TEXT TO SAY ###
# -> how to flip for lefty servers!
# -> relevance of descriptive things:
#       -> returner can recover with higher chips: cross court irrelevant!

##############################
#### EXPLANATION FOREHAND ####
##############################
# -> why is there a preference for a return to the forehand side 

retPnts_in_left = retPnts_in.filter(regFrm_in['is_left_handed_server'].values)
retPnts_in_right = retPnts_in.filter(regFrm_in['is_right_handed_server'].values)
retPnts_in_rect = retPnts_in_right.join(retPnts_in_left.flip(across_middle = True, across_net = False)).sort()

def shot_lands_in_or_near_area(myPnt, shot_idx = 0, num_subdivisions = 4):
    from ATMANpy.auxil.court import topReg, botReg

    try:
        myShot = myPnt[shot_idx]
    except IndexError:
        return None
    else:
        x, y = myShot.get_first_bounce_position()
        if x is None and y is None:
            return None
        else:
            bncPt = Point2D((x, y))

            if myShot.get_hit_side() == 'Top':
                myRegs = reversed(botReg.subdivide(num_subdivisions, how = 'vertical').regList)
            else:
                myRegs = list(topReg.subdivide(num_subdivisions, how = 'vertical').regList)

            center_points = [myReg.centroid() for myReg in myRegs]
            voronoi_dists = [bncPt.distance_to(myPt) for myPt in center_points]

            min_idx = np.argmin(voronoi_dists)
            area_nums = list(range(1, num_subdivisions + 1))
            return area_nums[min_idx]

third_is_foreArr = retPnts_in_rect.apply(vis_ut.is_third_shot_fore, dtype = bool)
return_areaArr = retPnts_in_rect.apply(shot_lands_in_or_near_area, dtype = str, shot_idx = 1, num_subdivisions = 4)
third_areaArr = retPnts_in_rect.apply(shot_lands_in_or_near_area, dtype = str, shot_idx = 2, num_subdivisions = 4)
        
assert all(retPnts_in_rect.apply(vis_ut.is_third_shot_fore, dtype = bool) == retPnts_in.apply(vis_ut.is_third_shot_fore, dtype = bool))

serve_foreArr = regFrm_in['serve_fore'].values.astype(bool)
is_chipArr = regFrm_in['is_chipped'].values
server_wins = regFrm_in['srv_wins'].values.astype(bool)
server_wins_weighted = regFrm_in['srv_wins_weighted'].values.astype(float)
# weight_pred = regFrm_in['weight_topspin'].values.astype(float)
weight_pred = regFrm_in['pred_is_topspin'].values.astype(float)
weight_bal = regFrm_in['balance_weight'].values.astype(float)
weight_len = regFrm_in['len_weight'].values.astype(float)

allArr = np.stack([ serve_foreArr, return_areaArr, is_chipArr, third_is_foreArr, third_areaArr, 
                    server_wins, server_wins_weighted,
                    weight_pred, weight_bal, weight_len], axis = 1)

allFrm = pd.DataFrame(allArr, columns = [   'serve_is_fore', 'return_area', 'return_is_chipped', 'third_fore', 'third_area', 
                                            'server_wins', 'server_wins_weighted',
                                            'weight_prd', 'weight_bal', 'weight_len'])

allFrm['serve_is_fore'] = allFrm['serve_is_fore'] == 'True'
allFrm['return_is_chipped'] = allFrm['return_is_chipped'] == 'True'
allFrm['third_fore'] = allFrm['third_fore'] == 'True'
allFrm['server_wins'] = allFrm['server_wins'] == 'True'
allFrm['server_wins_weighted'] = allFrm['server_wins_weighted'].astype(float)
allFrm['weight_prd'] = allFrm['weight_prd'].astype(float)
allFrm['weight_bal'] = allFrm['weight_bal'].astype(float)
allFrm['weight_len'] = allFrm['weight_len'].astype(float)

subFrm = allFrm[allFrm.third_area != 'None']


def weighted_win_percentage_chip(myFrm, attrib = True):
    ### this is for weighted averages if we have is_chipped in the groupby
    if myFrm['return_is_chipped'].unique()[0]: ### only for chipped returns do we apply the pred weight
        jnt_weight = myFrm['weight_prd'] * myFrm['weight_bal']
    else:
        jnt_weight = myFrm['weight_bal']

    if attrib:
        return (myFrm['server_wins_weighted'] * jnt_weight).sum() / jnt_weight.sum()
    else:
        return (myFrm['server_wins'] * jnt_weight).sum() / jnt_weight.sum()

def weighted_win_percentage_nochip(myFrm, attrib = True):
    ### this is for weighted averages if we dont have is_chipped in the groupby
    jnt_weight = myFrm['weight_bal']
    if attrib:
        return (myFrm['server_wins_weighted'] * jnt_weight).sum() / jnt_weight.sum()
    else:
        return (myFrm['server_wins'] * jnt_weight).sum() / jnt_weight.sum()

def weighted_fore_percentage(myFrm):
    jnt_weight = myFrm['weight_bal'] ### no need for the length adjustment! only in 
    return (myFrm['third_fore'] * jnt_weight).sum() / jnt_weight.sum()
   
### Weighted Win% by Area
chip_srv_winFrm = subFrm.groupby(['return_area', 'return_is_chipped']).apply(weighted_win_percentage_chip, attrib = True)
chip_srv_winFrm = chip_srv_winFrm.unstack('return_is_chipped')
chip_ret_winFrm = 1 - chip_srv_winFrm
print(chip_srv_winFrm)
print(chip_ret_winFrm)
# -> Win Percentage Diff between Chip and Tops is greatest in 3, lowest in 1.
# -> for choice chips, with all weights: area 1 is almost exactly equal.

### Weighted: What Percent Shots are Hit Forehand?
foreFrm = subFrm.groupby(['return_is_chipped', 'return_area']).apply(weighted_fore_percentage)
foreFrm = foreFrm.unstack('return_is_chipped')
print(foreFrm)
# -> all shots forehand in 1, most in 2. for 3: very few if tops, 2/3 if chip.

grpFrm = subFrm.groupby(['return_is_chipped', 'return_area', 'serve_is_fore']).apply(weighted_fore_percentage)
grpFrm = grpFrm.unstack('return_is_chipped')
print(grpFrm)
# -> if serve is made from the backhand side, higher likelihood of running around (because less distance!)

### Win% if server runs around 
grpFrm = subFrm.groupby(['return_area', 'third_fore'])['server_wins'].mean()
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)

### Win% if server runs around 
fore_srv_winFrm = subFrm.groupby(['return_area', 'third_fore']).apply(weighted_win_percentage_nochip, attrib = True)
fore_srv_winFrm = fore_srv_winFrm.unstack('third_fore')
fore_ret_winFrm = 1 - fore_srv_winFrm
print(fore_srv_winFrm)
print(fore_ret_winFrm)
# -> if the ball is hit with forehand, small difference in 2, huge difference in 3.

grpFrm = subFrm.groupby(['return_area', 'third_fore'])['server_wins'].count()
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)

### Where are balls hit given shot area
grpFrm = subFrm.groupby(['return_area', 'third_fore'])['third_area'].value_counts(normalize = True)
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)
# -> central ball and wide ball are played with more decisive angles on the forehand! (less goes to the middle)

grpFrm = subFrm.groupby(['return_area', 'third_area', 'third_fore'])['server_wins'].mean()
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)
# -> not necessary to show? For 3, every option is better when run around, for 2, only to 2 and 3; but: low sample size here (for 2, backhand)

grpFrm = subFrm.groupby(['return_area', 'third_area', 'third_fore']).apply(weighted_win_percentage_nochip, attrib = True)
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)

### -> check if the return down the line from A (1->4) is really that much worse for chip vs. topspin
grpFrm = subFrm.groupby(['return_area', 'third_area', 'third_fore', 'return_is_chipped']).apply(weighted_win_percentage_chip, attrib = True)
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)

grpFrm[True].unstack('return_is_chipped') ## -> it is! 57% vs 65% win percentage.

### -> check if the return down the line is so often attempted
grpFrm = subFrm.groupby(['return_area', 'third_area', 'third_fore'])['return_is_chipped'].count()
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)

grpFrm = subFrm.groupby(['return_area', 'third_area', 'third_fore', 'return_is_chipped'])['server_wins'].count()
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)

grpFrm.loc['1']

###### CONTINUE ANIMATING ######
fhPt_annot_fore = fhPt_annot.resize(2).right(16)
bhPt_annot_fore = bhPt_annot.resize(2).left(16)

from ATMANpy.types.Overlays import reverse_cascade, make_text_in

revRst, revFrms = reverse_cascade(myRst.to_frame_raster(), keep_which = None, frames_per_element = 10)
# DecapStream(revFrms, 0, 100).show()

revRst, revFrms = reverse_cascade(myRst.to_frame_raster(), keep_which = [(0, 0)], frames_per_element = 10)
# DecapStream(revFrms, 0, 100).show()



annotated_points = [fhPt_annot_fore, bhPt_annot_fore]

script_line =   """
                    To explain this result, we first divide the server's side of the court
                    into four parts, from A on the forehand side to D on the backhand side.
                """
backFrms0, _ = vs.animate_static(None, background_scene = courtScn, annotated_points = annotated_points, 
                                static_cam = heatCam.resize(2), **prmDict, num_frames = 25,
                                )

from ATMANpy.types.Frames import BGRFrame
origFrm = myRst[0, 0][0]
targFrm = backFrms0[0]
blackFrm = BGRFrame.make_black()
move_frames = 25

out_frames = []
for i in range(move_frames):
    lmbd = i / (move_frames - 1)

    scale_orig = lmbd * 2 + (1 - lmbd)
    scale_targ = (1 - lmbd) * 0.5 + lmbd

    origFrm_scaled = origFrm.rescale(scale_orig)
    targFrm_scaled = targFrm.rescale(scale_targ)
    combFrm = origFrm_scaled.dim(1 - lmbd) + targFrm_scaled.dim(lmbd)

    combFrm_full = overlay(blackFrm, combFrm, top_left_point = Point2D((0, 0)))

    out_frames.append(combFrm_full)

itplFrms0 = BGRFrames.from_single_frames(out_frames)
# DecapStream(itplFrms0, 0, 100).show()

from ATMANpy.types.Shapes3D import Region3D
from ATMANpy.auxil.court import top_a, top_b, top_c, top_d
top_a_3D, top_b_3D, top_c_3D, top_d_3D = Region3D.from_region2D(top_a, z = 0), Region3D.from_region2D(top_b, z = 0), Region3D.from_region2D(top_c, z = 0), Region3D.from_region2D(top_d, z = 0)

regLines3D = [myLn for myReg in [top_a_3D, top_b_3D, top_c_3D, top_d_3D] for myLn in myReg.lineList]
regLines3D = [myLn.recolor(const.BGR['leafy_green']).resize(0.5) for myLn in regLines3D]

abcd_params = {'color': const.BGR['leafy_green'], 'size': 2, 'horizontal_offset': 25}
aPt, bPt, cPt, dPt = make_text_in(top_a_3D, 'A', **abcd_params), make_text_in(top_b_3D, 'B', **abcd_params), make_text_in(top_c_3D, 'C', **abcd_params), make_text_in(top_d_3D, 'D', **abcd_params)

annotated_points = [fhPt_annot_fore, bhPt_annot_fore, aPt, bPt, cPt, dPt]

backFrms1, _ = vs.animate_static(regLines3D, background_scene = courtScn, annotated_points = annotated_points, 
                                static_cam = heatCam.resize(2), **prmDict
                                )

itplFrms1 = vs.interpolate_frames(backFrms0[-1], backFrms1[0], 50)

jointFrms = BGRFrames.from_frames([revFrms, itplFrms0, backFrms0, itplFrms1, backFrms1])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

#########################
#### MAKE ALL GRAPHS ####
#########################

from ATMANpy.auxil.court import birds_eye_half_court_cam
topCam = birds_eye_half_court_cam(side = 'Top', zoom = 1.5).up(10)

from ATMANpy.types.Overlays import Graph, LineChart, ScatterChart

# a) win probability on chip/tops
tops_rets = chip_ret_winFrm[False].values.tolist()
chip_rets = chip_ret_winFrm[True].values.tolist()

topsChart = LineChart([0.5, 1.5, 2.5, 3.5], tops_rets, color = TOPS_COLOR, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'above')
chipChart = LineChart([0.5, 1.5, 2.5, 3.5], chip_rets, color = CHIP_LO_COLOR, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'below')

myGraph1 = Graph([topsChart, chipChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'Ret. Win%', label_size = 2, tick_label_size = 2, 
                x_label_dist = 5, x_tick_label_left_shift = 5)
# myGraph1.render().show()

# b) fore probability on chip/tops
tops_rets = foreFrm[False].values.tolist()
chip_rets = foreFrm[True].values.tolist()

topsChart = LineChart([0.5, 1.5, 2.5, 3.5], tops_rets, color = TOPS_COLOR, thickness = 0.5,
                      annotate_line_heights = True, annotate_line_heights_where = 'above')
chipChart = LineChart([0.5, 1.5, 2.5, 3.5], chip_rets, color = CHIP_LO_COLOR, thickness = 0.5,
                      annotate_line_heights = True, annotate_line_heights_where = 'below')

myGraph2 = Graph([topsChart, chipChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'S+1 Fore%', label_size = 2, tick_label_size = 2,
                x_label_dist = 5, x_tick_label_left_shift = 5)
# myGraph2.render().show()

# c) win probability on fore/back
back_rets = (1 - fore_ret_winFrm[False].values).tolist()
fore_rets = (1 - fore_ret_winFrm[True].values).tolist()

# -> prolong the backhands a bit so they don't look so short on the x-axis
back_rets_extra_val = back_rets[2] - (back_rets[3] - back_rets[2]) / 2
back_rets_extra = [back_rets_extra_val, *back_rets[2:]]

back_color = const.BGR['minty_green']
fore_color = const.BGR['rose_orange']

backChart = LineChart([2, 2.5, 3.5], back_rets_extra, color = back_color, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'below')
foreChart = LineChart([0.5, 1.5, 2.5, 3.5], fore_rets, color = fore_color, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'above')

myGraph3 = Graph([backChart, foreChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 17, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'Srv. Win%', label_size = 2, tick_label_size = 2,
                x_label_dist = 5, x_tick_label_left_shift = 5)
# myGraph3.render().show()

##########################
#### MAKE THE LEGENDS ####
##########################
from ATMANpy.types.Overlays import Legend, LegendRow, overlay

# a) chip / tops
legRows = [LegendRow('... Topspin', TOPS_COLOR, text_size = 2),
           LegendRow('... (Choice) Chip', CHIP_COLOR, text_size = 2)]
myLeg_chip_tops = Legend(legRows, header_text = 'Return is ...', header_offset = 2.7, header_thickness = 1,
                            legend_dimensions = (400, 200), resolution = 2000, borders = None)
legFrm_chip_tops = myLeg_chip_tops.render()
# legFrm_chip_tops.show()

# b) fore / back
legRows = [LegendRow('... Forehand', fore_color, text_size = 2),
           LegendRow('... Backhand', back_color, text_size = 2)]
myLeg_chip_tops = Legend(legRows, header_text = 'Serve + 1 is ...', header_offset = 2.7, header_thickness = 1,
                            legend_dimensions = (400, 200), resolution = 2000, borders = None)
legFrm_fore_back = myLeg_chip_tops.render()
# legFrm_fore_back.show()

##########################
### ANIMATE THE GRAPHS ###
##########################

script_line = "Depending on the area in which the return lands, we look at the returner's success rate, depicted on the y-axis."
motionFrms, _ = vs.animate_cam_motion(regLines3D, start_cam = heatCam.resize(2), stop_cam = topCam, **prmDict, 
                                      background_scene = courtScn, interpolate_func = vs.interpolate_camera_naive,
                                      annotated_points = annotated_points, num_frames = 25)

staticFrms, _= vs.animate_static(regLines3D, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn,
                                      annotated_points = annotated_points, num_frames = 25)

courtLns = CourtLines.make()
srvLn = courtLns['Top Service Line']
lftLn = courtLns['Top Singles Deuce Line']
lftLn = Line(lftLn.high_point(), lftLn.weak_intersect(srvLn))

from ATMANpy.types.Shapes3D import Lines3D
srvLn3D = Line3D.from_line2D(srvLn, z = 1)
lftLn3D = Line3D.from_line2D(lftLn, z = 1)

srvReg3D = Region3D.from_line3D(srvLn3D, width = 0.5, dim = 'y')
lftReg3D = Region3D.from_line3D(lftLn3D, width = 0.5, dim = 'x')
courtScn_subset = Scene([srvReg3D, lftReg3D])

someLn3D = regLines3D[0]
def truncate_line(someLn3D, stop_y = 470):
    leftPt, rightPt = someLn3D.leftPt, someLn3D.rightPt
    lx, ly, lz = leftPt.coords
    rx, ry, rz = rightPt.coords

    ly_new, ry_new = min(stop_y, ly), min(stop_y, ry)
    leftPt_new = Point3D((lx, ly_new, lz))
    rightPt_new = Point3D((rx, ry_new, rz))

    return Line3D(leftPt_new, rightPt_new, color = someLn3D.color, thickness = someLn3D.thickness, alpha = someLn3D.alpha, name = someLn3D.name)

regLines3D_trunc = [truncate_line(someLn) for someLn in regLines3D]

reg_color = (40, 40, 40) #const.BGR['smokey_grey']
top_a_3D_trunc = Region3D(regLines3D_trunc[0:4]).down(1).recolor(reg_color)
top_b_3D_trunc = Region3D(regLines3D_trunc[4:8]).down(1).recolor(reg_color)
top_c_3D_trunc = Region3D(regLines3D_trunc[8:12]).down(1).recolor(reg_color)
top_d_3D_trunc = Region3D(regLines3D_trunc[12:16]).down(1).recolor(reg_color)

tlPt = Point2D((1519, 400))
graphScn, graph_annPnts = myGraph1.preview()
jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
jntPts = [*annotated_points, *graph_annPnts]
jntPts_sub = [*annotated_points, graph_annPnts[-1]]

staticFrms2, _ = vs.animate_static(regLines3D_trunc, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts_sub, num_frames = 25)

itplFrms = vs.interpolate_frames(staticFrms[0], staticFrms2[0], num_frames = 100)


jointFrms = BGRFrames.from_frames([motionFrms, staticFrms, itplFrms, staticFrms2])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = ""
graphFrms, _ = vs.animate_static(jntLns, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts, num_frames = 25)
graphFrms = overlay(graphFrms, legFrm_chip_tops, top_left_point = tlPt)
all_frames.append(graphFrms)
script_lines.append(script_line)


# for i, p in enumerate(graph_annPnts):
#     print(i)
#     print(p)
#     print('_')

line_dim = 0.35

jntLns_tops = [*jntLns[:-1], jntLns[-1].dim(line_dim)]
jntLns_chip = [*jntLns[:-2], jntLns[-2].dim(line_dim), jntLns[-1]]

annPnts_tops = [*annotated_points, *graph_annPnts[:6], *[p.dim(line_dim) for p in graph_annPnts[6:10]], graph_annPnts[10]]
annPnts_chip = [*annotated_points, *graph_annPnts[:2], *[p.dim(line_dim) for p in graph_annPnts[2:6]], *graph_annPnts[6:]]

script_line =   """
                    While the returner's success rate on a topspin ball generally increases as the ball is hit farther to the backhand...
                """
graphFrms_tops, _ = vs.animate_static(jntLns_tops, static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = annPnts_tops, num_frames = 25)
graphFrms_tops = overlay(graphFrms_tops, legFrm_chip_tops, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms[-1], graphFrms_tops[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_tops])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    ... the picture for the chip return is not so linear. 
                """

graphFrms_chip, _ = vs.animate_static(jntLns_chip, static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = annPnts_chip, num_frames = 25)
graphFrms_chip = overlay(graphFrms_chip, legFrm_chip_tops, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms_tops[-1], graphFrms_chip[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_chip])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    Returning into B (that is, slightly to the forehand side) promises the highest success rate ...
                """

graphFrms_chip_b, _ = vs.animate_static(Scene([*jntLns_chip, top_b_3D_trunc]), static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = annPnts_chip, num_frames = 25)
graphFrms_chip_b = overlay(graphFrms_chip_b, legFrm_chip_tops, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms_chip[-1], graphFrms_chip_b[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_chip_b])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    ... whereas returning into C (that is, slightly to the backhand side) is the returner's worst choice, on aggregate.
                """

graphFrms_chip_c, _ = vs.animate_static(Scene([*jntLns_chip, top_c_3D_trunc]), static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = annPnts_chip, num_frames = 25)
graphFrms_chip_c = overlay(graphFrms_chip_c, legFrm_chip_tops, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms_chip_b[-1], graphFrms_chip_c[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_chip_c])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = ""
statFrms = BGRFrames.from_frame(jointFrms[-1], 30)
all_frames.append(statFrms)
script_lines.append(script_line)

script_line =   """
                    One reason for this could be that chip returns, by virtue of their longer time-in-flight, are easier to run around,
                    allowing the server to hit a forehand on a ball to the backhand. 
                    
                    
                """

graphFrms_itpl = vs.interpolate_frames(graphFrms_chip_c[-1], graphFrms[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms])
all_frames.append(jointFrms)
script_lines.append(script_line)

graphScn, graph_annPnts = myGraph2.preview()

jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
jntPts = [*annotated_points, *graph_annPnts]

script_line =   """
                    Let's verify this by looking at the rate at which a ball is hit with the forehand by the server on the third shot, 
                    depending on the area that the return is hit in.
                """

graphFrms, _ = vs.animate_static(jntLns, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts, num_frames = 25)
graphFrms = overlay(graphFrms, legFrm_chip_tops, top_left_point = tlPt)

itplFrms = vs.interpolate_frames(jointFrms[-1], graphFrms[0], 25)
jointFrms = BGRFrames.from_frames([itplFrms, graphFrms])

all_frames.append(graphFrms)
script_lines.append(script_line)

script_line =   """
                    Unsurprisingly, virtually all returns are replied to with the forehand if they land in the forehand half of the server's court.
                """

graphFrms_fore, _ = vs.animate_static(Scene([*jntLns, top_a_3D_trunc, top_b_3D_trunc]), static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts, num_frames = 25)
graphFrms_fore = overlay(graphFrms_fore, legFrm_chip_tops, top_left_point = tlPt)

itplFrms = vs.interpolate_frames(graphFrms[-1], graphFrms_fore[0], 25)
jointFrms = BGRFrames.from_frames([itplFrms, graphFrms_fore])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    However, more than 80% of chipped returns into C are run around, and hit with the forehand. 
                """

jntLns_tops = [*jntLns[:-1], jntLns[-1].dim(line_dim)]
jntLns_chip = [*jntLns[:-2], jntLns[-2].dim(line_dim), jntLns[-1]]

annPnts_tops = [*annotated_points, *graph_annPnts[:6], *[p.dim(line_dim) for p in graph_annPnts[6:10]], graph_annPnts[10]]
annPnts_chip = [*annotated_points, *graph_annPnts[:2], *[p.dim(line_dim) for p in graph_annPnts[2:6]], *graph_annPnts[6:]]

graphFrms_chip_c, _ = vs.animate_static(Scene([*jntLns_chip, top_c_3D_trunc]), static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = annPnts_chip, num_frames = 25)
graphFrms_chip_c = overlay(graphFrms_chip_c, legFrm_chip_tops, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms_fore[-1], graphFrms_chip_c[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_chip_c])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    Even if a chip lands in D, more than half of the time the server will have time to run around the chip and reply with the forehand.
                """

graphFrms_chip_d, _ = vs.animate_static(Scene([*jntLns_chip, top_d_3D_trunc]), static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = annPnts_chip, num_frames = 25)
graphFrms_chip_d = overlay(graphFrms_chip_d, legFrm_chip_tops, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms_chip_c[-1], graphFrms_chip_d[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_chip_d])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

graphScn, graph_annPnts = myGraph3.preview()
graph_annPnts.pop(2)

jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
jntPts = [*annotated_points, *graph_annPnts]

script_line = ""
statFrms = BGRFrames.from_frame(jointFrms[-1], 50)
all_frames.append(statFrms)
script_lines.append(script_line)


script_line =   """
                    The data also shows that running around returns to the backhand pays off for the server: 
                """

graphFrms, _ = vs.animate_static(jntLns, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts, num_frames = 25)
graphFrms = overlay(graphFrms, legFrm_fore_back, top_left_point = tlPt)

itplFrms = vs.interpolate_frames(jointFrms[-1], graphFrms[0], 25)
jointFrms = BGRFrames.from_frames([itplFrms, graphFrms])

all_frames.append(graphFrms)
script_lines.append(script_line)

script_line =   """
                    
                    If a return to the backhand side can be run around, the server's win% goes up from 53% to 57% in area C...
                """

graphFrms_c, _ = vs.animate_static(Scene([*jntLns, top_c_3D_trunc]), static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = jntPts, num_frames = 25)
graphFrms_c = overlay(graphFrms_c, legFrm_fore_back, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms[-1], graphFrms_c[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_c])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    
                    ... and from 50% to 59% in area D.
                """

graphFrms_d, _ = vs.animate_static(Scene([*jntLns, top_d_3D_trunc]), static_cam = topCam, **prmDict, 
                                        background_scene = courtScn_subset,
                                        annotated_points = jntPts, num_frames = 25)
graphFrms_d = overlay(graphFrms_d, legFrm_fore_back, top_left_point = tlPt)

graphFrms_itpl = vs.interpolate_frames(graphFrms_c[-1], graphFrms_d[0], 25)
jointFrms = BGRFrames.from_frames([graphFrms_itpl, graphFrms_d])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    This might be the key in understanding why the chip return should be played to the forehand right away:
                    if it goes to the forehand, the only real attacking shot is a high-risk topspin down-the-line serve + 1, which, 
                    as we have shown earlier, is often employed prematurely against the chip, 
                    with a markedly lower success rate than a serve + 2 combination.
                """
from ATMANpy.types.Overlays import overlay_center
graphFrm = overlay_center(BGRFrame.make_black(), shr_graphFrm)
statFrms = BGRFrames.from_frame(jointFrms[-1], 125)
fadeFrms = vs.interpolate_frames(jointFrms[-1], graphFrm, 100)
jointFrms = BGRFrames.from_frames([statFrms, fadeFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

######### EXAMPLES #########
def get_example_spline_frames(myPnt, third_color = BALL_COLOR, only_last = False, rescale_factor = 3/4, which_regions = ['a']):
    from ATMANpy.auxil.court import bot_a, bot_b, bot_c, bot_d
    from ATMANpy.types.Visual3D import Camera

    serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').values()
    return_splines = myPnt[1].feat_spline_cumul(which_endpoint = 'penultimate').values()
    third_splines = myPnt[2].feat_spline_cumul(which_endpoint = 'penultimate').values()

    serveScns = [Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in splns if spln is not None]) for splns in serve_splines]
    returnScns = [Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in splns if spln is not None]) for splns in return_splines]
    thirdScns = [Scene([spln.recolor(third_color).resize(1.5) for spln in splns if spln is not None]) for splns in third_splines]
    splnScns = serveScns + returnScns + thirdScns

    myCam = Camera.calibrated_brest_behind()

    realFrms = myPnt[:3].stream(confs = confs).join().maybe_to_stream().decap().frames

    ### Final Frames
    regDict = { 'a': Region3D.from_region2D(bot_a, z = 0.25).recolor(const.BGR['soft_yellow']),
                'b': Region3D.from_region2D(bot_b, z = 0.25).recolor(const.BGR['soft_yellow']),
                'c': Region3D.from_region2D(bot_c, z = 0.25).recolor(const.BGR['soft_yellow']),
                'd': Region3D.from_region2D(bot_d, z = 0.25).recolor(const.BGR['soft_yellow'])}
    lastScn = splnScns[-1]
    for region in which_regions:
        lastScn = lastScn.put(regDict[region])

    statFrms, _ = vs.animate_static(lastScn, static_cam = myCam, background_scene = courtScn, num_frames = 1,
                                    background_resolution = 1, argument_resolution = 1, foreground_resolution = 0.5, argument_luminosity = 1)
    statFrm = statFrms[0].rescale(rescale_factor)

    if only_last:
        return statFrm

    ### Movement Frames
    myFrms, _ = vs.animate_sequence(splnScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False,
                                    animate_over = realFrms, 
                                    background_resolution = 1, argument_resolution = 0.5, foreground_resolution = 0.5, argument_luminosity = 1)
    myFrms = myFrms.rescale(rescale_factor)
    
    colFrms = vs.interpolate_frames(myFrms[-1], statFrm, 25)
    
    return myFrms.join(colFrms)

def shot_is_where(myPnt, shot_idx, areas):
    return shot_lands_in_or_near_area(myPnt, shot_idx = shot_idx, num_subdivisions = 4) in areas

### Where are balls hit given shot area
grpFrm = subFrm.groupby(['return_area', 'third_fore'])['third_area'].value_counts(normalize = False)
grpFrm = grpFrm.unstack('third_fore')
back_grpFrm = grpFrm.loc['3'] + grpFrm.loc['4']
back_grpFrm = back_grpFrm / back_grpFrm.sum()
print(back_grpFrm)

checkPnts = dec_retPnts.filter(lambda myPnt: len(myPnt) > 2).filter(vis_ut.server_is_right_handed).filter('is_top_to_bot')
checkPnts = checkPnts.filter(shot_is_where, shot_idx = 1, areas = [3, 4])

fore_checkPnts = checkPnts.filter(vis_ut.is_third_shot_fore)
back_checkPnts = checkPnts.filter(vis_ut.is_third_shot_back)

fore_a, fore_b, fore_c, fore_d = [  fore_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [1]),
                                    fore_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [2]),
                                    fore_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [3]),
                                    fore_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [4])]
                                    
back_a, back_b, back_c, back_d = [  back_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [1]),
                                    back_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [2]),
                                    back_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [3]),
                                    back_checkPnts.filter(shot_is_where, shot_idx = 2, areas = [4])]

### get representative number of examples
num_samples = 16
round_grpFrm = (back_grpFrm * num_samples).round()
print(round_grpFrm)

# ### run-around forehand
# for idx, somePnt in enumerate(back_a):
#     try:
#         myFrms = get_example_spline_frames(somePnt, third_color = fore_color)
#     except:
#         continue
#     else:
#         print(idx)
#         DecapStream(myFrms, 0, 100).show(ms_between_frames = 50)

fore_a_cands = [2, 5, 7, 8, 20, 22, 23, 29, 32, 33, 39]
fore_b_cands = [3]
fore_c_cands = [1, 3, 4, 7, 10, 11]
fore_d_cands = [1, 2, 3, 9, 10, 11, 12, 16, 17, 23, 27, 31, 33, 34, 35, 39, 41, 43, 44, 46]

back_a_cands = [1, 2, 3, 5, 6, 9, 10, 11, 14, 15, 19, 20, 23, 24]
back_b_cands = [2, 3, 6, 10, 13, 14]
back_c_cands = [1, 3, 8, 10, 11, 12, 14, 20, 26, 27, 28, 29, 31, 32, 37]
back_d_cands = [0, 1, 2, 3, 7, 9, 12, 15, 16, 17, 18, 22, 24, 25, 31]

script_line =   """
                    If the chip is run around, the server has two attacking options, not just one: 
                    he can still go down the line with the forehand inside-in - forcing, but risky.
                """

bigRst = FramesRaster(4, 4, borders = None)

### Forehand INSIDE-IN
fore_a_cands = [2, 5, 7, 8, 20, 22, 23, 29, 32, 33, 39]

myFrms = get_example_spline_frames(fore_a[22], third_color = fore_color, only_last = False, which_regions = ['a'])
bigRst.add(myFrms, 0, 0)
bigFrms0 = bigRst.render()
bigRst.add(myFrms[-1], 0, 0)

myFrms = get_example_spline_frames(fore_a[29], third_color = fore_color, only_last = False, which_regions = ['a'])
bigRst.add(myFrms, 0, 1)
bigFrms1 = bigRst.render()
bigRst.add(myFrms[-1], 0, 1)

myFrms = get_example_spline_frames(fore_a[39], third_color = fore_color, only_last = False, which_regions = ['a'])
bigRst.add(myFrms, 0, 2)
bigFrms2 = bigRst.render()
bigRst.add(myFrms[-1], 0, 2)

myFrm = get_example_spline_frames(fore_a[5], third_color = fore_color, only_last = True, which_regions = ['a'])
bigRst.add(myFrm, 0, 3)
statFrm = bigRst.render()[0]
bigFrms3 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(fore_a[8], third_color = fore_color, only_last = True, which_regions = ['a'])
bigRst.add(myFrm, 1, 0)
statFrm = bigRst.render()[0]
bigFrms4 = BGRFrames.from_frame(statFrm, 10)

jointFrms = BGRFrames.from_frames([bigFrms0, bigFrms1, bigFrms2, bigFrms3, bigFrms4])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    The additional option is the forehand inside-out, attacking the returner's backhand cross-court. 
                    At comparatively low risk, it lends itself well as a setup shot in a serve + 2 combination.

                    The 16 cases we are showing here are chosen to be representative 
                    with respect to where the server places them when he can run around the backhand;                   
                """

myFrms = get_example_spline_frames(fore_d[12], third_color = fore_color, only_last = False, which_regions = ['d'])
bigRst.add(myFrms, 1, 1)
bigFrms0 = bigRst.render()
bigRst.add(myFrms[-1], 1, 1)

myFrms = get_example_spline_frames(fore_d[16], third_color = fore_color, only_last = False, which_regions = ['d'])
bigRst.add(myFrms, 1, 2)
bigFrms1 = bigRst.render()
bigRst.add(myFrms[-1], 1, 2)

myFrms = get_example_spline_frames(fore_d[17], third_color = fore_color, only_last = False, which_regions = ['d'])
bigRst.add(myFrms, 1, 3)
bigFrms2 = bigRst.render()
bigRst.add(myFrms[-1], 1, 3)

myFrm = get_example_spline_frames(fore_d[23], third_color = fore_color, only_last = True, which_regions = ['d'])
bigRst.add(myFrm, 2, 0)
statFrm = bigRst.render()[0]
bigFrms3 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(fore_d[27], third_color = fore_color, only_last = True, which_regions = ['d'])
bigRst.add(myFrm, 2, 1)
statFrm = bigRst.render()[0]
bigFrms4 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(fore_d[31], third_color = fore_color, only_last = True, which_regions = ['d'])
bigRst.add(myFrm, 2, 2)
statFrm = bigRst.render()[0]
bigFrms5 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(fore_d[43], third_color = fore_color, only_last = True, which_regions = ['d'])
bigRst.add(myFrm, 2, 3)
statFrm = bigRst.render()[0]
bigFrms6 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(fore_d[46], third_color = fore_color, only_last = True, which_regions = ['d'])
bigRst.add(myFrm, 3, 0)
statFrm = bigRst.render()[0]
bigFrms7 = BGRFrames.from_frame(statFrm, 10)

jointFrms = BGRFrames.from_frames([bigFrms0, bigFrms1, bigFrms2, bigFrms3, bigFrms4, bigFrms5, bigFrms6, bigFrms7])
# DecapStream(jointFrms, 0, 100).show()
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    Only 3 times out of 16, that is, less than 20% of the time, does he go for the passive 
                    forehand down the middle.
                """

### Forehand DOWN THE MIDDLE
fore_b_cands = [3]
fore_c_cands = [1, 3, 4, 7, 10, 11]

myFrms = get_example_spline_frames(fore_c[4], third_color = fore_color, only_last = False, which_regions = ['b', 'c'])
bigRst.add(myFrms, 3, 1)
bigFrms0 = bigRst.render()
bigRst.add(myFrms[-1], 3, 1)

myFrms = get_example_spline_frames(fore_c[1], third_color = fore_color, only_last = False, which_regions = ['b', 'c'])
bigRst.add(myFrms, 3, 2)
bigFrms1 = bigRst.render()
bigRst.add(myFrms[-1], 3, 2)

myFrms = get_example_spline_frames(fore_b[3], third_color = fore_color, only_last = False, which_regions = ['b', 'c'])
bigRst.add(myFrms, 3, 3)
bigFrms2 = bigRst.render()
bigRst.add(myFrms[-1], 3, 3)

jointFrms = BGRFrames.from_frames([bigFrms0, bigFrms1, bigFrms2])
# DecapStream(jointFrms, 0, 100).show()
all_frames.append(jointFrms)
script_lines.append(script_line)

bigRst_orig = bigRst.deepcopy()
bigRst_inout = bigRst.deepcopy()
bigRst_inin = bigRst.deepcopy()

inin_idxs = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
inout_idxs = [(1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0)]
mid_idxs = [(3, 1), (3, 2), (3, 3)]

dim_factor = 0.1
for r, c in inout_idxs + mid_idxs:
    bigRst_inin[r, c] = bigRst_inin[r, c].dim(dim_factor)

for r, c in inin_idxs + mid_idxs:
    bigRst_inout[r, c] = bigRst_inout[r, c].dim(dim_factor)

bigFrm_inin = bigRst_inin.render()[0]
bigFrm_inout = bigRst_inout.render()[0]

bigFrms_itpl0 = vs.interpolate_frames(jointFrms[-1], bigFrm_inout, 50)
bigFrms_itpl1 = vs.interpolate_frames(bigFrm_inout, bigFrm_inin, 50)

script_line =   """
                    Around half of the time, he goes for the forehand inside-out...
                """
all_frames.append(bigFrms_itpl0)
script_lines.append(script_line)

script_line =   """
                    ... and about a third of the time, he goes for the forehand inside-in.
                    Now let's see what happens, placement-wise, if the return to the backhand is not run around.
                """
all_frames.append(bigFrms_itpl1)
script_lines.append(script_line)

script_line = ""
revRst, revFrms = reverse_cascade(bigRst_inin, frames_per_element = 10, acceleration = 1.15)
all_frames.append(revFrms)
script_lines.append(script_line)

###### BACKHAND ########
print(round_grpFrm)
back_a_cands = [1, 2, 3, 5, 6, 9, 10, 11, 14, 15, 19, 20, 23, 24]
back_b_cands = [2, 3, 6, 10, 13, 14]
back_c_cands = [1, 3, 8, 10, 11, 12, 14, 20, 26, 27, 28, 29, 31, 32, 37]
back_d_cands = [0, 1, 2, 3, 7, 9, 12, 15, 16, 17, 18, 22, 24, 25, 31]

bigRst = FramesRaster(4, 4)

script_line =   """
                    A few representative backhands go down the line...
                """

myFrms = get_example_spline_frames(back_a[10], third_color = back_color, only_last = False, which_regions = ['a'])
bigRst.add(myFrms, 0, 0)
bigFrms0 = bigRst.render()
bigRst.add(myFrms[-1], 0, 0)

myFrms = get_example_spline_frames(back_a[15], third_color = back_color, only_last = False, which_regions = ['a'])
bigRst.add(myFrms, 0, 1)
bigFrms1 = bigRst.render()
bigRst.add(myFrms[-1], 0, 1)

myFrms = get_example_spline_frames(back_a[20], third_color = back_color, only_last = False, which_regions = ['a'])
bigRst.add(myFrms, 0, 2)
bigFrms2 = bigRst.render()
bigRst.add(myFrms[-1], 0, 2)

myFrm = get_example_spline_frames(back_a[1], third_color = back_color, only_last = True, which_regions = ['a'])
bigRst.add(myFrm, 0, 3)
statFrm = bigRst.render()[0]
bigFrms3 = BGRFrames.from_frame(statFrm, 10)

jointFrms = BGRFrames.from_frames([bigFrms0, bigFrms1, bigFrms2, bigFrms3])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    Some others go cross-court...
                """

myFrms = get_example_spline_frames(back_d[17], third_color = back_color, only_last = False, which_regions = ['d'])
bigRst.add(myFrms, 1, 0)
bigFrms0 = bigRst.render()
bigRst.add(myFrms[-1], 1, 0)

myFrms = get_example_spline_frames(back_d[12], third_color = back_color, only_last = False, which_regions = ['d'])
bigRst.add(myFrms, 1, 1)
bigFrms1 = bigRst.render()
bigRst.add(myFrms[-1], 1, 1)

myFrms = get_example_spline_frames(back_d[9], third_color = back_color, only_last = False, which_regions = ['d'])
bigRst.add(myFrms, 1, 2)
bigFrms2 = bigRst.render()
bigRst.add(myFrms[-1], 1, 2)

myFrm = get_example_spline_frames(back_d[0], third_color = back_color, only_last = True, which_regions = ['d'])
bigRst.add(myFrm, 1, 3)
statFrm = bigRst.render()[0]
bigFrms3 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(back_d[1], third_color = back_color, only_last = True, which_regions = ['d'])
bigRst.add(myFrm, 2, 0)
statFrm = bigRst.render()[0]
bigFrms4 = BGRFrames.from_frame(statFrm, 10)

jointFrms = BGRFrames.from_frames([bigFrms0, bigFrms1, bigFrms2, bigFrms3, bigFrms4])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    ... but for the most part, our representative backhands go rather passively back toward the middle of the court.
                """

myFrms = get_example_spline_frames(back_b[10], third_color = back_color, only_last = False, which_regions = ['b', 'c'])
bigRst.add(myFrms, 2, 1)
bigFrms0 = bigRst.render()
bigRst.add(myFrms[-1], 2, 1)

myFrms = get_example_spline_frames(back_c[10], third_color = back_color, only_last = False, which_regions = ['b', 'c'])
bigRst.add(myFrms, 2, 2)
bigFrms1 = bigRst.render()
bigRst.add(myFrms[-1], 2, 2)

myFrms = get_example_spline_frames(back_c[20], third_color = back_color, only_last = False, which_regions = ['b', 'c'])
bigRst.add(myFrms, 2, 3)
bigFrms2 = bigRst.render()
bigRst.add(myFrms[-1], 2, 3)

myFrm = get_example_spline_frames(back_b[13], third_color = back_color, only_last = True, which_regions = ['b', 'c'])
bigRst.add(myFrm, 3, 0)
statFrm = bigRst.render()[0]
bigFrms3 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(back_b[14], third_color = back_color, only_last = True, which_regions = ['b', 'c'])
bigRst.add(myFrm, 3, 1)
statFrm = bigRst.render()[0]
bigFrms4 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(back_c[26], third_color = back_color, only_last = True, which_regions = ['b', 'c'])
bigRst.add(myFrm, 3, 2)
statFrm = bigRst.render()[0]
bigFrms5 = BGRFrames.from_frame(statFrm, 10)

myFrm = get_example_spline_frames(back_c[28], third_color = back_color, only_last = True, which_regions = ['b', 'c'])
bigRst.add(myFrm, 3, 3)
statFrm = bigRst.render()[0]
bigFrms6 = BGRFrames.from_frame(statFrm, 10)

jointFrms = BGRFrames.from_frames([bigFrms0, bigFrms1, bigFrms2, bigFrms3, bigFrms4, bigFrms5, bigFrms6])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    Putting it all together, we have confirmed that the model has good reason to avoid the backhand-side chip return;
                    it is unwilling to give the server the possibility of running around the backhand and hitting an inside-out or inside-in forehand.
                """
bigFrm = bigRst.render()[0]
bigFrms = BGRFrames.from_frame(bigFrm, STATIC_FRAMES)
all_frames.append(bigFrms)
script_lines.append(script_line)

####### SUMMARY #######

# what have we shown?
### -> where to play the chip
### -> how to play the chip

# caveat: specific subset of players, only one tournament of data!
### -> would be very interesting to do this for one specific player, but for this, much more data!

script_line = ""
itplFrms = vs.interpolate_frames(bigFrms[-1], statFrms_post[-1], STATIC_FRAMES)
all_frames.append(itplFrms)
script_lines.append(script_line)

script_line =   """
                    So - to summarize: for a specific set of players in a sample of twenty tournament matches played, 
                    the chip return is a viable strategy as long as it is hit low and wide to the forehand of the server, 
                    with maybe an exception on a serve out wide from the server's backhand side. 
                    The topspin return is always a solid choice: The way to go here is to hit cross court if pulled out wide, 
                    and to the feet or away from the server if the serve is down the middle. 
                    Note that these are aggregate results; they relate to the "average" challenger-level player.
                    Strictly speaking, however, the "average" challenger-level player does not exist:
                    This analysis would be more actionable if it was tailor-made for one specific player, 
                    but for this, we'd need much more data. However, in principle, the methods and models would be the same,
                    and would allow us to state with some confidence how to return a ball against that specific player on serve.
                """
                
all_frames.append(statFrms_post)
script_lines.append(script_line)

# show(all_frames)

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
