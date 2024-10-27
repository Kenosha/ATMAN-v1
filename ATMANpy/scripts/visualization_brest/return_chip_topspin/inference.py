
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


import ATMANpy.auxil.constants as const
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Point, Shots, read_in_all_tables
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

DESCRIPTOR = 'descriptive'
SAVE_FOLDER = 'return_chip_topspin'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

########################################
######## INSTRUCTIONS ##################
########################################
SAVE_REGS = False

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
prmDict_annot_heat['argument_resolution'] = 10
prmDict_annot_heat['argument_luminosity'] = 1.5

backView_half = View.make(halfCam, courtScn, resolution = 1, annotatedPoints = [fhPt_annot, bhPt_annot])
backView_heat = View.make(heatCam, courtScn, resolution = 1, annotatedPoints = [fhPt_annot_heat, bhPt_annot_heat])

backView_half.snap().show()
backView_heat.snap().show()

### Points over which to render
aPnt = retPnts_in.get(26, 40)
bPnt = retPnts_in.get(3, 141)
vis_ut.is_serve_down_middle(aPnt)
vis_ut.is_serve_out_wide(bPnt)
### b -> is out wide! can use it as initial example

def get_spline_frames(myPnt):
    serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').values()
    splnScns = [Scene([spln.recolor(BALL_COLOR) for spln in splns if spln is not None]) for splns in serve_splines]
    realFrms = myPnt[0].stream(confs = confs).decap().frames
    myFrms, _ = vs.animate_sequence(splnScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = realFrms.rescale(3), **prmDict)
    return myFrms

def get_wide_frames(myPnt, pre_frames = 150):
    preDtm = myPnt.stream(confs = confs).widen(pre_frames).decap()
    preFrms = preDtm.frames[:pre_frames]
    preArr_dimmed = np.stack([f.dim((idx / pre_frames) ** 2).array for idx, f in enumerate(preFrms)], axis = 0)
    preFrms_dimmed = BGRFrames(preArr_dimmed).rescale(3)
    return preFrms_dimmed

def get_spline_and_player_scene(myPnt):
    _, serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').last()
    outScn = Scene([spln.recolor(BALL_COLOR) for spln in serve_splines if spln is not None])
    
    x0, y0 = myPnt[0].get_player_hit_position()
    x1, y1 = myPnt[1].get_player_hit_position()

    srvPly = Player.make((x0, y0), color = const.BGR['light_brown'])
    recPly = Player.make((x1, y1), color = const.BGR['light_brown'])

    return outScn.put(srvPly).put(recPly) 

def is_return_topspin(myPnt):
    ### filter for topspin returns so that we can introduce counterfactual.
    return myPnt.get_return().is_topspin_visual()

realDict = {}
wideDict = {}
sceneDict = {}

pre_frames = 150

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
myFrms = get_spline_frames(relPnts[0])
preFrms = get_wide_frames(relPnts[0], pre_frames = pre_frames)
myScn = get_spline_and_player_scene(relPnts[0])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['deuce_middle'] = myFrms
wideDict['deuce_middle'] = preFrms
sceneDict['deuce_middle'] = myScn

# c) ad, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[41]) ### not nakashima again.
preFrms = get_wide_frames(relPnts[41], pre_frames = pre_frames) 
myScn = get_spline_and_player_scene(relPnts[41])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['ad_middle'] = myFrms
wideDict['ad_middle'] = preFrms
sceneDict['ad_middle'] = myScn

# d) ad, out wide
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[100])
preFrms = get_wide_frames(relPnts[100], pre_frames = pre_frames) 
myScn = get_spline_and_player_scene(relPnts[100])
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
myRst = FramesRaster(2, 2, borders = 2)
all_frames = []
script_lines = []
dim_factor = 0.25
annot_color = const.BGR['bright_grey']
bandwidth_field = 50

postRst = FramesRaster(2, 2, borders = 2)
backFrm = View.make(halfCam, courtScn).snap()

comparison_prob_dict = {'tp': 0.55, ### this way, target areas will have roughly the same size:
                        'lo': 0.55,      ### we then annotate their quality with annotation point
                        'hi': 0.5}

color_dict = {'tp': TOPS_COLOR,
              'lo': CHIP_LO_COLOR,
              'hi': CHIP_HI_COLOR}

from ATMANpy.types.Overlays import Legend, LegendRow, overlay
tpRow = LegendRow('Topspin', shape_color = TOPS_COLOR, text_size = 2)
loRow = LegendRow('Low Chip', shape_color = CHIP_LO_COLOR, text_size = 2)
hiRow = LegendRow('High Chip', shape_color = CHIP_HI_COLOR, text_size = 2)

leg_dims = (250, 250)
leg_params = {'borders': True, 'header_text': ' ', 'resolution': 100, 'legend_dimensions': leg_dims}

leg1 = Legend([tpRow, LegendRow(None), LegendRow(None), LegendRow(None)], **leg_params)
leg1Frm = leg1.render()
# leg1Frm.show()

leg2 = Legend([tpRow, loRow, LegendRow(None), LegendRow(None)], **leg_params)
leg2Frm = leg2.render()
# leg2Frm.show()

leg3 = Legend([tpRow, loRow, hiRow, LegendRow(None)], **leg_params)
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


###### DEUCE WIDE ######
side, where = 'deuce', 'wide'
i, j = 1, 1

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

myRst.add(realFrms_resc_last, i, j)
rastFrms = myRst.render()
all_frames.append(rastFrms)
script_lines.append(script_line)
# rastFrms[-1].show()

script_line = ""
# c) dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp']
script_line =   """
                    The orange area shows the target zone for the returning player on a topspin shot; 
                    In calculating this target zone, the model not only takes into account the probability of winning 
                    the point if a ball lands anywhere in the court, but also the deviation of the landing location from the
                    intended target. Therefore, areas too close to the base- and sidelines are excluded as targets, even though
                    a shot that would land here would typically yield the highest return win probability. 
                    Nevertheless, it still holds that any return that lands within the target zone 
                    improves the returner chance of winning the point to above 50%, according to the data and our model.
                    
                    In three words, the target zone basically says: aim here, and if you make it, you'll win the point
                    more often than you lose it. 

                    Our model believes that it is best hit the topspin return cross-court in this specific situation:
                    a right-handed server serving from the deuce side of the court, out wide. 
                    All the examples following will be depicted true to a right-handed server, 
                    in case of a left-handed server, simply flip the entire situation, and the target zone, across the middle:
                    the forehand side is then on the screen's right, and the exact same cross-court return would be optimal, 
                    according to our model, on an ad-serve out wide.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line =   """
                    As for the low chip, the model likes the cross-court return again. 
                    This time, it may land more shallow in the court, but needs a more acute angle to remain 
                    above 50%.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg2Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']))
script_line =   """
                    For a high bouncing chip, shown here in light-blue, 
                    the target zone is considerably smaller than either the topspin return or the low chip.
                    A promising high chip needs much more depth than the low chip; it may not land as shallow in the court.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

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
postRst.add(postFrms, i, j)
# postFrms[-1].show()

postFrms[-1].show()

###### DEUCE MIDDLE ######
side, where = 'deuce', 'middle'
i, j = 0, 1

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
                    Consider now a serve down the middle from the server's forehand side.
                """
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = ""
myRst.add(preFrms_resc, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = ""
myRst.add(realFrms_resc, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

# c) dim and fade in target area
script_line = ""
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp']
script_line =   """
                    For the topspin return, the model indicates a very generous target zone, 
                    ranging from the server's feet to his backhand side.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line = "On the other hand, the model still recommends to play the low chip cross-court, to the returner's forehand."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']))
script_line = "Again, the low chip has the smallest target zone, requiring depth and width on the server's forehand side."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

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
postRst.add(postFrms, i, j)

postFrms[-1].show()

# statFrms[-1].show()

###### AD MIDDLE ######
side, where = 'ad', 'middle'
i, j = 0, 0

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
                    Up next: a serve down the middle, from the server's backhand side, 
                    that is: the ad side for a right-handed server, or the deuce side for a lefty server
                """
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = ""
myRst.add(preFrms_resc, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = ""
myRst.add(realFrms_resc, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

# c) dim and fade in target area
script_line = ""
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp']
script_line =   """
                    Similarly to the forehand-side return down the middle, the backhand-side return down the middle should be played
                    anywhere from down the line to the returner's feet. 
                    However, presumably because now the ball lands on the server's forehand side, the return requires greater depth
                    to bring the returner's chances to above 50%.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line = "The recipe for a good low chip is bizarrely consistent: target the server's forehand with good width."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']))
script_line = "No change also in the high-chip strategy; the target zone remains woefully small, deep on the forehand side."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

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
postRst.add(postFrms, i, j)

postFrms[-1].show()

###### AD WIDE ######
side, where = 'ad', 'wide'
i, j = 1, 0

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
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = ""
myRst.add(preFrms_resc, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = ""
myRst.add(realFrms_resc, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

# c) dim and fade in target area
script_line = ""
itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp']
script_line =   """
                    In this situation, the model recommends playing the topspin return strictly cross-court. 
                    Recall from the earlier discussion that the returner is afforded with very little time to reposition on 
                    the topspin return. 
                    On wide serves, the returning player is already off the court, 
                    The model therefore sees it as imperative to send the ball cross-court to reduce the distance the returning player 
                    needs to cover. 
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']))
script_line =   """
                    The low-chip target zone is on the forehand side again, 
                    though decidedly smaller in this case, probably owing to the fact that the returner plays himself out of 
                    position.
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = consolidate_dummy_scene(zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']))
script_line =   """
                    The high-chip target zone is virtually identical with the low-chip zone and not much smaller, as usual:
                    though the returner would play himself out of position with the high-chip, the model takes into account
                    that the high-chip affords the returner with plenty of time to recover to almost anywhere along the baseline.

                    Having gone through the four serve cases, let's take a closer look at the target zones. 
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

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
postRst.add(postFrms, i, j)

postFrms[-1].show()

####### POST-STUFF #######

script_line = ""
postFrms = postRst.render()
all_frames.append(postFrms)
script_lines.append(script_line)

script_line =   """
                    The flags indicate the highest returner win probability point within the target zone.
                    Interestingly, though the low chip zones are typically smaller than the topspin target zones, 
                    their peak probability is higher, indicating that a low chip is higher risk, but also higher reward
                    according to our model, as opposed to the topspin return, which allows more room for error, 
                    but a slightly lower returner advantage beyond 50%.
                    The high chip return consistently has not only the smallest target zone, but also the lowest peak probability;
                    our model therefore would discourage its use. 
                """
statFrms_post = BGRFrames.from_frame(postFrms[-1], STATIC_FRAMES)
all_frames.append(statFrms_post)
script_lines.append(script_line)

script_line =   """
                    The remaining mystery to take a look at is the model's consistent infatuation with chip returns to the 
                    server's forehand side. Typically, these would be easier to attack; a player can redirect balls down the line
                    more easily on the forehand, and inject more pace on a floating ball; why then does the model want the chip 
                    to be played to the forehand?

                    We have alluded to one potential factor earlier: serve + 1 winners are not as promising on chip returns
                    than serve + 2 combinations; The returner may therefore goad the serving player 
                    into premature aggression with a well-placed chip to the forehand.

                    But what explains the aversion to the backhand side, on the chip return? 
                """
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


###### CONTINUE ANIMATING ######
fhPt_annot_fore = fhPt_annot.resize(2).right(5)
bhPt_annot_fore = bhPt_annot.resize(2)

from ATMANpy.types.Overlays import reverse_cascade, make_text_in

revRst, revFrms = reverse_cascade(postRst.to_frame_raster(), keep_which = None)

script_line = ""
all_frames.append(revFrms)
script_lines.append(script_line)

annotated_points = [fhPt_annot_fore, bhPt_annot_fore]

script_line =   """
                    To explain the aversion to the backhand-side chip at least partially, we first divide the server's side of the court
                    into four parts, from A on the forehand side to D on the backhand side.
                """
backFrms, _ = vs.animate_static(None, background_scene = courtScn, annotated_points = annotated_points, 
                                static_cam = heatCam.resize(2), **prmDict
                                )
all_frames.append(backFrms)
script_lines.append(script_line)

from ATMANpy.types.Shapes3D import Region3D
from ATMANpy.auxil.court import top_a, top_b, top_c, top_d
top_a_3D, top_b_3D, top_c_3D, top_d_3D = Region3D.from_region2D(top_a, z = 0), Region3D.from_region2D(top_b, z = 0), Region3D.from_region2D(top_c, z = 0), Region3D.from_region2D(top_d, z = 0)

regLines3D = [myLn for myReg in [top_a_3D, top_b_3D, top_c_3D, top_d_3D] for myLn in myReg.lineList]
regLines3D = [myLn.recolor(const.BGR['leafy_green']) for myLn in regLines3D]

script_line = ""
abcd_params = {'color': const.BGR['leafy_green'], 'size': 2, 'horizontal_offset': 25}
aPt, bPt, cPt, dPt = make_text_in(top_a_3D, 'A', **abcd_params), make_text_in(top_b_3D, 'B', **abcd_params), make_text_in(top_c_3D, 'C', **abcd_params), make_text_in(top_d_3D, 'D', **abcd_params)

annotated_points = [fhPt_annot_fore, bhPt_annot_fore, aPt, bPt, cPt, dPt]

backFrms, _ = vs.animate_static(regLines3D, background_scene = courtScn, annotated_points = annotated_points, 
                                static_cam = heatCam.resize(2), **prmDict
                                )
all_frames.append(backFrms)
script_lines.append(script_line)

from ATMANpy.auxil.court import birds_eye_half_court_cam
topCam = birds_eye_half_court_cam(side = 'Top', zoom = 1.5).up(10)

script_line = ""
motionFrms, _ = vs.animate_cam_motion(regLines3D, start_cam = heatCam.resize(2), stop_cam = topCam, **prmDict, 
                                      background_scene = courtScn,
                                      annotated_points = annotated_points, num_frames = 25)
all_frames.append(motionFrms)
script_lines.append(script_line)

script_line = "Let's look at the returner's success rate depending on the area in which his return lands."
staticFrms, _= vs.animate_static(regLines3D, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn,
                                      annotated_points = annotated_points, num_frames = 25)
all_frames.append(staticFrms)
script_lines.append(script_line)

courtLns = CourtLines.make()
srvLn = courtLns['Top Service Line']
lftLn = courtLns['Top Singles Deuce Line']
lftLn = Line(lftLn.high_point(), lftLn.weak_intersect(srvLn))

from ATMANpy.types.Shapes3D import Lines3D
srvLn3D = Line3D.from_line2D(srvLn, z = 1)
lftLn3D = Line3D.from_line2D(lftLn, z = 1)

srvReg3D = Region3D.from_line3D(srvLn3D, width = 2, dim = 'y')
lftReg3D = Region3D.from_line3D(lftLn3D, width = 2, dim = 'x')
courtScn_subset = Scene([srvReg3D, lftReg3D])

someLn3D = regLines3D[0]
def truncate_line(someLn3D, stop_y = 470):
    leftPt, rightPt = someLn3D.leftPt, someLn3D.rightPt
    lx, ly, lz = leftPt.coords
    rx, ry, rz = rightPt.coords

    ly_new, ry_new = min(stop_y, ly), min(stop_y, ry)
    leftPt_new = Point3D((lx, ly_new, lz))
    rightPt_new = Point3D((rx, ry_new, rz))

    return Line3D(leftPt_new, rightPt_new, color = someLn3D.color, thickness = someLn3D.thickness, alpha = someLn3D.alpha)

regLines3D_trunc = [truncate_line(someLn) for someLn in regLines3D]

# court = courtScn.shapes[0]
# courtBox = court[2]
# courtScn_subset = Scene([court[4], court[5], court[8], court[9]])

staticFrms2, _ = vs.animate_static(regLines3D_trunc, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = annotated_points, num_frames = 25)

itplFrms = vs.interpolate_frames(staticFrms[0], staticFrms2[0], num_frames = 100)

script_line = ""
all_frames.append(itplFrms)
script_lines.append(script_line)

script_line = ""
all_frames.append(staticFrms2)
script_lines.append(script_line)

#########################
#### MAKE ALL GRAPHS ####
#########################
from ATMANpy.types.Overlays import Graph, LineChart, ScatterChart

# a) win probability on chip/tops
tops_rets = chip_ret_winFrm[False].values.tolist()
chip_rets = chip_ret_winFrm[True].values.tolist()

topsChart = LineChart([0.5, 1.5, 2.5, 3.5], tops_rets, color = TOPS_COLOR, annotate_line_heights = True, annotate_line_heights_where = 'above')
chipChart = LineChart([0.5, 1.5, 2.5, 3.5], chip_rets, color = CHIP_LO_COLOR, annotate_line_heights = True, annotate_line_heights_where = 'below')

myGraph1 = Graph([topsChart, chipChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'Ret. Win%', label_size = 2, tick_label_size = 2, 
                x_label_dist = 5, x_tick_label_left_shift = 5)
myGraph1.render().show()

# b) fore probability on chip/tops
tops_rets = foreFrm[False].values.tolist()
chip_rets = foreFrm[True].values.tolist()

topsChart = LineChart([0.5, 1.5, 2.5, 3.5], tops_rets, color = TOPS_COLOR, annotate_line_heights = True, annotate_line_heights_where = 'above')
chipChart = LineChart([0.5, 1.5, 2.5, 3.5], chip_rets, color = CHIP_LO_COLOR, annotate_line_heights = True, annotate_line_heights_where = 'below')

myGraph2 = Graph([topsChart, chipChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'S+1 Fore%', label_size = 2, tick_label_size = 2,
                x_label_dist = 5, x_tick_label_left_shift = 5)
myGraph2.render().show()

# c) win probability on fore/back
back_rets = fore_ret_winFrm[False].values.tolist()
fore_rets = fore_ret_winFrm[True].values.tolist()

# -> prolong the backhands a bit so they don't look so short on the x-axis
back_rets_extra_val = back_rets[2] - (back_rets[3] - back_rets[2]) / 2
back_rets_extra = [back_rets_extra_val, *back_rets[2:]]

back_color = const.BGR['minty_green']
fore_color = const.BGR['rose_orange']

backChart = LineChart([2, 2.5, 3.5], back_rets_extra, color = back_color, annotate_line_heights = True, annotate_line_heights_where = 'above')
foreChart = LineChart([0.5, 1.5, 2.5, 3.5], fore_rets, color = fore_color, annotate_line_heights = True, annotate_line_heights_where = 'below')

myGraph3 = Graph([backChart, foreChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'Ret. Win%', label_size = 2, tick_label_size = 2,
                x_label_dist = 5, x_tick_label_left_shift = 5)
myGraph3.render().show()

##########################
#### MAKE THE LEGENDS ####
##########################
from ATMANpy.types.Overlays import Legend, LegendRow, overlay

# a) chip / tops
legRows = [LegendRow('... Topspin', TOPS_COLOR, text_size = 2),
           LegendRow('... Chip', CHIP_LO_COLOR, text_size = 2)]
myLeg_chip_tops = Legend(legRows, header_text = 'Return is ...', header_offset = 2.7, header_thickness = 1,
                            legend_dimensions = (400, 200), resolution = 100, borders = False)
legFrm_chip_tops = myLeg_chip_tops.render()
legFrm_chip_tops.show()

# b) fore / back
legRows = [LegendRow('... Forehand', fore_color, text_size = 2),
           LegendRow('... Backhand', back_color, text_size = 2)]
myLeg_chip_tops = Legend(legRows, header_text = 'Serve + 1 is ...', header_offset = 2.7, header_thickness = 1,
                            legend_dimensions = (400, 200), resolution = 100, borders = False)
legFrm_fore_back = myLeg_chip_tops.render()
legFrm_fore_back.show()


##########################
### ANIMATE THE GRAPHS ###
##########################
tlPt = Point2D((1519, 400))


graphScn, graph_annPnts = myGraph1.preview()

jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
jntPts = [*annotated_points, *graph_annPnts]

script_line =   """
                    While the returner's success rate on a topspin ball generally increases as the ball is hit farther to the backhand,
                    the picture for the chip return is not so linear; 
                    returning into B (that is, slightly to the forehand side) promises the highest success rate, 
                    whereas returning into C (that is, slightly to the backhand side) is the returner's worst choice, on aggregate.

                    One reason for this could be that chip returns, by virtue of their longer time-in-flight, are easier to run around,
                    allowing the server to hit a forehand inside-in or inside-out on a ball to the backhand. 
                    
                    Let's verify this by looking at the rate at which a ball is hit with the forehand by the server on the third shot, depending on the area that the return is hit in.
                """
graphFrms, _ = vs.animate_static(jntLns, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts, num_frames = 25)
graphFrms = overlay(graphFrms, legFrm_chip_tops, top_left_point = tlPt)
script_lines.append(script_line)
all_frames.append(graphFrms)

graphScn, graph_annPnts = myGraph2.preview()

jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
jntPts = [*annotated_points, *graph_annPnts]

script_line =   """
                    Reading the graph from left-to-right, virtually all returns, be they chipped or hit with topspin,
                    are replied to with the forehand if the return lands in the forehand half of the server's court, in zones A or B.
                    However, more than 80% of chipped returns into C are run around, and hit with the forehand. 
                    Even if a chip lands in D, more than half of the time the server will have time to run around the chip and reply with the forehand.
                """
graphFrms, _ = vs.animate_static(jntLns, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts, num_frames = 25)
graphFrms = overlay(graphFrms, legFrm_chip_tops, top_left_point = tlPt)
script_lines.append(script_line)
all_frames.append(graphFrms)

graphScn, graph_annPnts = myGraph3.preview()
graph_annPnts.pop(2)

jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
jntPts = [*annotated_points, *graph_annPnts]

script_line =   """
                    The data shows that running around returns to the backhand pays off for the server: 
                    if a return to the backhand side can be run around, the returner's win% goes from 48% to 42% in area C, and from 50% to 41% in area D.
                """
graphFrms, _ = vs.animate_static(jntLns, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPts, num_frames = 25)
graphFrms = overlay(graphFrms, legFrm_fore_back, top_left_point = tlPt)
script_lines.append(script_line)
all_frames.append(graphFrms)

######### ARROWS ###########
from ATMANpy.types.ShapesTennis3D import Arrow

def resize_arrow(someArrow, new_length, hat_length = 3, color = const.BGR['strong_blue'], alpha = 1, thickness = 1, gamma = 3 * np.pi / 4, theta = np.pi):
    mainLn = someArrow[0]
    vec, size = mainLn.to_vector(), mainLn.length()
    unit_vec = vec / size
    new_endPt = mainLn.leftPt + (unit_vec * new_length)
    return Arrow.make(someArrow.start_point(), new_endPt, hat_length = hat_length, color = color, alpha = alpha, thickness = thickness, gamma = gamma, theta = theta)

def get_arrows_variable_length(startPt, endPts, length_scales, base_len = 50, color = const.BGR['strong_blue']):
    longArrows = [Arrow.make(startPt, endPt) for endPt in endPts]
    resizedArrows = [resize_arrow(longArrow, length_scale * base_len, color = color) for longArrow, length_scale in zip(longArrows, length_scales)]
    return resizedArrows

def get_arrows_variable_alpha(startPt, endPts, alphas, base_len = 50, color = const.BGR['strong_blue']):
    longArrows = [Arrow.make(startPt, endPt) for endPt in endPts]
    resizedArrows = [resize_arrow(longArrow, base_len, color = color, alpha = alpha) for longArrow, alpha in zip(longArrows, alphas)]
    return resizedArrows


### Where are balls hit given shot area
grpFrm = subFrm.groupby(['return_area', 'third_fore'])['third_area'].value_counts(normalize = True)
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)
cFrm, dFrm = grpFrm.loc['3'], grpFrm.loc['4']

### Collect the Points
cPt = (top_c_3D.p1 + top_c_3D.p2) / 2
dPt = (top_d_3D.p1 + top_d_3D.p2) / 2

from ATMANpy.auxil.court import bot_a, bot_b, bot_c, bot_d
front_px = 220
aPt2 = Point3D.from_point2D(bot_a.centroid(), z = 0).front(front_px) ### move them up a bit so the arrows spread more
bPt2 = Point3D.from_point2D(bot_b.centroid(), z = 0).front(front_px)
cPt2 = Point3D.from_point2D(bot_c.centroid(), z = 0).front(front_px)
dPt2 = Point3D.from_point2D(bot_d.centroid(), z = 0).front(front_px)
endPts = [aPt2, bPt2, cPt2, dPt2]

### Make Arrows
c_alphaFrm = cFrm / cFrm.max().max()
d_alphaFrm = dFrm / dFrm.max().max()
base_len = 35

# a) forehand
alphas = c_alphaFrm[True].values 
fore_ca, fore_cb, fore_cc, fore_cd = get_arrows_variable_alpha(cPt.front(10), endPts, alphas, base_len = base_len, color = fore_color)

alphas = d_alphaFrm[True].values 
fore_da, fore_db, fore_dc, fore_dd = get_arrows_variable_alpha(dPt.front(10), endPts, alphas, base_len = base_len, color = fore_color)

# b) backhand
alphas = c_alphaFrm[False].values 
back_ca, back_cb, back_cc, back_cd = get_arrows_variable_alpha(cPt.back(30), endPts, alphas, base_len = base_len, color = back_color)

alphas = c_alphaFrm[False].values 
back_da, back_db, back_dc, back_dd = get_arrows_variable_alpha(dPt.back(30), endPts, alphas, base_len = base_len, color = back_color)

####### CONTINUE ANIMATING ########

dim_factor = 0.4
def dim_if_annot(annPnt):
    if annPnt.text.replace(' ', '') not in ['0%', '100%', 'Ret.Win%']:
        return annPnt.dim(dim_factor)
    else:
        return annPnt


script_line =   """         
                    Running around a chip to the backhand side opens up the forehand inside-out for the server, 
                    which, for one, targets the returner's backhand, and therefore opens up an advantage for the server; 
                    for two, it goes cross-court, thus allowing a greater margin of error. 

                    Our data shows that on returns that can be run around, the server indeed goes for the cross-court inside-out forehand most frequently:
                    more than half of all run-around backhands are hit back to the returner's D sector (the extreme backhand side).
                """

showArrows = [  
                fore_cd, fore_dd
            ]

jntLns_lo_lum = [l.dim(dim_factor) for l in jntLns]
lines_and_arrows = [*jntLns_lo_lum, *showArrows]

jntPnts_lo_lum = [dim_if_annot(annPnt) for annPnt in jntPts]

graphFrms, _ = vs.animate_static(lines_and_arrows, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPnts_lo_lum, num_frames = 25)
graphFrms = overlay(graphFrms, legFrm_fore_back, top_left_point = tlPt)

graphFrms[-1].show()

script_lines.append(script_line)
all_frames.append(graphFrms)


script_line =   """

                    The second most employed alternative is the more forcing, but higher risk, forehand inside-in, 
                    which is employed around a third of the time on run-around returns; the lower frequency is shown graphically with greater transparence of the arrows.
                """

showArrows = [  
                fore_cd, fore_dd, fore_ca, fore_da
            ]
jntLns_lo_lum = [l.dim(dim_factor) for l in jntLns]
lines_and_arrows = [*jntLns_lo_lum, *showArrows]

jntPnts_lo_lum = [dim_if_annot(annPnt) for annPnt in jntPts]

graphFrms, _ = vs.animate_static(lines_and_arrows, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPnts_lo_lum, num_frames = 25)

graphFrms = overlay(graphFrms, legFrm_fore_back, top_left_point = tlPt)

graphFrms[-1].show()

script_lines.append(script_line)
all_frames.append(graphFrms)



script_line =   """
                    Only very few run-around returns are hit back towards the middle of the court, which is arguably the more passive continuation.
                """
showArrows = [  
                fore_ca, fore_cb, fore_cc, fore_cd,
                fore_da, fore_db, fore_dc, fore_dd,
            ]
jntLns_lo_lum = [l.dim(dim_factor) for l in jntLns]
lines_and_arrows = [*jntLns_lo_lum, *showArrows]

jntPnts_lo_lum = [dim_if_annot(annPnt) for annPnt in jntPts]

graphFrms, _ = vs.animate_static(lines_and_arrows, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPnts_lo_lum, num_frames = 25)

graphFrms = overlay(graphFrms, legFrm_fore_back, top_left_point = tlPt)

graphFrms[-1].show()

script_lines.append(script_line)
all_frames.append(graphFrms)

script_line =   """
                    On the other hand, if the return is NOT run around, the server will not look for aggressive cross-court or down-the-line angles so often,
                    opting for a ball down the middle much more frequently. 
                """

showArrows = [  
                fore_ca, fore_cb, fore_cc, fore_cd,
                fore_da, fore_db, fore_dc, fore_dd,
                back_ca, back_cb, back_cc, back_cd,
                back_da, back_db, back_dc, back_dd
            ]

jntLns_lo_lum = [l.dim(dim_factor) for l in jntLns]
lines_and_arrows = [*jntLns_lo_lum, *showArrows]

jntPnts_lo_lum = [dim_if_annot(annPnt) for annPnt in jntPts]

graphFrms, _ = vs.animate_static(lines_and_arrows, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn_subset,
                                      annotated_points = jntPnts_lo_lum, num_frames = 25)

graphFrms = overlay(graphFrms, legFrm_fore_back, top_left_point = tlPt)

graphFrms[-1].show()

script_lines.append(script_line)
all_frames.append(graphFrms)

show(all_frames)
