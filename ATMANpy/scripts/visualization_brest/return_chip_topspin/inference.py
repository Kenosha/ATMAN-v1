
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
p_orig, p_target = Point3D.from_point2D(cnt_2D, z = 100), Point3D.from_point2D(top_cnt_2D, z = 0)
heatCam = Camera.from_to(p_orig, p_target, f = 1000, sensor_dims = (960, 540)).zoom(0.5)

halfCam = myCam.resize(0.5)

# annotated points
from ATMANpy.auxil.court import top_bsl_deu_2D, top_bsl_ad_2D
fhPt_annot = Point3D.from_point2D(top_bsl_deu_2D, z = 0).front(10).annotate('Forehand')
bhPt_annot = Point3D.from_point2D(top_bsl_ad_2D, z = 0).front(10).left(35).annotate('Backhand')

fhPt_annot_half = fhPt_annot.left(50).front(25)
bhPt_annot_half = bhPt_annot.left(15).front(25)

prmDict_annot = {**prmDict}
prmDict_annot['annotated_points'] = [fhPt_annot_half, bhPt_annot_half]

backView = View.make(myCam, courtScn, resolution = 1, annotatedPoints = [fhPt_annot, bhPt_annot])

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

def is_return_topspin(myPnt):
    ### filter for topspin returns so that we can introduce counterfactual.
    return myPnt.get_return().is_topspin_visual()

realDict = {}
wideDict = {}

pre_frames = 150

# a) deuce, out wide
myFrms = get_spline_frames(bPnt)
preFrms = get_wide_frames(bPnt, pre_frames = pre_frames)
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['deuce_wide'] = myFrms
wideDict['deuce_wide'] = preFrms

# b) deuce, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_deuce').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[0])
preFrms = get_wide_frames(relPnts[0], pre_frames = pre_frames)
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['deuce_middle'] = myFrms
wideDict['deuce_middle'] = preFrms

# c) ad, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[41]) ### not nakashima again.
preFrms = get_wide_frames(relPnts[41], pre_frames = pre_frames) 
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['ad_middle'] = myFrms
wideDict['ad_middle'] = preFrms

# d) ad, out wide
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[100])
preFrms = get_wide_frames(relPnts[100], pre_frames = pre_frames) 
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['ad_wide'] = myFrms
wideDict['ad_wide'] = preFrms

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
                                    'ret_dist_to_bh_serve_fore_sq_x_launch_angle', 'ret_dist_to_bh_serve_back_sq_x_launch_angle'],}

###### ESTIMATE MODEL ######
reg_name = 'chip_inter_opt_launch_even_more'
regressors = regDict[reg_name]

endog = 'ret_wins_weighted'
weight = 'weight_bal_pred'

glmReg = vis_ut.make_and_save_model(regFrm_in, regressors, save_path = None, save_name = None, 
                                                endog_col = endog, weight_col = weight)
    
########## SETUP #########
myRst = FramesRaster(2, 2)
all_frames = []
script_lines = []
dim_factor = 0.25

postRst = FramesRaster(2, 2)
backFrm = View.make(halfCam, courtScn).snap()


comparison_prob_dict = {'tp': 0.535, ### this way, target areas will have roughly the same size:
                        'lo': 0.57,      ### we then annotate their quality with annotation point
                        'hi': 0.57}

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
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)
    colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                                    alpha_exponent = ALPHA_EXPONENT)
    rastScn0 = DummyScene([rastArr_3D], [colArr], [alArr])
    heatDict[angle] = rastScn0

    colArr, alArr = vis_ut.get_field_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                            factor_array = shareArr, color = color)
    rastScn1 = DummyScene([rastArr_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1


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
                    The best topspin return is hit cross-court, according to the model.
                    MAKE A NOTE ABOUT FOREHAND RECTIFICATION!
                """
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo'])
script_line = "As for the low chip, the model likes a shallower chip that lands inside the court at an acute cross-court angle."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg2Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi'])
script_line = "The model does not recommend playing a high chip in this situation."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = courtScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot)

postFrms = BGRFrames.from_frames([transFrms, moveFrms])
postRst.add(postFrms, i, j)


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
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)
    colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                                    alpha_exponent = ALPHA_EXPONENT)
    rastScn0 = DummyScene([rastArr_3D], [colArr], [alArr])
    heatDict[angle] = rastScn0

    colArr, alArr = vis_ut.get_field_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                            factor_array = shareArr, color = color)
    rastScn1 = DummyScene([rastArr_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1


script_line =   """
                    Consider now a serve down the middle, on the deuce side.
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
script_line = "On serves down the middle, the model prefers that the returner targets the backhand of the server with a topspin serve"
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo'])
script_line = "As for the low chip, the model likes a shallower chip that lands inside the court at an acute cross-court angle."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi'])
script_line = "The model does not recommend playing a high chip in this situation."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = courtScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot)

postFrms = BGRFrames.from_frames([transFrms, moveFrms])
postRst.add(postFrms, i, j)



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
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)
    colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                                    alpha_exponent = ALPHA_EXPONENT)
    rastScn0 = DummyScene([rastArr_3D], [colArr], [alArr])
    heatDict[angle] = rastScn0

    colArr, alArr = vis_ut.get_field_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                            factor_array = shareArr, color = color)
    rastScn1 = DummyScene([rastArr_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1


script_line =   """
                    Consider now a serve down the middle, on the ad side.
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
script_line = "On serves down the middle, the model prefers that the returner targets the backhand of the server with a topspin serve"
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo'])
script_line = "As for the low chip, the model likes a shallower chip that lands inside the court at an acute cross-court angle."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi'])
script_line = "The model does not recommend playing a high chip in this situation."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = courtScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot)

postFrms = BGRFrames.from_frames([transFrms, moveFrms])
postRst.add(postFrms, i, j)


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
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)
    colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                                    alpha_exponent = ALPHA_EXPONENT)
    rastScn0 = DummyScene([rastArr_3D], [colArr], [alArr])
    heatDict[angle] = rastScn0

    colArr, alArr = vis_ut.get_field_array_from_predictions(glmReg, regressor_frame, relative_to = comparison_prob, 
                                                            factor_array = shareArr, color = color)
    rastScn1 = DummyScene([rastArr_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1


script_line =   """
                    Consider now a serve down the middle, on the ad side.
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
script_line = "On serves down the middle, the model prefers that the returner targets the backhand of the server with a topspin serve"
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo'])
script_line = "As for the low chip, the model likes a shallower chip that lands inside the court at an acute cross-court angle."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi'])
script_line = "The model does not recommend playing a high chip in this situation."
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = realFrms_resc[-1].dim(dim_factor))
myRst.add(myFrms, i, j)
rastFrms = myRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

lsFrm = myFrms[-1]
myRst.add(lsFrm, i, j)

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms_resc[-1].dim(dim_factor), backFrm, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = courtScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot)

postFrms = BGRFrames.from_frames([transFrms, moveFrms])
postRst.add(postFrms, i, j)

####### POST-STUFF #######

script_line = ""
postFrms = postRst.render()
all_frames.append(postFrms)
script_lines.append(script_line)

show(all_frames)
