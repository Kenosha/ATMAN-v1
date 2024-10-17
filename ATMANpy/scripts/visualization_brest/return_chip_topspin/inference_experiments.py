
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
CHIP_COLOR = const.BGR['marine_blue']
TOPS_COLOR = const.BGR['light_brown']

### Cam and View
from ATMANpy.types.Visual3D import View, Scene, Camera
from ATMANpy.auxil.court import back_view_asymmetric_cam
myCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Bot')
myCam = Camera.calibrated_brest_behind().resize(3)
courtScn = Scene.make_court()

# annotated points
from ATMANpy.auxil.court import top_bsl_deu_2D, top_bsl_ad_2D
fhPt_annot = Point3D.from_point2D(top_bsl_deu_2D, z = 0).front(10).annotate('Forehand')
bhPt_annot = Point3D.from_point2D(top_bsl_ad_2D, z = 0).front(10).left(35).annotate('Backhand')

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

def is_return_topspin(myPnt):
    ### filter for topspin returns so that we can introduce counterfactual.
    return myPnt.get_return().is_topspin_visual()

realDict = {}

# a) deuce, out wide
myFrms = get_spline_frames(bPnt)
# DecapStream(myFrms, 0, 100).show()
realDict['deuce_wide'] = myFrms

# b) deuce, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_deuce').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[0])
DecapStream(myFrms, 0, 100).show()
myFrms[-1].show()
realDict['deuce_middle'] = myFrms

# c) ad, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[41]) ### not nakashima again.
DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
realDict['ad_middle'] = myFrms

# d) ad, out wide
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[100])
DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
myFrms[-1].show()
realDict['ad_wide'] = myFrms

### Get area of confidence for prediction
kdeScn = vs.kde_heatmap(retPnts_in.get_returns(), showReg = vis_ut.topReg, resolution = RESOLUTION_KDE, normalize_density = True, bandwidth = BANDWIDTH_KDE)
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
View.make(myCam, checkScn).snap().show()

### SOME PARAMETERS

### adapt these with reference to descriptive
MODE_CHIP_LO = 11.75
MODE_CHIP_HI = 18.75 
MODE_TOPS = 7.28
DIST_CHIP_LO = 6.25
DIST_CHIP_HI = 8.08
DIST_TOPS = 3.33

### rendering parameters
ALPHA_EXPONENT = 0.5
RENDER_ON_REAL = True
LENIENCY = 0.01
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

traj_means = regFrm_in.groupby('is_chipped')[['ret_speed', 'ret_spin']].mean()
chip_speed, chip_spin = traj_means.loc[True]
xDict['ret_speed'] = np.ones(N) * chip_speed
xDict['ret_spin'] = np.ones(N) * chip_spin

regressors_base = pd.DataFrame(xDict)

# these depend on the launch angle
regressors_angle = {}
for chip_name, chip_mode in {'lo': MODE_CHIP_LO, 'hi': MODE_CHIP_HI}.items():
    xDict = {}
    xDict['launch_angle'] = np.ones(N) * chip_mode
    xDict['ret_dist_to_bh_x_launch_angle'] = rel_rastArr[:, 0] * chip_mode
    xDict['ret_dist_to_bsl_x_launch_angle'] = rel_rastArr[:, 1] * chip_mode
    
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
for chip_name, chip_mode in {'lo': MODE_CHIP_LO, 'hi': MODE_CHIP_HI}.items():
    for serve_case, dist in serve_dist_dict.items():
        side, where = serve_case.split('_')
        xDict = {}
        
        xDict['ret_dist_to_opt_x_launch_angle'] = dist * chip_mode
        xDict['ret_dist_to_opt_sq_x_launch_angle'] = (dist ** 2) * chip_mode

        if side == 'deuce': ### Deuce side is 'forehand' side
            xDict['ret_dist_to_bh_serve_fore_x_launch_angle'] = rel_rastArr[:, 0] * chip_mode
            xDict['ret_dist_to_bh_serve_back_x_launch_angle'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_fore_sq_x_launch_angle'] = rel_rastArr[:, 0] ** 2 * chip_mode
            xDict['ret_dist_to_bh_serve_back_sq_x_launch_angle'] = np.zeros(N)

        if side == 'ad':
            xDict['ret_dist_to_bh_serve_fore_x_launch_angle'] = np.zeros(N)
            xDict['ret_dist_to_bh_serve_back_x_launch_angle'] = rel_rastArr[:, 0] * chip_mode
            xDict['ret_dist_to_bh_serve_fore_sq_x_launch_angle'] = np.zeros(N) 
            xDict['ret_dist_to_bh_serve_back_sq_x_launch_angle'] = rel_rastArr[:, 0] ** 2 * chip_mode

        regressors_angle_case[chip_name + '_' + serve_case] = pd.DataFrame(xDict)

### Models we want to try out
regDict = { 'base_weighted': ['is_chipped'],
            'dist_bsl_weighted': ['is_chipped', 'ret_dist_to_bsl'],
            'dist_bsl_bh_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl'],
            'dist_bsl_bh_sq_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bsl_sq'],
            'dist_bsl_bh_parab_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl',  'ret_dist_to_bsl_bh'],
            'dist_bsl_bh_sq_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq', 'ret_dist_to_bsl_bh'],

            'dist_bsl_bh_opt_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt'],
            'dist_bsl_bh_opt_inter_weighted': [ 'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt',
                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle'],
            'dist_bsl_bh_opt_inter2_weighted': [ 'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt',
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
# needed here: opt_sq; bsl_bh


regDict = {'dist_bsl_bh_side_sq_opt_sq_inter2_weighted':  ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_opt_sq',
                                                        'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                        'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                        'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle'],
            'dist_bsl_bh_side_sq_opt_sq_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle'],
            'dist_bsl_bh_side_sq_opt_sq_speed_inter2_weighted': ['ret_speed', 'ret_dist_to_bsl', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle'],}

### only speed and spin (highest R**2)
regDict = {'dist_bsl_bh_side_sq_opt_sq_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 
                                                            'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle'],
        'dist_bsl_bh_side_sq_opt_sq_parab_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'ret_dist_to_bh_serve_fore_bsl', 'ret_dist_to_bh_serve_back_bsl',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 
                                                            'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle']}

### try with bh_x_launch_angle interactions.
regDict = {'dist_bsl_bh_side_sq_opt_sq_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 
                                                            'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle',
                                                            'ret_dist_to_bh_serve_fore_x_launch_angle', 'ret_dist_to_bh_serve_back_x_launch_angle'],
        'dist_bsl_bh_side_sq_opt_sq_parab_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'ret_dist_to_bh_serve_fore_bsl', 'ret_dist_to_bh_serve_back_bsl',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 
                                                            'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle',
                                                            'ret_dist_to_bh_serve_fore_x_launch_angle', 'ret_dist_to_bh_serve_back_x_launch_angle']}

### all variables, launch_angled
regDict = {'dist_bsl_bh_side_sq_opt_sq_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 
                                                            'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle',
                                                            'ret_dist_to_bh_serve_fore_x_launch_angle', 'ret_dist_to_bh_serve_back_x_launch_angle',
                                                            'ret_dist_to_bh_serve_fore_sq_x_launch_angle', 'ret_dist_to_bh_serve_back_sq_x_launch_angle'],
        'dist_bsl_bh_side_sq_opt_sq_parab_chars_inter2_weighted': ['ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                                            'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                                            'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                                            'ret_dist_to_bh_serve_fore_bsl', 'ret_dist_to_bh_serve_back_bsl',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle', 
                                                            'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle',
                                                            'ret_dist_to_bh_serve_fore_x_launch_angle', 'ret_dist_to_bh_serve_back_x_launch_angle',
                                                            'ret_dist_to_bh_serve_fore_sq_x_launch_angle', 'ret_dist_to_bh_serve_back_sq_x_launch_angle']}

### experimenting with topspin placement also:
# -> need some interactions here!
### all variables, launch_angled
regDict = {'speed_interaction': [   'ret_speed', 'ret_spin', 'launch_angle', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                    'ret_dist_to_bh_serve_fore', 'ret_dist_to_bh_serve_back',
                                    'ret_dist_to_bh_serve_fore_sq', 'ret_dist_to_bh_serve_back_sq',
                                    'speed_x_ret_dist_to_bh_serve_fore', 'speed_x_ret_dist_to_bh_serve_back',
                                    'speed_x_ret_dist_to_bh_serve_fore_sq', 'speed_x_ret_dist_to_bh_serve_back_sq']}

### Result Path
project_results_path = get_folder_struct('results_video')['return_chip_topspin']
current_results_path = maybe_make_folder(project_results_path, f'regressions_' + make_timestamp())

if SAVE_REGS:
    #########################################
    ### REGRESSION FOR LOCATION OPTIMALITY ##
    #########################################
    endog = 'ret_wins_weighted'
    for weight in [None, 'weight_topspin', 'weight_bal_pred']:
        if weight is None:
            var_folder = maybe_make_folder(current_results_path, f'{endog}_none')
        else:
            var_folder = maybe_make_folder(current_results_path, f'{endog}_{weight}')

        for save_name, regressors in regDict.items():
            reg_path = maybe_make_folder(var_folder, save_name)

            glmReg = vis_ut.make_and_save_model(regFrm_in, regressors, save_path = reg_path, save_name = save_name, 
                                                endog_col = endog, weight_col = weight)

            ### get the chip and top predicted average
            X, y = vis_ut.retrieve_X_y(glmReg, regressors)
            meanFrm = X.groupby(regFrm_in['is_chipped'].values).mean()
            chipSer = meanFrm.loc[True]
            topsSer = meanFrm.loc[False]
                
            avg_tops_pred = glmReg.predict(topsSer).values[0]
            avg_chip_pred = glmReg.predict(chipSer).values[0]

            avg_tops_pred = avg_tops_pred - LENIENCY
            
            for angle in ['lo', 'hi']:
                for side in ['ad', 'deuce']:
                    for where in ['wide', 'middle']:
                        ### make players
                        srv_x, srv_y = server_pos_dict[side + '_' + where]
                        ret_x, ret_y = return_pos_dict[side + '_' + where]
                        srvPly = Player.make((srv_x, srv_y))
                        retPly = Player.make((ret_x, ret_y))
                        plyScn = Scene([srvPly, retPly])
                        plyView = View.make(myCam, plyScn, resolution = RESOLUTION_OBJECTS)

                        ### make regressor frame
                        regFrm_base = regressors_base
                        regFrm_angle = regressors_angle[angle]
                        regFrm_case = regressors_case[side + '_' + where]
                        regFrm_angle_case = regressors_angle_case[angle + '_' + side + '_' + where]

                        regressor_frame = pd.concat([regFrm_base, regFrm_angle, regFrm_case, regFrm_angle_case], axis = 1)
                        regressor_frame = regressor_frame[[*regressors, 'const']]

                        ### create the frame
                        colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = avg_tops_pred, 
                                                                                        alpha_exponent = ALPHA_EXPONENT)
                        rastScn = DummyScene([rastArr_3D], [colArr], [alArr])
                        kdeView = View.make(myCam, rastScn, resolution = RESOLUTION_KDE, annotatedPoints = [fhPt_annot, bhPt_annot])

                        if RENDER_ON_REAL:
                            realFrm = realDict[side + '_' + where][-1].dim(DIM_FACTOR)
                            chipFrm = kdeView.snap(over_array = realFrm.array.copy())
                        else:
                            chipFrm = backView.combine(kdeView).combine(plyView).snap()
                                                
                        # chipFrm.show()
                        chipFrm.save_as_png(reg_path, f'{angle}_{side}_{where}_map')

                        ### create the target zone
                        colArr, alArr = vis_ut.get_field_array_from_predictions(glmReg, regressor_frame, relative_to = avg_tops_pred, 
                                                               factor_array = shareArr,
                                                               color = CHIP_COLOR)

                        rastScn = DummyScene([rastArr_3D], [colArr], [alArr])
                        kdeView = View.make(myCam, rastScn, resolution = RESOLUTION_KDE, annotatedPoints = [fhPt_annot, bhPt_annot])

                        if RENDER_ON_REAL:
                            realFrm = realDict[side + '_' + where][-1].dim(DIM_FACTOR)
                            chipFrm = kdeView.snap(over_array = realFrm.array.copy())
                        else:
                            chipFrm = backView.combine(kdeView).combine(plyView).snap()

                        # chipFrm.show()
                        chipFrm.save_as_png(reg_path, f'{angle}_{side}_{where}_target')


reg_name = 'dist_bsl_bh_side_sq_opt_sq_chars_inter2_weighted'
regressors = regDict[reg_name]

endog = 'ret_wins_weighted'
weight = 'weight_bal_pred'

glmReg = vis_ut.make_and_save_model(regFrm_in, regressors, save_path = None, save_name = None, 
                                                endog_col = endog, weight_col = weight)

### get the chip and top predicted average
X, y = vis_ut.retrieve_X_y(glmReg, regressors)
meanFrm = X.groupby(regFrm_in['is_chipped'].values).mean()
chipSer = meanFrm.loc[True]
topsSer = meanFrm.loc[False]
    
avg_tops_pred = glmReg.predict(topsSer).values[0]
avg_chip_pred = glmReg.predict(chipSer).values[0]

avg_tops_pred = avg_tops_pred - LENIENCY

def get_reg_frame(angle, side, where):
    ### make regressor frame
    regFrm_base = regressors_base
    regFrm_angle = regressors_angle[angle]
    regFrm_case = regressors_case[side + '_' + where]
    regFrm_angle_case = regressors_angle_case[angle + '_' + side + '_' + where]

    regressor_frame = pd.concat([regFrm_base, regFrm_angle, regFrm_case, regFrm_angle_case], axis = 1)
    regressor_frame = regressor_frame[[*regressors, 'const']]
    return regressor_frame

### Setup
myRst = FramesRaster(2, 2)
all_frames = []
halfCam = myCam.resize(0.5)
fhPt_annot_half = fhPt_annot.left(50).front(25)
bhPt_annot_half = bhPt_annot.left(15).front(25)

combinations = [('lo', 'deuce', 'wide'), ('lo', 'deuce', 'middle'), ('lo', 'ad', 'wide'), ('lo', 'ad', 'middle')]
locations = [(0, 0), (0, 1), (1, 0), (1, 1)]

for combo, loc in zip(combinations, locations):
    angle, side, where = combo
    i, j = loc

    regressor_frame = get_reg_frame(angle, side, where)
    colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = avg_tops_pred, 
                                                                    alpha_exponent = ALPHA_EXPONENT)
    rastScn = DummyScene([rastArr_3D], [colArr], [alArr])

    ### Animate
    # a) spline scene
    realFrms = realDict[side + '_' + where]
    realFrms_resc = realFrms.rescale(0.5)

    myRst.add(realFrms_resc, i, j)
    rastFrms = myRst.render()
    all_frames.append(rastFrms)

    # b) still scene
    rastFrms = BGRFrames.from_frame(rastFrms[-1], STATIC_FRAMES)
    all_frames.append(rastFrms)

    # c) dim and fade in heatmap
    itplFrms = vs.interpolate_frames(realFrms_resc[-1], realFrms_resc[-1].dim(0.5), num_frames = TRANSITION_FRAMES)
    itplScns = vs.interpolate_dummy_scenes(rastScn.transparify(0), rastScn, TRANSITION_FRAMES)

    myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict, animate_over = itplFrms)

    myRst.add(myFrms, i, j)
    rastFrms = myRst.render()
    all_frames.append(rastFrms)

    # d) add in annotation points
    myFrms, _ = vs.animate_static(Scene([]), annotated_points = [fhPt_annot_half, bhPt_annot_half], num_frames = STATIC_FRAMES, static_cam = halfCam,
                                animate_over = myFrms[-1])
    myRst.add(myFrms, i, j)
    rastFrms = myRst.render()
    all_frames.append(rastFrms)

### Fade in Target Areas
trgRst = FramesRaster(2, 2)

for combo, loc in zip(combinations, locations):
    angle, side, where = combo
    i, j = loc

    realFrms = realDict[side + '_' + where]
    realFrms_resc = realFrms.rescale(0.5)

    regressor_frame = get_reg_frame(angle, side, where)
    colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg, regressor_frame, relative_to = avg_tops_pred, 
                                                                    alpha_exponent = ALPHA_EXPONENT)
    rastScn0 = DummyScene([rastArr_3D], [colArr], [alArr])

    colArr, alArr = vis_ut.get_field_array_from_predictions(glmReg, regressor_frame, relative_to = avg_tops_pred, 
                                                            factor_array = shareArr, color = CHIP_COLOR)
    rastScn1 = DummyScene([rastArr_3D], [colArr], [alArr])

    itplScns = vs.interpolate_dummy_scenes(rastScn0, rastScn1, TRANSITION_FRAMES)    

    myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict, animate_over = realFrms_resc[-1].dim(DIM_FACTOR),
                                    annotated_points = [fhPt_annot_half, bhPt_annot_half])

    trgRst.add(myFrms, i, j)

trgFrms = trgRst.render()
all_frames.append(trgFrms)

jntFrms = BGRFrames.from_frames(all_frames)
DecapStream(jntFrms, 0, 100).show()
jntFrms[-1].show()

show(all_frames)

### Show Reaching Distance for Server --> can he run around?

### empirical run around check.
third_is_foreArr = retPnts_in.apply(vis_ut.is_third_shot_fore, dtype = bool)
ySer = pd.Series(third_is_foreArr, name = 'is_forehand')

# regressors = ['ret_dist_to_bh', 'ret_dist_to_bsl_bh', 'is_chipped', 'launch_angle', 'serve_fore']
# regressors = ['ret_dist_to_bh', 'ret_dist_to_bsl_bh', 'is_chipped', 'launch_angle', 'serve_fore', 'ret_speed']
regressors = ['ret_dist_to_bh', 'is_chipped', 'launch_angle', 'serve_fore']
xFrm = regFrm_in[regressors].astype('float')
xFrm['const'] = 1
xFrm = xFrm.reset_index(drop = True)

glmReg_fore = sm.GLM(ySer, xFrm, family = sm.families.Binomial()).fit()
summary = glmReg_fore.summary()
print(summary)

### Construct Meshgridded regressor dataframe for each case in question
# These are always the same!

def get_reg_frame_fore(angle, is_chipped, side):
    xDict = {}
    xDict['const'] = np.ones(N)

    if is_chipped:
        xDict['is_chipped'] = np.ones(N)
    else:
        xDict['is_chipped'] = np.zeros(N)

    xDict['ret_dist_to_bsl'] = rel_rastArr[:, 1]
    xDict['ret_dist_to_bh'] = rel_rastArr[:, 0]
    xDict['ret_dist_to_bsl_bh'] = rel_rastArr[:, 1] * rel_rastArr[:, 0]
    
    xDict['launch_angle'] = np.ones(N) * angle

    if side == 'deuce':
        xDict['serve_fore'] = np.ones(N)
    elif side == 'ad':
        xDict['serve_fore'] = np.zeros(N)

    regressor_frame = pd.DataFrame(xDict)
    regressor_frame = regressor_frame[[*regressors, 'const']]
    return regressor_frame

regressor_frame = get_reg_frame_fore(MODE_TOPS, False, 'deuce')
colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_fore, regressor_frame, relative_to = 0.5, 
                                                                alpha_exponent = ALPHA_EXPONENT)
rastScn_fore = DummyScene([rastArr_3D], [colArr], [alArr])
View.make(myCam, rastScn_fore).snap().show()

regressor_frame = get_reg_frame_fore(MODE_CHIP_LO, True, 'deuce')
colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_fore, regressor_frame, relative_to = 0.5, 
                                                                alpha_exponent = ALPHA_EXPONENT)
rastScn_fore = DummyScene([rastArr_3D], [colArr], [alArr])
View.make(myCam, rastScn_fore).snap().show()

regressor_frame = get_reg_frame_fore(MODE_CHIP_HI, True, 'deuce')
colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_fore, regressor_frame, relative_to = 0.5, 
                                                                alpha_exponent = ALPHA_EXPONENT)
rastScn_fore = DummyScene([rastArr_3D], [colArr], [alArr])
View.make(myCam, rastScn_fore).snap().show()


regressor_frame = get_reg_frame_fore(MODE_TOPS, False, 'ad')
colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_fore, regressor_frame, relative_to = 0.5, 
                                                                alpha_exponent = ALPHA_EXPONENT)
rastScn_fore = DummyScene([rastArr_3D], [colArr], [alArr])
View.make(myCam, rastScn_fore).snap().show()

regressor_frame = get_reg_frame_fore(MODE_CHIP_LO, True, 'ad')
colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_fore, regressor_frame, relative_to = 0.5, 
                                                                alpha_exponent = ALPHA_EXPONENT)
rastScn_fore = DummyScene([rastArr_3D], [colArr], [alArr])
View.make(myCam, rastScn_fore).snap().show()

regressor_frame = get_reg_frame_fore(MODE_CHIP_HI, True, 'ad')
colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_fore, regressor_frame, relative_to = 0.5, 
                                                                alpha_exponent = ALPHA_EXPONENT)
rastScn_fore = DummyScene([rastArr_3D], [colArr], [alArr])
View.make(myCam, rastScn_fore).snap().show()


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

# def weighted_win_percentage(myFrm):
#     if myFrm['return_is_chipped'].unique()[0]: ### only for chipped returns do we apply the pred weight
#         jnt_weight = myFrm['weight_prd'] * myFrm['weight_bal'] * myFrm['weight_len']
#     else:
#         jnt_weight = myFrm['weight_bal'] * myFrm['weight_len']
    
#     return (myFrm['server_wins'] * jnt_weight).sum() / jnt_weight.sum()

# def weighted_win_percentage(myFrm):
#     if myFrm['return_is_chipped'].unique()[0]: ### only for chipped returns do we apply the pred weight
#         jnt_weight = myFrm['weight_prd'] * myFrm['weight_bal']
#     else:
#         jnt_weight = myFrm['weight_bal']
    
#     return (myFrm['server_wins'] * jnt_weight).sum() / jnt_weight.sum()
   
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
grpFrm = subFrm.groupby(['return_area', 'return_is_chipped']).apply(weighted_win_percentage_chip, attrib = True)
grpFrm = grpFrm.unstack('return_is_chipped')
print(grpFrm)
# -> Win Percentage Diff between Chip and Tops is greatest in 3, lowest in 1.
# -> for choice chips, with all weights: area 1 is almost exactly equal.

### Weighted: What Percent Shots are Hit Forehand?
grpFrm = subFrm.groupby(['return_is_chipped', 'return_area']).apply(weighted_fore_percentage)
grpFrm = grpFrm.unstack('return_is_chipped')
print(grpFrm)
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
grpFrm = subFrm.groupby(['return_area', 'third_fore']).apply(weighted_win_percentage_nochip, attrib = True)
grpFrm = grpFrm.unstack('third_fore')
print(grpFrm)
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





# a -> a --> down the line
# a -> b --> to the middle
# a -> c/d --> cross-court
### and equiv for d

# b -> a --> down the line
# b -> b/c --> to the middle
# b -> d --> cross-court
### and equiv for c


regFrm_in['srv_wins']


### Factors:
# server position (deuce/ad (i.e. fore/back))
# return is chipped
# return landing (a, b, c, d) --> a has to be the forehand side!
# serve + 1 forehand/backhand (is it run around?)
# serve + 1 landing (cross court, middle, dtl)

### Values of Interest
### conditional on factors server position and return landing and return is chipped...
# ... run-around-rate ---> do we run around?
# ... serve + 1 landing frequency --> where do we aim?

### conditional on all factors (? do we need server position here?)
# ... win%





myPnt = retPnts_in[0]
def is_third_shot_cross(myPnt):
    try:
        thirdShot = myPnt[2]
    except IndexError:
        return None
    else:
        return thirdShot.is_cross_court()

retPnts_in.get_



########### TESTING ############

distArr = regressors_case['ad_wide']['ret_dist_to_opt']
realFrm = realDict['ad_wide'][-1]

distArr_norm = (distArr - distArr.min()) / (distArr.max() - distArr.min())
distArr_norm_brdcst = distArr_norm.values[:, np.newaxis]
red, green = np.array(const.BGR['intense_red'])[np.newaxis, :], np.array(const.BGR['leafy_green'])[np.newaxis, :]

colArr = distArr_norm_brdcst * red + (1 - distArr_norm_brdcst) * green
alArr = np.ones_like(colArr[:, [0]])

dummyScn = DummyScene([rastArr_3D], [colArr], [alArr])

View.make(myCam, dummyScn).snap().show()

distArr.min()


srvArr_norm = (srvArr[:, 0] - srvArr[:, 0].min()) / (srvArr[:, 0].max() - srvArr[:, 0].min())
srvArr_norm_brdcst = srvArr_norm[:, np.newaxis]
colArr = srvArr_norm_brdcst * red + (1 - srvArr_norm_brdcst) * green
alArr = np.ones_like(colArr[:, [0]])

dummyScn_srv_heat = DummyScene([rastArr_3D], [colArr], [alArr])

View.make(myCam, dummyScn_srv_heat).snap().show()

colArr = srvArr_norm_brdcst * red + (1 - srvArr_norm_brdcst) * green
alArr = np.ones_like(colArr[:, [0]]) * ((srvArr_norm_brdcst > 0.49) & (srvArr_norm_brdcst < 0.51))

dummyScn_srv_line = DummyScene([rastArr_3D], [colArr], [alArr])

lineView_srv = View.make(myCam, dummyScn_srv_line)
lineView_srv.snap(over_array = realFrm.array.copy()).show()


recArr_norm = (recArr[:, 0] - recArr[:, 0].min()) / (recArr[:, 0].max() - recArr[:, 0].min())
recArr_norm_brdcst = recArr_norm[:, np.newaxis]
colArr = recArr_norm_brdcst * red + (1 - recArr_norm_brdcst) * green
alArr = np.ones_like(colArr[:, [0]])

dummyScn_rec_heat = DummyScene([rastArr_3D], [colArr], [alArr])

View.make(myCam, dummyScn_rec_heat).snap().show()

colArr = recArr_norm_brdcst * red + (1 - recArr_norm_brdcst) * green
alArr = np.ones_like(colArr[:, [0]]) * ((recArr_norm_brdcst > 0.49) & (recArr_norm_brdcst < 0.51))

dummyScn_rec_line = DummyScene([rastArr_3D], [colArr], [alArr])

lineView_rec = View.make(myCam, dummyScn_rec_line)
lineView_rec.snap(over_array = realFrm.array.copy()).show()

ptArr = recArr[((recArr_norm_brdcst > 0.49) & (recArr_norm_brdcst < 0.51))[:, 0]]
x, y = ptArr.mean(axis = 0)
pt = Point3D((x, y, 0))

ptView = View.make(myCam, Scene([pt]))

lineView_rec.combine(ptView).snap(over_array = realFrm.array.copy()).show()