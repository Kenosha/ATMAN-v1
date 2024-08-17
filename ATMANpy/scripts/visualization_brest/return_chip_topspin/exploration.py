
import numpy as np

import ATMANpy.auxil.constants as const

from ATMANpy.funcs.util_funcs import get_folder_struct
from ATMANpy.types.Abstract import Points, Point, read_in_all_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature

#### READ IN CONFIGS ####
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import collect_video_frames

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature


### READ IN DATA
allTable = read_in_all_tables(confs)
allPnts = Points.from_datatable(allTable, split_by_match = False)
firstPnts = allPnts.filter('is_first_serve').enforce_direction('Bot') ### Server is at the Bottom, Receiver at the Top

### Remove aces, faults and net balls.
def get_returned_balls(myPnt):
    return len(myPnt) > 1
all_retPnts = firstPnts.filter(get_returned_balls)

### filter pathological balls
# a) serve speed too low!
# b) no bounce recorded!
# c) too many frames!
def speed_is_above_threshold(myPnt, threshold = 140):
    return myPnt.get_serve().get_initial_speed() > threshold

def speed_is_below_threshold(myPnt, threshold = 140):
    return myPnt.get_serve().get_initial_speed() <= threshold

def too_many_frames(myPnt, threshold = 35):
    return myPnt.get_serve().get_num_frames() > threshold

def no_bounce(myPnt):
    x, y = myPnt.get_serve().get_first_bounce_position()
    return x is None and y is None

def is_not_pathological(myPnt):
    return speed_is_above_threshold(myPnt) and not too_many_frames(myPnt) and not no_bounce(myPnt)

def is_pathological(myPnt):
    return not is_not_pathological(myPnt)

retPnts = all_retPnts.filter(is_not_pathological)
pathPnts = all_retPnts.filter(is_pathological)

# for pnt in retPnts[-10:]:
#     idx = pnt.get_match_index()
#     cap = confs[idx]['cap']
#     myStm = pnt.get_serve().stream_ball(cap, P)
#     myStm.show(ms_between_frames = 100)


### TODO: Have to filter out chipped returns on the stretch -> compare only returns which are chipped by choice!
### But then, on the other hand -> if we try to match them to topspin returns, we won't find any, since on the stretch, most returns are blocked!
### -> automatically reduced!!

### 1st Try: Propensity Score
myPnt = retPnts[0]

courtLines = CourtLines.make()
x, y = courtLines['Top Baseline'].center().coords ### note that direction has been enforced -> Bot to Top
x, y = x / const.SCALE * const.FOOT_TO_METER, y / const.SCALE * const.FOOT_TO_METER

### Dependent Variable: chipped or topspin?
def is_chipped(myPnt):
    return myPnt.get_return().is_slice_visual()
chipArr = retPnts.apply(is_chipped)

### Independent Variables:
# a) Serve Speed
def get_serve_speed(myPnt):
    return myPnt.get_serve().get_initial_speed()
speedArr = retPnts.apply(get_serve_speed)

# b) Location of Return
def get_ret_dist_to_bsl(myPnt):
    return myPnt.get_serve().get_ball_next_position()
locArr = retPnts.apply(get_ret_dist_to_bsl) / const.SCALE * const.FOOT_TO_METER

# b1) Width of the return -> relative to center line
widthArr = np.abs(locArr[:, 0] - x) ### absolute, because we control for side by looking at forehand/backhand
# b2) Depth of return
depthArr = -(locArr[:, 1] - y) ### negative -> behind baseline, positive, in front -> so we turn it around!!
# b3) Height of Return
heightArr = locArr[:, 2]

# c) Distance of Returner from Flight Line
def distance_from_flight_line(myPnt):
    srvShot = myPnt.get_serve()
    retPnt = Point2D(srvShot.get_player_rec_position())
    fsPnt = Point2D(srvShot.get_ball_hit_position()[:2])
    lsPnt = Point2D(srvShot.get_ball_last_position()[:2])
    flightLn = Line(fsPnt, lsPnt)
    return flightLn.distance_to(retPnt) ### perpendicular distance to the flight line of the returner
distArr = retPnts.apply(distance_from_flight_line) / const.SCALE * const.FOOT_TO_METER

# d) Forehand/Backhand
def is_fore(myPnt):
    return myPnt.get_return().is_fore()
foreArr = retPnts.apply(is_fore)

# e) spin
def get_serve_spin(myPnt):
    return myPnt.get_serve().get_initial_spin()
spinArr = retPnts.apply(get_serve_spin)
# spinArr = spinArr.clip(min = -1000, max = 5000) ### take it easy on the backspin... no such serve

varDict = {'speed': speedArr,
           'spin': spinArr,
           'width': widthArr,
           'depth': depthArr,
           'height': heightArr,
           'dist': distArr,
           'is_fore': foreArr}

### Logistic Regression 
import pandas as pd
from statsmodels.discrete.discrete_model import Logit

X = np.stack(list(varDict.values()), axis = 1)
y = chipArr

xFrm = pd.DataFrame(X, columns = varDict.keys())
ySer = pd.Series(y, name = 'is_chipped')

logReg = Logit(ySer, xFrm)
logReg = logReg.fit()
logReg.summary()

predSer = pd.Series(logReg.predict(), name = 'pred_is_chipped')

logReg.predict([1, 2, 2, 1, 1, 1, 1])

#################################
####### Who Wins the Point? #####
#################################
def get_winners_from_score(retPnts, allPnts):
    winner_codes = []
    for myPnt in retPnts:
        match_idx, point_idx = myPnt.get_match_index(), myPnt.get_point_index()
        nextPnt = allPnts.get(match_idx, point_idx + 1)

        winner = myPnt.get_winner_from_score(nextPnt, verbose = True)

        if winner is None:
            winner_codes.append(0)
        elif winner == myPnt.get_server().get_name():
            winner_codes.append(1)
        elif winner == myPnt.get_returner().get_name():
            winner_codes.append(-1)

    return winner_codes

winner_codes = get_winners_from_score(retPnts, allPnts)

winSer = pd.Series(winner_codes, name = 'winner_code')

regFrm = pd.concat([xFrm, ySer, winSer, predSer], axis = 1)

#########################################
##### CREATE REGRESSORS AND WEIGHTS #####
#########################################

### length weight
DECAY_RATE = 0.75
regFrm['len'] = retPnts.apply('__len__')
# regFrm['len_weight'] = (1 / (regFrm['len'] - 1)) ** 0.5
regFrm['len_weight'] = DECAY_RATE ** (regFrm['len'] - 2)

### winner vars
regFrm['ret_wins'] = regFrm['winner_code'] == -1
regFrm['srv_wins'] = regFrm['winner_code'] == 1

regFrm['is_chipped'] = regFrm['is_chipped'].astype(int)
regFrm['odds_topspin'] = (1 - regFrm['pred_is_chipped']) / regFrm['pred_is_chipped']
regFrm['weight_topspin'] = np.where(regFrm.is_chipped, regFrm.odds_topspin, 1)
regFrm['weight_ret_wins'] = regFrm['ret_wins'] * regFrm['weight_topspin']
regFrm['weight_len_pred'] = regFrm['len_weight'] * regFrm['weight_topspin']
regFrm['weight_jnt_ret_wins'] = regFrm['ret_wins'] * regFrm['weight_len_pred']

def bounce_is_not_none(myShot):
    x, y = myShot.get_first_bounce_position()
    return x is not None and y is not None
regFrm['did_bounce'] = retPnts.get_returns().apply(bounce_is_not_none)

regFrm_clear = regFrm[regFrm.winner_code != 0]

regFrm_clear.groupby('is_chipped')['ret_wins'].mean() 
######## NAIVELY, THE WIN PROBABILITY ON A CHIPPED RETURN IS ALMOST HALF THAT OF A TOPSPIN RETURN ... GAP: 18%

regFrm_clear.groupby('is_chipped')['weight_ret_wins'].sum() / regFrm_clear.groupby('is_chipped')['weight_topspin'].sum()
####### ... BUT CONTROLLING FOR ABILITY TO CHOOSE, THE WIN PROBABILITY FOR A CHIPPED RETURN INCREASES! GAP: 13%

regFrm_clear.groupby('is_chipped')['weight_jnt_ret_wins'].sum() / regFrm_clear.groupby('is_chipped')['weight_len_pred'].sum()
####### WEIGHTING SHORT EXCHANGES HIGHER (HERE, THE RETURN IS MORE LIKELY TO BE OF IMPORTANCE), THE GAP DECREASES TO 8%

##################################
### Heatmap of Return Location ###
##################################

from ATMANpy.types.Visual3D import View, Scene, Camera
import ATMANpy.funcs.vis_funcs as vs
from ATMANpy.auxil.court import back_view_asymmetric_cam

myCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Bot')
retShots = retPnts.get_returns()
botReg = courtLines.add_singles_netline().get_half_region(side = 'Bottom', width = 'Singles')
showReg = botReg.scale(1.25)
courtScn = Scene.make_court()
backView = View.make(myCam, courtScn, resolution = 1)

def neither_slice_nor_topspin(myShot):
    return not myShot.is_slice_visual() and not myShot.is_topspin_visual()

##### WEIRD !!! ####
weirdShots = retShots.filter(neither_slice_nor_topspin)
for shot in weirdShots:
    print(shot.get_hit_event_descriptor())
######################

chipFrm = regFrm[regFrm.is_chipped == 1]
topspin_preds = 1 - chipFrm.pred_is_chipped.values
topspin_odds = (1 - chipFrm.pred_is_chipped.values) / chipFrm.pred_is_chipped.values

retShots_lh_server = retShots.filter(lambda shot: shot.recPlayer.get_hand() == 'left').flip(across_middle = True, across_net = False)
retShots_rh_server = retShots.filter(lambda shot: shot.recPlayer.get_hand() == 'right')
retShots_rectified = retShots_lh_server.join(retShots_rh_server).sort()

chip_retShots = retShots.filter('is_slice_visual')
tops_retShots = retShots.filter('is_topspin_visual')

chip_retShots_rectified = retShots_rectified.filter('is_slice_visual')
tops_retShots_rectified = retShots_rectified.filter('is_topspin_visual')

### Topspin Heatmaps
topsScn = vs.kde_heatmap(tops_retShots, 
                         showReg = showReg, 
                         position_func = 'get_first_bounce_position', 
                         bandwidth = 45)

kdeView = View.make(myCam, topsScn, resolution = 1)
topsFrm = backView.combine(kdeView).snap()
topsFrm.show()

topsScn = vs.kde_heatmap(tops_retShots_rectified, 
                         showReg = showReg, 
                         position_func = 'get_first_bounce_position', 
                         bandwidth = 45)

kdeView = View.make(myCam, topsScn, resolution = 1)
topsFrm_rect = backView.combine(kdeView).snap()
topsFrm_rect.show()

topsFrm.show()
topsFrm_rect.show()
(topsFrm - topsFrm_rect).show()

### Chip Heatmaps
chipScn = vs.kde_heatmap(chip_retShots,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = topspin_preds,
                        normalize_density = True,
                        bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = topspin_preds,
                        normalize_density = True,
                        bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm_rect = backView.combine(kdeView).snap()
chipFrm_rect.show()

chipFrm.show()
chipFrm_rect.show()
(chipFrm - chipFrm_rect).show()

######################################
####### COMPARING CLUSTERS ###########
######################################

topsScn, tops_comps = vs.kde_contourmap(tops_retShots_rectified,  showReg = showReg,
                                        position_func = 'get_first_bounce_position', 
                                        normalize_density = True,
                                        bandwidth = 45,
                                        display_at_height = False,
                                        n_components = 1)

kdeView = View.make(myCam, topsScn, resolution = 1)
topsFrm = backView.combine(kdeView).snap()
topsFrm.show()

chipScn, chip_comps = vs.kde_contourmap(chip_retShots_rectified,  showReg = showReg,
                                        position_func = 'get_first_bounce_position', 
                                        weight_func_or_arr = topspin_preds,
                                        normalize_density = True,
                                        bandwidth = 45, 
                                        display_at_height = False,
                                        n_components = 2)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

################################
#### RETURN DEPTH VS WIN PROB ##
################################

def get_ret_dist_to_bsl(myPnt):
    retShot = myPnt.get_return()

    botLn = courtLines['Bottom Baseline']
    bot_y = botLn.leftPt.y
    if retShot is None:
        return None
    
    ret_y = retShot.get_first_bounce_position()[1]
    if ret_y is None:
        return None

    dist = bot_y - ret_y
    dist_in_meters = dist / const.SCALE * const.FOOT_TO_METER
    return dist_in_meters

def get_ret_dist_to_bh(myPnt):
    srvShot = myPnt.get_serve()
    retShot = myPnt.get_return()
    botLn = botReg['Bottom']

    if srvShot.hitPlayer.get_hand() == 'right':
        bhPnt = botLn.leftPt
    elif retShot.recPlayer.get_hand() == 'left':
        bhPnt = botLn.rightPt
    else:
        return None

    bh_x = bhPnt.x

    if retShot is None:
        return None
    
    ret_x = retShot.get_first_bounce_position()[0]
    if ret_x is None:
        return None

    dist = ret_x - bh_x
    dist_in_meters = dist / const.SCALE * const.FOOT_TO_METER
    return dist_in_meters

regFrm['ret_dist_to_bsl'] = retPnts.apply(get_ret_dist_to_bsl)
regFrm['ret_dist_to_bh'] = retPnts.apply(get_ret_dist_to_bh)

import statsmodels.api as sm

def get_X_y_w(regFrm, exog, endog, weight_col = 'weight_topspin', add_constant = True):
    regFrm_clear = regFrm[regFrm.winner_code != 0]
    regFrm_clear_sub = regFrm_clear[[*exog, endog, weight_col]]

    nan_filter = ~regFrm_clear_sub.isna().any(axis = 1)
    regFrm_valid = regFrm_clear_sub[nan_filter]

    xFrm, yFrm = regFrm_valid[exog], regFrm_valid[endog]
    if add_constant:
        xFrm['const'] = 1
    
    wFrm = regFrm_valid[weight_col]

    return xFrm, yFrm, wFrm

####### BASE TEST ######
X, y, w = get_X_y_w(regFrm, ['is_chipped'], 'ret_wins', add_constant = True)
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial()).fit()
print(glmReg_win.predict([1, 1]), glmReg_win.predict([0, 1]))

glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
print(glmReg_win.predict([1, 1]), glmReg_win.predict([0, 1]))

### WITH PREDICTION WEIGHTS ########
X, y, w = get_X_y_w(regFrm, ['is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True)
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
glmReg_win.summary()

X, y, w = get_X_y_w(regFrm, ['is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True)
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
glmReg_win.summary()
print(glmReg_win.predict([1, 0, 1]), glmReg_win.predict([0, 2, 1]))

####### WITH COMBINED WEIGHTS #####
X, y, w = get_X_y_w(regFrm, ['is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
glmReg_win.summary()

### find average distance to baseline for a topspin ball
mean_topspin_dist_to_bsl = X.loc[X.is_chipped == 0, 'ret_dist_to_bsl'].mean()

avg_topspin_pred = glmReg_win.predict([0, mean_topspin_dist_to_bsl, 1])
print(avg_topspin_pred)

for chip_dist_to_bsl in [0, 1, 2, 3, 4, 5]:
    print(glmReg_win.predict([1, chip_dist_to_bsl, 1]))

X, y, w = get_X_y_w(regFrm, ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_bh'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
glmReg_win.summary()

mean_topspin_dist_to_bsl = X.loc[X.is_chipped == 0, 'ret_dist_to_bsl'].mean()
mean_topspin_dist_to_bh = X.loc[X.is_chipped == 0, 'ret_dist_to_bh'].mean()

avg_topspin_pred = glmReg_win.predict([0, mean_topspin_dist_to_bsl, mean_topspin_dist_to_bh, 1])
print(avg_topspin_pred)

import ATMANpy.funcs.util_funcs as ut
prod = ut.cartesian_product(np.arange(5), np.arange(5))

for dists in prod:
    print(dists, glmReg_win.predict([1, *dists, 1]))

### Change the chipScn to reflect win probability over topspin average
chipScn = vs.kde_heatmap(chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = topspin_preds,
                        normalize_density = True,
                        bandwidth = 45)

bhPt = botReg['Bottom'].leftPt
rel_rastArr = np.copy(chipScn.rastArrs[0])
rel_rastArr[:, 0] = rel_rastArr[:, 0] - bhPt.x
rel_rastArr[:, 1] = bhPt.y - rel_rastArr[:, 1] 
rel_rastArr = rel_rastArr[:, :2]
rel_rastArr = rel_rastArr / const.SCALE * const.FOOT_TO_METER

ones = np.ones((len(rel_rastArr), 1))
regArr = np.concatenate([ones, rel_rastArr, ones], axis = 1)
pred_winArr = glmReg_win.predict(regArr)
pred_winArr = pred_winArr[:, np.newaxis]

pred_winArr_relative = pred_winArr - avg_topspin_pred
pred_winArr_relative_renorm = (pred_winArr_relative - pred_winArr_relative.min()) / (pred_winArr_relative.max() - pred_winArr_relative.min())

colArr = pred_winArr_relative_renorm * const.BGR['leafy_green'] + (1 - pred_winArr_relative_renorm) * const.BGR['intense_red']

chipScn.colArrs[0] = colArr


kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()


#### LOOK AT BOUNCES AND OUTCOMES DESCRIPTIVELY ####
colArr = (regFrm.loc[regFrm.is_chipped == 1, 'winner_code'] + 1).values

markScn = vs.scatter_markmap(chip_retShots_rectified, color_func_or_arr = colArr, 
                             zero_color = const.BGR['leafy_green'], one_color = const.BGR['intense_red'])
markFrm = View.make(myCam, markScn, resolution = 1).snap()
markFrm.show()

# X, y, w = get_X_y_w(regFrm, [*list(varDict.keys()), 'is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
# glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
# glmReg_win.summary()

# X, y, w = get_X_y_w(regFrm, ['is_fore', 'speed', 'dist', 'is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
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

def epanechnikov_one_dim(values, space, weightArr = None, bandwidth = 10, normalize_density = True):
    diffArr = space[np.newaxis] - values[:, np.newaxis]
    norm_diffArr = diffArr / bandwidth
    norm_devArr = norm_diffArr ** 2

    kde_values_for_each_kernel = (1 - norm_devArr) * (norm_devArr < 1)

    if weightArr is not None:
        kde_values_for_each_kernel = kde_values_for_each_kernel * weightArr[:, np.newaxis]

    kde_values = kde_values_for_each_kernel.mean(axis = 0)
    
    if normalize_density:
        kde_values = kde_values / kde_values.max()

    return kde_values

def detect_distribution_modes(kdeArr, spaceArr, min_rel_mode_height = 0.5):
    loc_max_idxs = ut.detect_local_minima(-kdeArr)
    max_angles, max_heights = spaceArr[loc_max_idxs], kdeArr[loc_max_idxs]

    glob_max = max_heights.max()
    is_valid = max_heights > min_rel_mode_height * glob_max # is above min height

    max_angles_valid = max_angles[is_valid]
    return max_angles_valid

from ATMANpy.types.Shapes3D import Point3D, Line3D, Lines3D
someLn = courtLines['Top Baseline']
someLn3D = Line3D.from_line2D(someLn, z = 0)

bandwidth = 10
resolution = 10
def kde_angle_distribution(myShots, value_func = 'get_launch_angle', weight_func_or_arr = None, 
                            min_angle = 0, max_angle = 90, resolution = 1,
                            axis_origin_point = None, axis_length = 100, x_unit_vector = None, y_unit_vector = None, show_axes = True,
                            bandwidth = 5, color = const.BGR['intense_red'],
                            min_rel_mode_height = 0.5):

    spaceArr = np.arange(min_angle, max_angle, step = 1 / resolution)

    ### Get angle values
    if value_func is None:
        angleArr = np.arange(min_angle - 45, max_angle + 45, step = 1) #### this just makes the unit circle!
    elif isinstance(value_func, str):
        angleArr = getattr(myShots, value_func)()
    else:
        angleArr = np.array([value_func(aShot) for aShot in myShots], dtype = float)
    
    is_not_nan = ~np.isnan(angleArr)
    angleArr_nonnan = angleArr[is_not_nan]

    ### figure out if weights need to be applied ###
    if weight_func_or_arr is None:
        weightArr = None
    elif isinstance(weight_func_or_arr, np.ndarray):
        weightArr = weight_func_or_arr[is_not_nan]
    elif isinstance(weight_func_or_arr, str):
        weightArr = getattr(myShots, weight_func_or_arr)()[is_not_nan]
    else:
        weightArr = np.array([weight_func_or_arr(aShot) for aShot in myShots], dtype = float)[is_not_nan]

    if value_func is None:
        kdeArr = np.ones_like(spaceArr)
    else:
        ### get the classic, one dimensional epanechnikov kde on a line
        kdeArr = epanechnikov_one_dim(angleArr_nonnan, spaceArr, weightArr = weightArr, bandwidth = bandwidth)

    modes = detect_distribution_modes(kdeArr, spaceArr, min_rel_mode_height = min_rel_mode_height)

    ### now, wrap the KDE values around the angle, on a flat unit circle
    space_radArr = spaceArr * const.ANGLE_TO_RAD
    x, y = np.cos(space_radArr) * kdeArr, np.sin(space_radArr) * kdeArr

    ### transform to the 3D coordinates given in kwargs
    if axis_origin_point is None:
        axis_origin_point = Point3D.from_point2D(CourtLines.make().get_center_point(), z = 0)
    if x_unit_vector is None:
        x_unit_vector = Point3D((1, 0, 0))
    if y_unit_vector is None:
        y_unit_vector = Point3D((0, 1, 0))

    x_unit_norm = x_unit_vector * axis_length
    y_unit_norm = y_unit_vector * axis_length

    x3D = x_unit_norm * x
    y3D = y_unit_norm * y

    kdePnt = x3D + y3D + axis_origin_point

    kdeScn = Scene([kdePnt.recolor(color)])

    if show_axes:
        x_axis = Line3D(axis_origin_point, axis_origin_point + x_unit_norm)
        y_axis = Line3D(axis_origin_point, axis_origin_point + y_unit_norm)
        kdeScn = kdeScn.put(x_axis).put(y_axis)
        
    return kdeScn, modes

kwargDict = {'bandwidth': 2.5,
             'resolution': 10,
             'axis_length': 100,
             'axis_origin_point': someLn3D.leftPt,
             'x_unit_vector': Point3D((1, 0, 0)), 
             'y_unit_vector': Point3D((0, 0, 1))}

# UNIT CIRCLE 
angleScn, _ = kde_angle_distribution(chip_retShots_rectified, **kwargDict, value_func = None)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

# CHIP LAUNCH ANGLES
angleScn, _ = kde_angle_distribution(chip_retShots_rectified, **kwargDict)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

weightArr = regFrm.loc[regFrm.is_chipped == 1, 'weight_topspin'].values
weightArr_preds = 1 - regFrm.loc[regFrm.is_chipped == 1, 'pred_is_chipped'].values

angleScn, _ = kde_angle_distribution(chip_retShots_rectified, weight_func_or_arr = weightArr, **kwargDict)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

angleScn, _ = kde_angle_distribution(chip_retShots_rectified, weight_func_or_arr = weightArr_preds, **kwargDict)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

# TOPSPIN LAUNCH ANGLES 
angleScn, _ = kde_angle_distribution(tops_retShots_rectified, **kwargDict)
angleView = View.make(myCam, angleScn)

angleFrm = backView.combine(angleView).snap()
angleFrm.show()

# CHIP LAUNCH ANGLES ONLY BOUNCING
weightArr = regFrm.loc[(regFrm.is_chipped == 1) & (regFrm.did_bounce == 1), 'weight_topspin'].values
weightArr_preds = 1 - regFrm.loc[(regFrm.is_chipped == 1) & (regFrm.did_bounce == 1), 'pred_is_chipped'].values

angleScn, _ = kde_angle_distribution(chip_retShots_rectified.filter(bounce_is_not_none), **kwargDict)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

angleScn, _ = kde_angle_distribution(chip_retShots_rectified.filter(bounce_is_not_none), weight_func_or_arr = weightArr, **kwargDict)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

angleScn, _ = kde_angle_distribution(chip_retShots_rectified.filter(bounce_is_not_none), weight_func_or_arr = weightArr_preds, **kwargDict)
angleView = View.make(myCam, angleScn)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

# TOPSPIN LAUNCH ANGLES ONLY BOUCNING
angleScn, _ = kde_angle_distribution(tops_retShots_rectified.filter(bounce_is_not_none), **kwargDict)
angleView = View.make(myCam, angleScn)

angleFrm = backView.combine(angleView).snap()
angleFrm.show()

tops_retShots_rectified.filter(bounce_is_not_none).get_launch_angle().clip(0, 90).mean()
chip_retShots_rectified.filter(bounce_is_not_none).get_launch_angle().clip(0, 90).mean()

### VISUALIZE MAPPING OF LAUNCH ANGLE ####

# a) map from unit circle to angle kde
unitScn, _ = kde_angle_distribution(chip_retShots_rectified.filter(bounce_is_not_none), **kwargDict, value_func = None)
angleScn, chip_modes = kde_angle_distribution(chip_retShots_rectified.filter(bounce_is_not_none), weight_func_or_arr = weightArr_preds, **kwargDict)

itplScns = vs.interpolate_scenes(unitScn, angleScn, 25)
myFrms, _ = vs.animate_sequence(itplScns)
DecapStream(myFrms, 0, 100).show()

mode_lines = []
for mode_idx in [0, 4]:
    ang = chip_modes[mode_idx]
    rad = ang * const.ANGLE_TO_RAD
    line_len, x_ax, y_ax, orig = kwargDict['axis_length'], kwargDict['x_unit_vector'], kwargDict['y_unit_vector'], kwargDict['axis_origin_point']
    mode_x, mode_y = np.cos(rad) * line_len, np.sin(rad) * line_len
    mode_vec = x_ax * mode_x + y_ax * mode_y
    modeLn = Line3D(orig, mode_vec + orig)
    mode_lines.append(modeLn)


angleScn_with_line = angleScn.put(Lines3D(mode_lines))
angleView = View.make(myCam, angleScn_with_line)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

angleScn, tops_modes = kde_angle_distribution(tops_retShots_rectified.filter(bounce_is_not_none), **kwargDict)

itplScns = vs.interpolate_scenes(unitScn, angleScn, 25)
myFrms, _ = vs.animate_sequence(itplScns)
DecapStream(myFrms, 0, 100).show()

mode_idx = 0
ang = tops_modes[mode_idx]
rad = ang * const.ANGLE_TO_RAD
line_len, x_ax, y_ax, orig = kwargDict['axis_length'], kwargDict['x_unit_vector'], kwargDict['y_unit_vector'], kwargDict['axis_origin_point']
mode_x, mode_y = np.cos(rad) * line_len, np.sin(rad) * line_len
mode_vec = x_ax * mode_x + y_ax * mode_y
modeLn = Line3D(orig, mode_vec + orig)

angleScn_with_line = angleScn.put(modeLn)
angleView = View.make(myCam, angleScn_with_line)
angleFrm = backView.combine(angleView).snap()
angleFrm.show()

# b) map from line to circle 
pass

##################################################
### DOES LAUNCH ANGLE CORRELATE WITH PLACEMENT ###
##################################################

thresh_val = chip_modes[[0, 4]].mean()

regFrm['launch_angle'] = retPnts.get_returns().apply('get_launch_angle')
weightArr_preds_hi = 1 - regFrm.loc[(regFrm.is_chipped == 1) & (regFrm.launch_angle > thresh_val) , 'pred_is_chipped'].values
weightArr_preds_lo = 1 - regFrm.loc[(regFrm.is_chipped == 1) & (regFrm.launch_angle < thresh_val) , 'pred_is_chipped'].values

hi_chip_retShots_rectified = chip_retShots_rectified.filter('get_launch_angle', is_above = thresh_val)
lo_chip_retShots_rectified = chip_retShots_rectified.filter('get_launch_angle', is_below = thresh_val)

### Chip Heatmaps
chipScn = vs.kde_heatmap(hi_chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = weightArr_preds_hi,
                        normalize_density = True,
                        bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

### Chip Heatmaps
chipScn = vs.kde_heatmap(lo_chip_retShots_rectified,  showReg = showReg,
                        position_func = 'get_first_bounce_position', 
                        weight_func_or_arr = weightArr_preds_lo,
                        normalize_density = True,
                        bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

#######################################
### LAUNCH ANGLE IN A SPECIFICATION ###
#######################################

X, y, w = get_X_y_w(regFrm, ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_bh', 'launch_angle'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
glmReg_win.summary()

regFrm['ret_dist_to_bsl_x_launch_angle'] = regFrm['ret_dist_to_bsl'] * regFrm['launch_angle']

X, y, w = get_X_y_w(regFrm, ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_bh', 
                             'launch_angle', 'ret_dist_to_bsl_x_launch_angle'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
glmReg_win.summary()
### ret_dist_to_bsl_x_launch_angle coeff should be negative -> high distance, high launch is a sitter -> easy to put away!

X, y, w = get_X_y_w(regFrm, ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_bh', 
                             'ret_dist_to_bsl_x_launch_angle'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
glmReg_win.summary()

################################
####### A LOOK AT INTENTION ####
################################

# idea: chip location leans heavily to the backhand side of the server -> harder to attack!
# --> can we see this when comparing LH servers and righthanded servers?
def server_is_right_handed(myPnt):
    return myPnt.get_server().get_hand() == 'right'

def server_is_left_handed(myPnt):
    return myPnt.get_server().get_hand() == 'left'

regFrm['is_right_handed_server'] = retPnts.apply(server_is_right_handed)
regFrm['is_left_handed_server'] = retPnts.apply(server_is_left_handed)

chipPnts_right = retPnts.filter(server_is_right_handed).filter(is_chipped)
chipPnts_left = retPnts.filter(server_is_left_handed).filter(is_chipped)

chipShots_right = chipPnts_right.get_returns()
chipShots_left = chipPnts_left.get_returns()

chipFrm_left = regFrm[(regFrm.is_chipped == 1) & (regFrm.is_left_handed_server == 1)]
topspin_preds_left = 1 - chipFrm_left.pred_is_chipped.values
topspin_odds_left = (1 - chipFrm_left.pred_is_chipped.values) / chipFrm_left.pred_is_chipped.values

chipFrm_right = regFrm[(regFrm.is_chipped == 1) & (regFrm.is_right_handed_server == 1)]
topspin_preds_right = 1 - chipFrm_right.pred_is_chipped.values
topspin_odds_right = (1 - chipFrm_right.pred_is_chipped.values) / chipFrm_right.pred_is_chipped.values

chipScn = vs.kde_heatmap(chipShots_right,  showReg = showReg,
                                  position_func = 'get_first_bounce_position', 
                                  weight_func_or_arr = topspin_preds_right,
                                  normalize_density = True,
                                  bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
rightFrm = backView.combine(kdeView).snap()
rightFrm.show()

chipScn = vs.kde_heatmap(chipShots_left,  showReg = showReg,
                                  position_func = 'get_first_bounce_position', 
                                  weight_func_or_arr = topspin_preds_left,
                                  normalize_density = True,
                                  bandwidth = 45)

kdeView = View.make(myCam, chipScn, resolution = 1)
leftFrm = backView.combine(kdeView).snap()
leftFrm.show()

########################################
#### return dependence on serve side ###
########################################

adPnts = retPnts.filter('is_serve_ad')
deucePnts = retPnts.filter('is_serve_deuce')

regFrm['is_serve_ad'] = retPnts.apply('is_serve_ad')
regFrm['is_serve_deuce'] = retPnts.apply('is_serve_deuce')

chipShots_ad = adPnts.filter(is_chipped).get_returns()
chipShots_deuce = deucePnts.filter(is_chipped).get_returns()

chipFrm_ad = regFrm[(regFrm.is_chipped == 1) & (regFrm.is_serve_ad == 1)]
chipFrm_deuce = regFrm[(regFrm.is_chipped == 1) & (regFrm.is_serve_deuce == 1)]

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



#### why are so many nan? #####

retPnts.filter(is_chipped)[500].stream(confs = confs).widen(35).show()
chip_retShots


#################################
###### Visualise some stuff #####
#################################

srvs, rets = retPnts.get_serves(), retPnts.get_returns()

for idx in range(10):
    srv, ret = srvs[idx], rets[idx]
    print(ret.get_hit_event_descriptor())
    print(ret.get_initial_spin())
    frms = collect_video_frames([srv, ret], confs, widen_by = 0)
    frm = frms[0].join(frms[1])
    DecapStream(frm, 0, 100).show(ms_between_frames = 25)

        
    # path = get_folder_struct('graphs')['main']
    # fig, ax = plt.subplots(1, 1)
    # plotFrm = ret.shotTraj.trajFrm[['px_t', 'pz_t']]
    # plotFrm.plot.scatter(x = 'px_t', y = 'pz_t', ax = ax)
    # ax.set_xlim([0, 750])
    # fig.suptitle(ret.get_initial_spin())
    # fig.savefig(os.path.join(path, f'{idx}.png'))

import matplotlib.pyplot as plt
import os

rets[0].get_initial_spin()


srvs[0].get_hit_event_descriptor()
rets[2].shotTraj.trajFrm


def point_is_chipped_return(myPnt):
    retShot = myPnt.get_return()
    return retShot.is_slice_visual()

chipPnts = retPnts.filter(point_is_chipped_return)

chipFrms = collect_video_frames(chipPnts[20:40], confs)

for frm in chipFrms:
    DecapStream(frm, 0, 100).show()



chipPnts[0].get_match_index()

for idx in retPnts:


retPnts[0:10]

