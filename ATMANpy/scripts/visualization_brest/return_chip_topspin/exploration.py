
import os
import numpy as np
import pandas as pd

import ATMANpy.auxil.constants as const

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Point, read_in_all_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature

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
firstPnts = allPnts.filter('is_first_serve').enforce_direction('Bot') ### Server is at the Bottom, Receiver at the Top

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

predSer = pd.Series(logReg.predict(), name = 'pred_is_chipped')

#########################################
##### CREATE REGRESSORS AND WEIGHTS #####
#########################################

regFrm = pd.concat([xFrm, ySer, winSer, predSer], axis = 1)
regFrm['pred_is_topspin'] = 1 - predSer

### Return Shot characteristics
regFrm['ret_speed'] = retPnts.apply(vis_ut.get_return_speed)
regFrm['ret_spin'] = retPnts.apply(vis_ut.get_return_spin)

### weights
DECAY_RATE = 0.75
regFrm['len'] = retPnts.apply('__len__')
regFrm['len_weight'] = DECAY_RATE ** (regFrm['len'] - 2)
regFrm['odds_topspin'] = (1 - regFrm['pred_is_chipped']) / regFrm['pred_is_chipped']
regFrm['weight_topspin'] = np.where(regFrm.is_chipped, regFrm.odds_topspin, 1)
regFrm['weight_len_pred'] = regFrm['len_weight'] * regFrm['weight_topspin']

### winner vars
regFrm['ret_wins'] = regFrm['winner_code'] == -1
regFrm['srv_wins'] = regFrm['winner_code'] == 1

### figure out bounces
regFrm['did_bounce'] = retPnts.apply(vis_ut.is_return_bounce_not_none, dtype = bool)
regFrm['did_bounce_in'] = retPnts.apply(vis_ut.is_return_bounce_in, dtype = bool)

regFrm['must_have_bounced_in'] = np.where(regFrm['ret_wins'], True, 
                                          np.where(regFrm['len'] > 3, True, ### why 3? because sometimes a third shot comes even though the ball is out on the second shot.
                                                   regFrm['did_bounce_in'])).astype(bool)

regFrm['must_have_bounced_out'] = ~regFrm['must_have_bounced_in']

regFrm['must_have_bounced_in_but_didnt'] = regFrm['must_have_bounced_in'] & regFrm['did_bounce'] & ~regFrm['did_bounce_in']
### these cases are weird, but mostly harmless -> ball seen as just out when it's just in.

vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', bounce_col = 'must_have_bounced_in', weight_col = None)
vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', bounce_col = 'must_have_bounced_in', weight_col = 'weight_topspin')
vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', bounce_col = 'must_have_bounced_in', weight_col = 'weight_len_pred')

### OVERALL DELTA
######## NAIVELY, THE WIN PROBABILITY ON A CHIPPED RETURN IS ALMOST HALF THAT OF A TOPSPIN RETURN ... DELTA: 18%
####### ... BUT CONTROLLING FOR ABILITY TO CHOOSE, THE WIN PROBABILITY FOR A CHIPPED RETURN INCREASES! DELTA: 13%
######## WEIGHTING SHORT EXCHANGES HIGHER (HERE, THE RETURN IS MORE LIKELY TO BE OF IMPORTANCE), THE DELTA DECREASES TO 9%

### MAKE
####### LOOKING AT RETURNS KEPT IN PLAY NAIVELY, A TOPSPIN SHOT HAS A 17% HIGHER CHANCE OF LANDING IN.
######## WITH WEIGHTS, THE DELTA NARROWS TO 7%
### WITH LENGTH WEIGHTS ADDITIONALLY (DOESN'T REALLY MAKE SENSE...) -> DELTA = 6%

### COND WIN
# -> GIVEN THAT THE BALL BOUNCES IN -> WHAT ARE WINNING PERCENTAGES?
######## NAIVELY, DELTA: 16% ...
######## ... BUT CONTROLLING FOR ABILITY TO CHOOSE: DELTA: 14%
### WITH LENGTH WEIGHTS ADDITIONALLY (DOESN'T REALLY MAKE SENSE...) -> DELTA = 8%

##################################
######## LOOK AT MAKE% ###########
##################################
# -> how does a ball typically go out?
### Problem -> we'd have to classify every ball that does not have a bounce, and say whether it goes long, or quasi ace, or what.
regFrm['ret_dist_to_bsl'] = retPnts.apply(vis_ut.get_ret_dist_to_bsl)
regFrm['ret_dist_to_bh'] = retPnts.apply(vis_ut.get_ret_dist_to_bh)
regFrm['ret_dist_to_opt'] = retPnts.apply(vis_ut.get_ret_dist_to_opt)

regFrm['bounces_in_own_court'] = regFrm['ret_dist_to_bsl'] > const.SIDE_LENGTH * const.FOOT_TO_METER
regFrm['is_in_net'] = retPnts.apply(vis_ut.return_is_in_net, dtype = bool)
regFrm['return_is_long'] = regFrm['ret_dist_to_bsl'] < 0
regFrm['return_is_wide_bh'] = regFrm['ret_dist_to_bh'] < 0
regFrm['return_is_wide_fh'] = regFrm['ret_dist_to_bh'] > const.SINGLES_WIDTH * const.FOOT_TO_METER
regFrm['return_is_junk'] = regFrm['bounces_in_own_court'] | ~regFrm['did_bounce']

regFrm['return_is_fine'] = ~regFrm['return_is_junk']


# regFrm['is_in_net_for_sure'] = regFrm['must_have_bounced_out'] & regFrm['is_in_net'] ### difference is false net calls...
# regFrm['return_is_long_for_sure'] = regFrm['must_have_bounced_out'] & regFrm['return_is_long'] 
# regFrm['return_is_wide_bh_for_sure'] = regFrm['must_have_bounced_out'] & regFrm['return_is_wide_bh'] 
# regFrm['return_is_wide_fh_for_sure'] = regFrm['must_have_bounced_out'] & regFrm['return_is_wide_fh'] 
# regFrm['ace_with_a_racket_on'] = regFrm['must_have_bounced_out'] & (regFrm['return_is_junk'] & ~regFrm['is_in_net'])
# grp = regFrm.groupby('is_chipped')
# my_cols = ['ace_with_a_racket_on', 'is_in_net_for_sure', 'return_is_long_for_sure', 'return_is_wide_bh_for_sure', 'return_is_wide_fh_for_sure']
# meanFrm = grp[my_cols].mean()
# meanFrm.sum(axis = 1)

#### From here on out, we look at returns made
# -> filter!
in_col = 'return_is_fine' #'did_bounce_in'
retPnts_in = retPnts.filter(regFrm[in_col].values)
regFrm_in = regFrm[regFrm[in_col]]

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
tops_retShots = retShots.filter('is_topspin_visual')

chip_retShots_rectified = retShots_rectified.filter('is_slice_visual')
tops_retShots_rectified = retShots_rectified.filter('is_topspin_visual')

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

regFrm_in['ret_dist_to_bh_sq'] = regFrm_in['ret_dist_to_bh'] ** 2
regFrm_in['ret_dist_to_bsl_sq'] = regFrm_in['ret_dist_to_bsl'] ** 2
regFrm_in['ret_dist_to_opt_sq'] = regFrm_in['ret_dist_to_opt'] ** 2

project_results_path = get_folder_struct('results_video')['return_chip_topspin']
save_path = maybe_make_folder(project_results_path, 'regression' + make_timestamp())

### Unweighted for a Base Case
regDict = {'base': ['is_chipped']}
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = None)

### Weighted with topspin odds (weight_topspin)
regDict = {'base_odds': ['is_chipped'],
           'dist_bsl_odds': ['is_chipped', 'ret_dist_to_bsl']}
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = 'weight_topspin')

### Weighted with topspin odds and len weights (weight_len_pred)
regDict = {'base_weighted': ['is_chipped'],
           'dist_bsl_weighted': ['is_chipped', 'ret_dist_to_bsl'],
           'dist_bsl_opt_weighted': ['is_chipped', 'ret_dist_to_bsl', 'ret_dist_to_opt'],
           'dist_bsl_bh_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl'],
           'dist_bsl_bh_opt_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_opt'],
           'dist_bsl_bh_sq_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq'],
           'dist_bsl_bh_opt_sq_weighted': ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq', 'ret_dist_to_opt', 'ret_dist_to_opt_sq']} 
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = 'weight_len_pred')

### Get exog, endog data from model
regressors = ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = 'weight_len_pred')
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
regArr = np.concatenate([ones, rel_rastArr, ones], axis = 1)

colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regArr, relative_to = avg_topspin_pred)
chipScn.colArrs[0] = colArr
chipScn.alphaArrs[0] = (chipScn.alphaArrs[0] * alArr) ** 0.5

kdeView = View.make(myCam, chipScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()


### Get exog, endog data from model
regressors = ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = 'weight_len_pred')
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
regArr = np.concatenate([ones, rel_rastArr, rel_rastArr ** 2, ones], axis = 1)

colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regArr, relative_to = avg_topspin_pred)
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

colArr = (regFrm_in.loc[regFrm_in.is_chipped == 1, 'winner_code'] + 1).values
markScn = vs.scatter_markmap(tops_retShots_rectified, color_func_or_arr = colArr, 
                             zero_color = const.BGR['leafy_green'], one_color = const.BGR['intense_red'])
markFrm = View.make(myCam, markScn, resolution = 1).snap()
markFrm.show()

# X, y, w = vis_ut.get_X_y_w(regFrm, [*list(varDict.keys()), 'is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
# glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
# glmReg_win.summary()

# X, y, w = vis_ut.get_X_y_w(regFrm, ['is_fore', 'speed', 'dist', 'is_chipped', 'ret_dist_to_bsl'], 'ret_wins', add_constant = True, weight_col = 'weight_len_pred')
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

mode_angles = chip_modes[[0, 4]]
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

mode_angles = tops_modes[[0]]
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

#######################################
### LAUNCH ANGLE IN A SPECIFICATION ###
#######################################
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
            'dist_bsl_bh_sq_launch_inter_weighted': [   'is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                        'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq',
                                                        'launch_angle', 'ret_dist_to_bsl_x_launch_angle'],
            'dist_bsl_bh_sq_launch_inter_speed_weighted':  ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                            'ret_dist_to_bh_sq', 'ret_dist_to_bsl_sq',
                                                            'launch_angle', 'ret_dist_to_bsl_x_launch_angle',
                                                            'ret_speed', 'ret_spin'], 
            'dist_bsl_bh_halfsq_launch_inter_speed_weighted':  ['is_chipped', 'ret_dist_to_bh', 'ret_dist_to_bsl', 
                                                                'ret_dist_to_bh_sq',
                                                                'launch_angle', 'ret_dist_to_bsl_x_launch_angle'], 
          
          }
for save_name, regressors in regDict.items():
    vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = 'weight_len_pred')


###################################
## CHIP RETURN QUALITY LOCATIONS ##
###################################
from ATMANpy.types.Visual3D import DummyScene

regressors = regDict['dist_bsl_bh_launch_inter_weighted']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = 'weight_len_pred')
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

regArr = np.concatenate([ones, rel_rastArr, angs, dist_x_angs, ones], axis = 1)

colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regArr, relative_to = avg_topspin_pred)
zArr = np.zeros((num_x * num_y, 1))
rastArr_3D = np.concatenate([rastArr_2D, zArr], axis = 1)
rastScn = DummyScene([rastArr_3D], [colArr], [alArr])

kdeView = View.make(myCam, rastScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()


regressors = regDict['dist_bsl_bh_launch_inter_weighted']
glmReg_win = vis_ut.make_and_save_model(regFrm_in, regressors, save_path, save_name = save_name, weight_col = 'weight_len_pred')
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

regArr = np.concatenate([ones, rel_rastArr, angs, dist_x_angs, ones], axis = 1)

colArr, alArr = vis_ut.get_color_and_alpha_array_from_predictions(glmReg_win, regArr, relative_to = avg_topspin_pred)
zArr = np.zeros((num_x * num_y, 1))
rastArr_3D = np.concatenate([rastArr_2D, zArr], axis = 1)
rastScn = DummyScene([rastArr_3D], [colArr], [alArr])

kdeView = View.make(myCam, rastScn, resolution = 1)
chipFrm = backView.combine(kdeView).snap()
chipFrm.show()

### TODO: for heatmaps: maybe use topspin_odds, but zero color is also red -> this way, the KDE does not appear so falsely sparse!

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

