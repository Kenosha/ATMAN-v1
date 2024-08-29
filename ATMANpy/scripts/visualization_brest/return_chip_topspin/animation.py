
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


import ATMANpy.auxil.constants as const

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

retPnts = retPnts.enforce_direction('Bot') ### Server is at the Bottom, Receiver at the Top
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

varDict['serve_dist_sq'] = varDict['serve_dist'] ** 2
varDict.pop('serve_dist')
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

### Add something to help us remember which row we changed the direction of play for.
regFrm['is_top_to_bot'] = idxFrm_new['is_top_to_bot']

### Return Shot characteristics
regFrm['ret_speed'] = retPnts.apply(vis_ut.get_return_speed)
regFrm['ret_spin'] = retPnts.apply(vis_ut.get_return_spin)
regFrm['ret_time'] = retPnts.apply(vis_ut.get_return_flight_time, fps = 25)

regFrm['serve_fore'] = retPnts.apply(vis_ut.is_serve_from_fore)

### weights
DECAY_RATE = 0.75
regFrm['len'] = retPnts.apply('__len__')
regFrm['len_weight'] = DECAY_RATE ** (regFrm['len'] - 2)
regFrm['odds_topspin'] = regFrm['pred_is_topspin'] / regFrm['pred_is_chipped']
regFrm['weight_topspin'] = np.where(regFrm.is_chipped, regFrm.odds_topspin, 1)
regFrm['weight_len_pred'] = regFrm['len_weight'] * regFrm['weight_topspin']

### winner vars
regFrm['ret_wins'] = regFrm['winner_code'] == -1
regFrm['srv_wins'] = regFrm['winner_code'] == 1

### winner vars, weighted with length --> attribution not via weight, but (more intuitively) with length of rally
    ## -> in essence, by doing this we can also measure the propensity of a return to prolong the point 
    # ---> ret_wins_weighted tends to 0.5 as the length increases.
regFrm['ret_wins_weighted'] = (regFrm['ret_wins'] * regFrm['len_weight']) + (0.5 * (1 - regFrm['len_weight']))
regFrm['srv_wins_weighted'] = (regFrm['srv_wins'] * regFrm['len_weight']) + (0.5 * (1 - regFrm['len_weight']))

### figure out bounces
regFrm['did_bounce'] = retPnts.apply(vis_ut.is_return_bounce_not_none, dtype = bool)
regFrm['did_bounce_in'] = retPnts.apply(vis_ut.is_return_bounce_in, dtype = bool)

regFrm['must_have_bounced_in'] = np.where(regFrm['ret_wins'], True, 
                                          np.where(regFrm['len'] > 3, True, ### why 3? because sometimes a third shot comes even though the ball is out on the second shot.
                                                   regFrm['did_bounce_in'])).astype(bool)

regFrm['must_have_bounced_out'] = ~regFrm['must_have_bounced_in']

regFrm['must_have_bounced_in_but_didnt'] = regFrm['must_have_bounced_in'] & regFrm['did_bounce'] & ~regFrm['did_bounce_in']
### these cases are weird, but mostly harmless -> ball seen as just out when it's just in.

#################################
###### ATTRIBUTION TABLES #######
#################################
from ATMANpy.types.Overlays import Table

chips_naive, tops_naive, attrib_naive = vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', 
                                                                     bounce_col = 'must_have_bounced_in', weight_col = None)
chips_smart, tops_smart, attrib_smart = vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', 
                                                                     bounce_col = 'must_have_bounced_in', weight_col = 'weight_topspin')
### tops_naive == tops_smart

DIMENSIONS = (1920 // 3, 1080 // 3)

columns_rename_dict = {'total': 'Returner Wins', 'make': 'Return is Made', 'cond': 'Returner Wins if Return is Made'}
columns_rename_dict_signs = {'total': 'Returner Wins =', 'make': 'Return is Made x', 'cond': 'Returner Wins if Return is Made'}

chips_naive_tot = {k: v for k, v in chips_naive.items() if k == 'total'}
tops_naive_tot = {k: v for k, v in tops_naive.items() if k == 'total'}
chips_smart_tot = {k: v for k, v in chips_smart.items() if k == 'total'}
tops_smart_tot = {k: v for k, v in tops_smart.items() if k == 'total'}

### First, simply compare total win percentage
table_data = {'Topspin Return': tops_naive_tot, 'Chipped Return': chips_naive_tot}
myTab = Table(table_data, number_format = '.1%', index_offset = 0.25, index_col_width = 1.25, header_row_height = 0.5,
              skip_after_how_many_symbols = 20,
              dimensions = DIMENSIONS, columns_rename_dict = columns_rename_dict)
myTab.render().show()

table_data = {'Topspin Return': tops_naive, 'Chipped Return': chips_naive}
myTab = Table(table_data, number_format = '.1%', index_offset = 0.25, index_col_width = 1.25, header_row_height = 0.5,
              skip_after_how_many_symbols = 20,
              dimensions = DIMENSIONS, columns_rename_dict = columns_rename_dict)
myTab.render().show()

table_data = {'Topspin Return': tops_naive, 'Chipped Return': chips_naive}
myTab = Table(table_data, number_format = '.1%', index_offset = 0.25, index_col_width = 1.25, header_row_height = 0.5,
              skip_after_how_many_symbols = 20,
              dimensions = DIMENSIONS, columns_rename_dict = columns_rename_dict_signs)
myTab.render().show()

table_data = {'Topspin Return': tops_naive, 'Chipped Return - Choice': chips_smart}
myTab = Table(table_data, number_format = '.1%', index_offset = 0.25, index_col_width = 1.25, header_row_height = 0.5,
              skip_after_how_many_symbols = 20,
              dimensions = DIMENSIONS, columns_rename_dict = columns_rename_dict_signs)
myTab.render().show()

###################################
#### FRAMES RASTER WITH TABLE #####

tabFrm = myTab.render()

from ATMANpy.types.Overlays import FramesRaster, frame_cascade
baseRst = FramesRaster(3, 3)
baseRst.add(tabFrm, 1, 1)

dirFrm = regFrm[(regFrm['is_top_to_bot'])]

high_pred_exFrm = dirFrm.sort_values('pred_is_chipped').iloc[-8:]
high_pred_exFrms = [retPnts[idx][:2].stream(confs = confs).decap().frames for idx in high_pred_exFrm.index]

cascFrms = frame_cascade(high_pred_exFrms, baseRst)
DecapStream(cascFrms, 0, 100).show() 
### all these have three things in common 
#   -> they're returned off the backhand...
#   -> ... on the stretch
#   -> ... as a chip!
# In fact, there is virtually no way to return these as topspin balls!
#   -> no choice! forced chip return.
# we are interested in the opposite: 'choice' chips! 
# given that the receiver can make a choice to chip, we want to assess whether that choice is worth it.

### Propensity Wording:
# To be able to differentiate between forced chips and choice chips, we employ a technique called PSM
# PSM calculates the propensity to chip a ball given a set of explanatory variables.
# in our case, these variables pertain to the quality of the serve 
# (show and discuss table now?)
# the higher the propensity to chip, the more the model believes a chipped return to be forced.
# in fact, the eight returns we just saw are the eight highest propensity returns in our sample.
# on the other hand, on a low propensity chipped return, the model believes that playing a chip in such a situation is very unlikely 
#   -> in most cases, the returner would have opted for a topspin return
#   -> therefore, a chipped return in such a low propensity situation expresses the returner's strong choice to chip the ball back.
# (maybe show a continuum line: low propensity -> choice; high propensity -> forced)
# To illuminate this, consider the 8 lowest propensity chipped returns.
low_pred_exFrm = dirFrm[dirFrm.is_chipped].sort_values('pred_is_chipped').iloc[:8]
low_pred_exFrms = [retPnts[idx][:2].stream(confs = confs).decap().frames for idx in low_pred_exFrm.index]

cascFrms = frame_cascade(low_pred_exFrms, baseRst)
DecapStream(cascFrms, 0, 100).show() 

# on these returns, the PSM believes that the returner could very well have played a topspin return; instead, the returner opted for a chipped return.
# In this sense, the PSM expresses choice. 
# We can now use these propensities to focus on chipped returns where the returner actually had a choice, 
# and thereby attempt to assess the quality of this choice.

# (show table with choice returns)

### still pretty underwhelming for the chipped return: choice chipped returns fare only moderately better.
### we try to understand why.

# (split -> make% and cond%)


fitted_regression = logReg
def regression_to_table(fitted_regression):
    pass

logReg.params
logReg.pvalues
dir(logReg)

list_of_cascFrms = high_dist_exFrms
baseRst = myRst



table_data = {'Chipped Return': chips_naive, 'Chipped Return - Choice': chips_smart, 'Topspin Return': tops_smart}

# class Table:
#     table_parameters: list
#     table_dimensions: tuple = (600, 400)
#     render_height: int = 100
#     resolution: int = 10
#     borders: bool = True
#     header_text: str = 'Legend'
#     header_size: int = 2
#     header_thickness: int = 2
#     header_offset: int = 10
#     header_color: tuple = const.BGR['smokey_grey']

#     def render(self):
#         legCam = Camera.birds_eye(self.render_height, sensor_dims = self.legend_dimensions)
#         legReg = legCam.approximate_fov(self.render_height)

#         legLines = legReg.lineList
#         legLines = legLines.resize(0.25)

#         if not self.borders:
#             legLines = legLines.realpha(0)

#         num_els = len(self.element_parameters)

#         subRegs = legReg.subdivide(elements = num_els + 1, how = 'horizontal')
#         headerReg, otherRegs = subRegs[0], subRegs[1:]

#         headerPt = make_text_in(headerReg, self.header_text, 
#                                 size = self.header_size, 
#                                 thickness = self.header_thickness,
#                                 offset = self.header_offset,
#                                 color = self.header_color)
