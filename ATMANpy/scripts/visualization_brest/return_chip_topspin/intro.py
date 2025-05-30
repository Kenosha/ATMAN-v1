
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Point, NullPoint, read_in_data_tables, read_in_placeholder_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.Overlays import overlay, overlay_center
from ATMANpy.types.Shapes3D import Region3D

from statsmodels.discrete.discrete_model import Logit

DESCRIPTOR = 'intro'
SAVE_FOLDER = 'return_chip_topspin'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

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
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

firstPnts = dataPnts.filter('is_first_serve')

### Remove aces, faults and net balls.
def get_returned_balls(myPnt):
    return len(myPnt) > 1
all_retPnts = firstPnts.filter(get_returned_balls)

##################################
############### CAM ##############
##################################

M_ext = ConstFeature.from_store(vid, sto, featname = 'M_ext').feature
M_int = ConstFeature.from_store(vid, sto, featname = 'M_int').feature

# brestCam_third = Camera(M_int, M_ext, (640, 360))
brestCam_third = Camera.calibrated_brest_behind()
brestCam_half = brestCam_third.resize(1.5)
brestCam_full = brestCam_third.resize(3)

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

retPnts_pre = dec_retPnts.filter(vis_ut.is_not_pathological, threshold_speed = THRESHOLD_SPEED, threshold_frames = THRESHOLD_FRAMES)
pathPnts_pre = dec_retPnts.filter(vis_ut.is_pathological, threshold_speed = THRESHOLD_SPEED, threshold_frames = THRESHOLD_FRAMES)

pathSer = pd.Series(dec_retPnts.apply(vis_ut.is_pathological))
winSer = dec_winSer[pathSer != 1].reset_index(drop = True)

##### ENFORCE DIRECTION
change_direction_array = retPnts_pre.apply('is_top_to_bot', dtype = bool) ### for convenience, store whether we'll change the direction of the point.
indices = [(pt.get_match_index(), pt.get_point_index()) for pt in retPnts_pre]
idxFrm = pd.DataFrame(indices, columns = ['match_idx', 'point_idx'])
idxFrm['is_top_to_bot'] = change_direction_array

retPnts = retPnts_pre.enforce_direction('Bot') ### Server is at the Bottom, Receiver at the Top
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

xFrm = xFrm.drop(['serve_width', 'serve_spin'], axis = 1) ### these two are highly insignificant

logReg = Logit(ySer, xFrm)
logReg = logReg.fit()
logReg.summary()

#########################################
##### CREATE REGRESSORS AND WEIGHTS #####
#########################################
DECAY_RATE = 0.9
regFrm = vis_ut.make_regressor_frame(retPnts, logReg, winSer, idxFrm_new, decay_rate = DECAY_RATE)

# ###### FIXED IN MAKE_REGRESSOR_FRAME
# SHOW_CHIP_WEIRDNESS = False
# SHOW_TOPS_WEIRDNESS = False

# ### little test: why do almost 10% of chipped returns in rallys that end on that return result in a return winner???!!!
# checkFrm_chip = regFrm[(regFrm.winner_code == -1) & regFrm.is_chipped & (regFrm.len == 2)]
# weird_idxFrm = checkFrm_chip[['mtc_idx', 'pnt_idx']]

# if SHOW_CHIP_WEIRDNESS:
#     i = 0
#     for idx, mtc_pnt in weird_idxFrm.iterrows():
#         mtc, pnt = mtc_pnt
#         myPnt = retPnts.get(mtc, pnt)
#         nextPnt = allPnts.get(mtc, pnt + 1)

#         winner_name = myPnt.get_winner_from_score(nextPnt, verbose = True)
#         print(myPnt)
#         myPnt.stream(confs = confs).widen(200).show(ms_between_frames = 50)
#         print('_________')
        
#         i += 1

# ### fix these instances manually! none of these are 2nd shot winners, they're all length 3 (but directionally correct)!
# regFrm.loc[weird_idxFrm.index, 'len'] = 3

# ### same test for topspin return stuff.
# checkFrm_tops = regFrm[(regFrm.winner_code == -1) & ~regFrm.is_chipped & (regFrm.len == 2)]
# weird_idxFrm = checkFrm_tops[['mtc_idx', 'pnt_idx']]

# if SHOW_TOPS_WEIRDNESS:
#     i = 0
#     for idx, mtc_pnt in weird_idxFrm.iterrows():
#         mtc, pnt = mtc_pnt
#         myPnt = retPnts.get(mtc, pnt)
#         nextPnt = allPnts.get(mtc, pnt + 1)

#         print(i)
#         winner_name = myPnt.get_winner_from_score(nextPnt, verbose = True)
#         print(myPnt)
#         myPnt.stream(confs = confs).widen(200).show(ms_between_frames = 50)
#         print('_________')

#         i += 1

# to_winner_code_zero = [1, 2, 10, 17, 19]
# to_len_three = [0, 5, 6, 7, 9, 13]

# regFrm.loc[weird_idxFrm.index[to_winner_code_zero], 'winner_code'] = 0
# regFrm.loc[weird_idxFrm.index[to_len_three], 'len'] = 3


#################################
####### PARAMETERS ##############
#################################
SKELETON_COLOR = const.BGR['leafy_green']

#################################
###### ATTRIBUTION TABLES #######
#################################
from ATMANpy.types.Overlays import Table, Graph

### a) unweighted
chips_naive, tops_naive, attrib_naive = vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', 
                                                                     bounce_col = 'must_have_bounced_in', weight_col = None)
### b) PSM-weighted
chips_smart, tops_smart, attrib_smart = vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', 
                                                                     bounce_col = 'must_have_bounced_in', weight_col = 'weight_topspin')
### c) PSM and Balance-Weighted
chips_smart_bal, tops_smart_bal, attrib_smart_bal = vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', 
                                                                     bounce_col = 'must_have_bounced_in', weight_col = 'weight_bal_pred')

### d1) PSM and Balance and Len-Weighted
# chips_smart_bal_len, tops_smart_bal_len, attrib_smart_bal_len = vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', 
#                                                                      bounce_col = 'must_have_bounced_in', weight_col = 'weight_bal_len_pred',
#                                                                      assert_digits = 1)

### d2) PSM and Balance and Len-Weighted (outcome method)
chips_smart_bal_len, tops_smart_bal_len, attrib_smart_bal_len = vis_ut.calculate_attribution(regFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins_weighted', 
                                                                     bounce_col = 'must_have_bounced_in', weight_col = 'weight_bal_pred',
                                                                     assert_digits = 1)


### b) len weighted
print(f'Topspin Returns:')
print(f'   No len adj: {tops_smart["cond"]:.2f}; len adj: {chips_smart_bal_len["cond"]:.2f}')
print(f'Chipped Returns:')
print(f'   No len adj: {chips_smart["cond"]:.2f}; len adj: {chips_smart_bal_len["cond"]:.2f}')

#################################
###### LENGTH OF RALLY ##########
#################################

GRAPH_RESOLUTION = 50

tops_naive_len, chips_naive_len, chips_smart_len = vis_ut.calculate_rally_lengths(regFrm)
naiveGraph, smartGraph = vis_ut.uniform_kdes_for_rally_length(regFrm, normalize_density = False, bandwidth = 1.5)
# smartGraph.render().show()
### if rally length is not affected by chip/tops, 
# why does the conditional win percentage improve so drastically for chipped returns
# when introducing the length weight adjustment?
# (whereas topspin conditional win % does not change much?)

### Make a Graph: Win% returner over Rally Length
jntFrm, jntFrm_decayed, shrFrm, diffFrm_chip, diffFrm_top, avgFrm = vis_ut.chip_topspin_return_avgs_by_length(regFrm, decay_rate = DECAY_RATE)
diffFrm_chip.sum() 
### these are the 7.5% we see in overall difference. 
#If we were to exclude the 2-length rallys (where the return is usually not made)
# we will get closer to 10%. The analysis could be re-done with just the conditional returns.

### To build an intuition on the improvement -> bucket into <4, 4-8, >8 rallys.
def bucket(frm_or_ser, share_frm_or_ser = None):
    short_filter = frm_or_ser.index < 5
    mid_filter = (frm_or_ser.index >= 5) & (frm_or_ser.index < 9)
    long_filter = frm_or_ser.index >= 9

    if share_frm_or_ser is None:
        print('Short Rally: \n', frm_or_ser[short_filter].sum())
        print('Medium Rally: \n', frm_or_ser[mid_filter].sum())
        print('Long Rally: \n', frm_or_ser[long_filter].sum())
    else:
        print('Short Rally: \n', frm_or_ser[short_filter].sum() / share_frm_or_ser[short_filter].sum())
        print('Medium Rally: \n', frm_or_ser[mid_filter].sum() / share_frm_or_ser[mid_filter].sum())
        print('Long Rally: \n', frm_or_ser[long_filter].sum() / share_frm_or_ser[long_filter].sum())

bucket(jntFrm, shrFrm)
bucket(jntFrm_decayed, shrFrm)
bucket(diffFrm_chip['diff'])
bucket(diffFrm_top['diff'])

jntFrm.sum() / shrFrm.sum()
jntFrm_decayed.sum() / shrFrm.sum()

DIMENSIONS = (1920 // 3, 1080 // 3)
x1, y1 = DIMENSIONS
x2, y2 = x1 * 2, y1 * 2

#### CHART FOR SERVE + 1 and + 2

cond_shrFrm = shrFrm.loc[3:] / (1 - shrFrm.loc[2])
cond_shrFrm.loc[3]
cond_shrFrm.loc[5]

avgFrm.loc[3]
avgFrm.loc[5]

combFrm = avgFrm * shrFrm
combFrm.loc[7:].sum() / shrFrm.loc[7:].sum()

avgFrm.loc[6]

chip_shareFrm = shrFrm['chip_smart']
chip_avgFrm = avgFrm['chip_smart']

shares_2_6 = chip_shareFrm.loc[2:6].values
shares_7_8   = chip_shareFrm.loc[7:8].sum()
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
                'header_text': 'Server Success against Choice Chipped Returns', 'header_size': 2, 'header_horizontal_offset': 5,
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

tlPt = Point2D((900, 100))

myGraph = Graph([nullChart, avgChart], **graph_kwargs)
avg_graphFrm0 = myGraph.render()
avg_graphFrm0 = overlay(avg_graphFrm0, legFrm1, top_left_point = tlPt)
# avg_graphFrm0.show()

myGraph.add(vChart1)
avg_graphFrm1 = myGraph.render()
avg_graphFrm1 = avg_graphFrm1.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
avg_graphFrm1 = overlay(avg_graphFrm1, legFrm1, top_left_point = tlPt)
# avg_graphFrm1.show()

myGraph.add(vChart2)
avg_graphFrm2 = myGraph.render()
avg_graphFrm2 = avg_graphFrm2.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
avg_graphFrm2 = avg_graphFrm2.put('Serve+2', color = const.BGR['intense_red'], where = (716, 425), size = 1)
avg_graphFrm2 = overlay(avg_graphFrm2, legFrm1, top_left_point = tlPt)
# avg_graphFrm2.show()

myGraph.add(shrChart)
shr_graphFrm = myGraph.render()
shr_graphFrm = shr_graphFrm.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
shr_graphFrm = shr_graphFrm.put('Serve+2', color = const.BGR['intense_red'], where = (716, 425), size = 1)
shr_graphFrm = overlay(shr_graphFrm, legFrm2, top_left_point = tlPt)
# shr_graphFrm.show()

###### INVESTIGATE SELECTION BIAS #########

TABLE_RESOLUTION = 50

### A) example tables with Player A and B

table_kwargs_examples = {   'number_format': '.1%', 'data_right_adjust': False, 
                            'index_offset': 0.25, 'index_col_width': 1.25, 'header_row_height': 0.5,
                            'skip_after_how_many_symbols': 20, 'dimensions': (1920 // 2, 1080 // 2),
                            'skeleton_shrinkage': 0.35, 'skeleton_vertical_offset': 1,
                            'title_vertical_offset': 0, 'title_horizontal_offset': 11.2,
                            'skeleton_vertical_offset': 1, 'title_color': const.BGR['bright_grey'],
                            'title_size': 2, 'title_thickness': 1, 'header_line_thickness': 0.1,
                            'data_text_size': 2, 'header_text_size': 2, 'index_text_size': 2,
                            'resolution': TABLE_RESOLUTION,
                        }

player_a = {'Serve': 'Great', 'Return': 'Average'}
player_b = {'Serve': 'Average', 'Return': 'Average'}
table_data = {'Player A': player_a, 'Player B': player_b}

ex1_table = Table(table_data, title = 'Example 1', **table_kwargs_examples)
ex1Frm = ex1_table.render()
# ex1Frm.show()

player_a = {'Serve': 'Average', 'Return': 'Great'}
player_b = {'Serve': 'Average', 'Return': 'Average'}
table_data = {'Player A': player_a, 'Player B': player_b}

ex2_table = Table(table_data, title = 'Example 2', **table_kwargs_examples)
ex2Frm = ex2_table.render()
# ex2Frm.show()

myStr = '-> Player A plays more points on return!'

ex1Frm_with_text = ex1Frm.put(myStr, where = (118, 500), size = 2, color = const.BGR['bright_grey'])
# ex1Frm_with_text.show()

ex2Frm_with_text = ex2Frm.put(myStr, where = (118, 500), size = 2, color = const.BGR['bright_grey'])
# ex2Frm_with_text.show()


#### TODO: Possible selection bias: the better player typically plays fewer serves (holds fast, hangs in for longer in return games).
### -> more returns observed by the better player!
### -> maybe do an importance weight correction?

BAR_RESOLUTION = 1500

f1 = regFrm.len < 5
f2 = (regFrm.len >= 5) & (regFrm.len < 7)
f3 = (regFrm.len >= 7) & (regFrm.len < 9)
f4 = regFrm.len >= 9

m1 = regFrm.loc[f1, 'ret_wins'].mean()
m2 = regFrm.loc[f2, 'ret_wins'].mean()
m3 = regFrm.loc[f3, 'ret_wins'].mean()
m4 = regFrm.loc[f4, 'ret_wins'].mean()

graph_kwargs = {'color': const.BGR['bright_grey'], 'y_label_format': '>4.0%', 'y_label': None, 'offset': 0.3, 
                'make_default_x_labels': False,
                'y_label_dist': 2.5, 'dimensions': (x2, y2), 'x_label_dist': 1.15, 'x_label_left_shift': 3.0, 
                'x_tick_label_left_shift': 0.6,
                'label_size': 2, 'tick_label_size': 2, 
                'header_text': 'Returner Win% over Rally Length', 'header_size': 2, 'header_horizontal_offset': 10,
                'x_label': 'Shots in Rally', 'resolution': BAR_RESOLUTION, 'z_epsilon': 0.02}

from ATMANpy.types.Overlays import BarChart, ScatterChart, Graph
half_line_xs = np.arange(0, 5, 0.025)
half_line_ys = np.full_like(half_line_xs, fill_value = 0.5)

barChart = BarChart([1, 2, 3, 4], [m1, m2, m3, m4], color = const.BGR['leafy_green'], external_gap = 0.4, 
                    x_labels = ['2-4', '5-6', '7-8', '> 8'], annotate_bar_heights = True)
halfChart = ScatterChart(half_line_xs, half_line_ys, color = const.BGR['intense_red'])
halfChart_2 = ScatterChart(half_line_xs, half_line_ys - 0.0015, color = const.BGR['intense_red'])
halfChart_3 = ScatterChart(half_line_xs, half_line_ys + 0.0015, color = const.BGR['intense_red'])

barGraph = Graph([barChart, halfChart, halfChart_2, halfChart_3], **graph_kwargs)
# barGraph.render().show()

bar_graphScn, bar_annotatedPnts = barGraph.preview()

newPnt = bar_annotatedPnts[7]
newPnt = newPnt.retext('50%').recolor(const.BGR['intense_red']).right(20.5).front(1.9)
bar_annotatedPnts.append(newPnt)

for idx, pnt in enumerate(bar_annotatedPnts):
    print(idx, pnt.text)

bar_annotatedPnts_start = [pnt for idx, pnt in enumerate(bar_annotatedPnts) if idx not in [7, 8, 9, 10]]

g1, g2, g3, g4 = bar_graphScn[2], bar_graphScn[3], bar_graphScn[4], bar_graphScn[5]

def make_null_region(reg):
    h = reg['Left'].length()
    t1, t2 = reg['Top'].leftPt, reg['Top'].rightPt
    b1, b2 = reg['Bottom'].leftPt, reg['Bottom'].rightPt

    return Region3D.from_points3D(b1, t1.back(h), t2.back(h), b2)

g1_null, g2_null, g3_null, g4_null = make_null_region(g1), make_null_region(g2), make_null_region(g3), make_null_region(g4) 

bar_graphScn_null = Scene([bar_graphScn[0], bar_graphScn[1], g1_null, g2_null, g3_null, g4_null, bar_graphScn[6], bar_graphScn[7], bar_graphScn[8]])

# View.make(myGraph.cam, bar_graphScn_null, annotatedPoints = bar_annotatedPnts_start, resolution = BAR_RESOLUTION).snap().show()
# View.make(myGraph.cam, bar_graphScn, annotatedPoints = bar_annotatedPnts, resolution = BAR_RESOLUTION).snap().show()


r_f, c_f = 1080, 1920
graphFrm = barGraph.render()
r_t, c_t, _ = graphFrm.shape
r_d, c_d = (r_f - r_t) // 2, (c_f - c_t) // 2
graph_offsetPt = Point2D((c_d, r_d))

### TODO: use this to animate the bars -> better to explain while bars appear.

### SCRIPT
# -> in a perfect world: ask each player to hit 100 serves to each other player in our sample: observe equally many serves and returns for each single player
# however, our data generating process are actual tennis matches. In these, what happens is that we observe more returns from the better player
### -> selection bias!
# why? imagine player A has a great serve and average return; player B has an average return and serve. 
# We expect player A to hold faster (due to his strong serve), and therefore play rel. fewer points on serve and rel. more on the return.
# similarly, if A had a great return and an average serve, we'd expect him to hang in for longer in return games, thus playing rel. more points on return, again.
# yet another way to show this -> what happens to win% if rallys go long? In this case, the impact of the srv/ret are nullified, 
# if both players were equally matched, we'd expect the win probability on long points to be at 50%.
# However, what we observe is that the return player has greater win probability in long rallys, 
# indicating that the better player indeed plays more returns in a match, simply due to the nature of tennis scoring.
# ---> show graph!
# to remove this bias, and approximate the experimental setting, 
# we introduce a balancing factor that gives more weight to returns made by the player that made fewer returns (ostensibly, the weaker player).

### tops_naive == tops_smart


columns_rename_dict = {'total': 'Returner Wins the Point', 'make': 'Return is Made', 'cond': 'Returner Wins the Point if Return is Made'}
columns_rename_dict_signs = {'total': 'Returner Wins =', 'make': 'Return is Made x', 'cond': 'Returner Wins if Return is Made'}

chips_naive_tot = {k: v for k, v in chips_naive.items() if k == 'total'}
tops_naive_tot = {k: v for k, v in tops_naive.items() if k == 'total'}
chips_smart_tot = {k: v for k, v in chips_smart.items() if k == 'total'}
tops_smart_tot = {k: v for k, v in tops_smart.items() if k == 'total'}
chips_smart_bal_tot = {k: v for k, v in chips_smart_bal.items() if k == 'total'}
tops_smart_bal_tot = {k: v for k, v in tops_smart_bal.items() if k == 'total'}
chips_smart_bal_len_tot = {k: v for k, v in chips_smart_bal_len.items() if k == 'total'}
tops_smart_bal_len_tot = {k: v for k, v in tops_smart_bal_len.items() if k == 'total'}

chips_smart_make = {k: v for k, v in chips_smart.items() if k != 'cond'}
tops_smart_make = {k: v for k, v in tops_smart.items() if k != 'cond'}

table_kwargs = {'number_format': '.1%', 'index_offset': 0.25, 'index_col_width': 1.25, 'header_row_height': 0.5,
                'skip_after_how_many_symbols': 13, 'dimensions': DIMENSIONS, 'columns_rename_dict': columns_rename_dict}

table_kwargs_nonull = {'number_format': '.1%', 'index_offset': 2.5, 'index_col_width': 1, 'header_row_height': 0.5,
                'skip_after_how_many_symbols': 13, 'dimensions': DIMENSIONS, 'columns_rename_dict': columns_rename_dict}

### First, simply compare total win percentage
table_data = {'Topspin Return': tops_naive_tot, 'Chipped Return': chips_naive_tot}
basic_naive_table_nonull = Table(table_data, **table_kwargs_nonull, null_columns = 0)
# basic_naive_table_nonull.render().show()

table_data = {'Topspin Return': tops_naive_tot, 'Chipped Return': chips_naive_tot}
basic_naive_table = Table(table_data, **table_kwargs, null_columns = 2)
# basic_naive_table.render().show()

table_data = {'Topspin Return': tops_smart_tot, 'Chipped Return (Choice)': chips_smart_tot}
basic_smart_table = Table(table_data, **table_kwargs, null_columns = 2)
# basic_smart_table.render().show()

### Now, add in make data
table_data = {'Topspin Return': tops_smart_make, 'Chipped Return (Choice)': chips_smart_make}
make_smart_table = Table(table_data, **table_kwargs, null_columns = 1)
# make_smart_table.render().show()

### finally, add in cond data
table_data = {'Topspin Return': tops_smart, 'Chipped Return (Choice)': chips_smart}
cond_smart_table =  Table(table_data, **table_kwargs, null_columns = 0)
# cond_smart_table.render().show()

###########################################
### Tables with incremental adjustments ###
###########################################

x, y = DIMENSIONS
index_name_t = 'Topspin Returns (on first serve)'
index_name_c = 'Chipped Returns (on first serve)'
index_name_a = 'Total' ### weirdly, need two extra spaces...
title = 'Returner Point Win%'

table_kwargs_small = {  'number_format': '.1%', 'index_offset': 0.25, 'index_col_width': 1.25, 'header_row_height': 0.5,
                        'skip_after_how_many_symbols': 15, 'dimensions': (x, y),
                        'skeleton_shrinkage': 0.2, 'skeleton_vertical_offset': 1, 
                        'title_vertical_offset': 0, 'title_horizontal_offset': 3.5,
                        'title_size': 2, 'title_thickness': 1, 'header_line_thickness': 0.1, 'resolution': TABLE_RESOLUTION,
                        }

table_kwargs_mid =  {   'number_format': '.1%', 'index_offset': 0.25, 'index_col_width': 1.25, 'header_row_height': 0.5,
                        'skip_after_how_many_symbols': 15, 'dimensions': (int(1.75 * x), int(1.75 * y)),
                        'skeleton_shrinkage': 0.2, 'skeleton_vertical_offset': 1, 
                        # 'title_vertical_offset': 0, 'title_horizontal_offset': 3.5,
                        'data_text_size': 2, 'header_text_size': 2, 'index_text_size': 2,
                        'title_size': 2, 'title_thickness': 1, 'header_line_thickness': 0.1, 'resolution': TABLE_RESOLUTION,
                        }

table_kwargs_large = {  'number_format': '.1%', 'data_right_adjust': False, 
                        'index_offset': 0.25, 'index_col_width': 1.25, 'header_row_height': 0.5,
                        'skip_after_how_many_symbols': 15, 'dimensions': (int(2.7 * x), int(2.7 * y)),
                        'skeleton_shrinkage': 0.15, 'skeleton_vertical_offset': 1,
                        'title_vertical_offset': -2, 'title_horizontal_offset': 19,
                        'title_size': 2, 'title_thickness': 1, 'header_line_thickness': 0.1,
                        'data_text_size': 2, 'header_text_size': 2, 'index_text_size': 2,
                        'title_color': const.BGR['bright_grey'], 'resolution': TABLE_RESOLUTION,
                        }

t0, t1, t2, t3 = tops_naive['total'], tops_smart['total'], tops_smart_bal['total'], tops_smart_bal_len['total']
c0, c1, c2, c3 = chips_naive['total'], chips_smart['total'], chips_smart_bal['total'], chips_smart_bal_len['total']

tDict = {'Unadjusted': t0}
cDict = {'Unadjusted': c0}
table_data = {index_name_t: tDict, index_name_c: cDict}

initial_naive_table = Table(table_data, **table_kwargs_small, title = title, header_text_size = 0)
# initial_naive_table.render().show()

naive_table = Table(table_data, **table_kwargs_large, title = title)
# naive_table.render().show()

tDict = {'Unadjusted': t0, '+Choice Adj.': t1}
cDict = {'Unadjusted': c0, '+Choice Adj.': c1}
table_data = {index_name_t: tDict, index_name_c: cDict}

smart_table = Table(table_data, **table_kwargs_large, title = title)
# smart_table.render().show()

tDict = {'Unadjusted': t0, '+Choice Adj.': t1, '+Balance Adj.': t2}
cDict = {'Unadjusted': c0, '+Choice Adj.': c1, '+Balance Adj.': c2}
table_data = {index_name_t: tDict, index_name_c: cDict}

smart_bal_table = Table(table_data, **table_kwargs_large, title = title)
# smart_bal_table.render().show()

tDict = {'Unadjusted': t0, '+Choice Adj.': t1, '+Balance Adj.': t2, '+Attrib. Adj.': t3}
cDict = {'Unadjusted': c0, '+Choice Adj.': c1, '+Balance Adj.': c2, '+Attrib. Adj.': c3}
table_data = {index_name_t: tDict, index_name_c: cDict}

smart_bal_len_table = Table(table_data, **table_kwargs_large, title = title)
# smart_bal_len_table.render().show()

r_f, c_f = 1080, 1920
tabFrm = naive_table.render()
r_t, c_t, _ = tabFrm.shape
r_d, c_d = (r_f - r_t) // 2, (c_f - c_t) // 2
table_offsetPt = Point2D((c_d, r_d))

###############################################
########### SHARE OF CHIPS AND TOPS ###########
###############################################

num_chip, num_tops = regFrm['is_chipped'].sum(), (~regFrm['is_chipped']).sum()  
num_tot = num_tops + num_chip

str_chips0 = f'{num_chip:>4}'
str_tops0 = f'{num_tops:>4}'

str_chips1 = f'{num_chip:>4} ({num_chip/num_tot:.0%})'
str_tops1 = f'{num_tops:>4} ({num_tops/num_tot:.0%})'

tDict = {'Sample Size': str_tops1}
cDict = {'Sample Size': str_chips1}
totDict = {'Sample Size': str(num_tot)}

table_data = {'': {'Sample Size': ''}, ' ': {'Sample Size': ''}, 'Total': totDict}
null_table = Table(table_data, **table_kwargs_mid, null_columns = 1,
                      title = "2021 Brest Challenger Tournament", title_color = const.BGR['bright_grey'],
                      title_horizontal_offset = 3.5)
# null_table.render().show()

table_data = {index_name_t: tDict, ' ': {'Sample Size': ''}, 'Total': totDict}
tops_table = Table(table_data, **table_kwargs_mid, null_columns = 1,
                      title = "2021 Brest Challenger Tournament", title_color = const.BGR['bright_grey'],
                      title_horizontal_offset = 3.5)
# tops_table.render().show()

table_data = {index_name_t: tDict, index_name_c: cDict, index_name_a: totDict}
chip_table = Table(table_data, **table_kwargs_mid, null_columns = 1,
                      title = "2021 Brest Challenger Tournament", title_color = const.BGR['bright_grey'],
                      title_horizontal_offset = 3.5)
# chip_table.render().show()

tDict = {'Sample Size': str_tops1, 'Returner Win%': ''}
cDict = {'Sample Size': str_chips1, 'Returner Win%': ''}
totDict = {'Sample Size': str(num_tot), 'Returner Win%': ''}
table_data = {index_name_t: tDict, index_name_c: cDict, index_name_a: totDict}

sample_and_win_table_null = Table(table_data, **table_kwargs_mid,
                            title = "2021 Brest Challenger Tournament", title_color = const.BGR['bright_grey'],
                            title_horizontal_offset = 7.5)
# sample_and_win_table_null.render().show()

tDict = {'Sample Size': str_tops1, 'Returner Win%': t0}
cDict = {'Sample Size': str_chips1, 'Returner Win%': ''}
totDict = {'Sample Size': str(num_tot), 'Returner Win%': ''}
table_data = {index_name_t: tDict, index_name_c: cDict, index_name_a: totDict}

sample_and_win_table_tops = Table(table_data, **table_kwargs_mid,
                            title = "2021 Brest Challenger Tournament", title_color = const.BGR['bright_grey'],
                            title_horizontal_offset = 7.5)
# sample_and_win_table_tops.render().show()

tDict = {'Sample Size': str_tops1, 'Returner Win%': t0}
cDict = {'Sample Size': str_chips1, 'Returner Win%': c0}
totDict = {'Sample Size': str(num_tot), 'Returner Win%': ''}
table_data = {index_name_t: tDict, index_name_c: cDict, index_name_a: totDict}

sample_and_win_table_chip = Table(table_data, **table_kwargs_mid,
                            title = "2021 Brest Challenger Tournament", title_color = const.BGR['bright_grey'],
                            title_horizontal_offset = 7.5)
# sample_and_win_table_chip.render().show()

tDict = {'Sample Size': str_tops1, 'Returner Win%': t0}
cDict = {'Sample Size': str_chips1, 'Returner Win%': c0}
totDict = {'Sample Size': str(num_tot), 'Returner Win%': (num_chip * c0 + num_tops * t0) / num_tot}
table_data = {index_name_t: tDict, index_name_c: cDict, index_name_a: totDict}

sample_and_win_table_all = Table(table_data, **table_kwargs_mid,
                            title = "2021 Brest Challenger Tournament", title_color = const.BGR['bright_grey'],
                            title_horizontal_offset = 7.5)
# sample_and_win_table_all.render().show()

###############################################
###### GET HIGH AND LOW PROPENSITY CHIPS ######
###############################################
dirFrm = regFrm[(regFrm['is_top_to_bot'])]

high_pred_exFrm = dirFrm.sort_values('pred_is_chipped').iloc[-8:]
high_pred_exFrms = [retPnts[idx][:2].stream(confs = confs).join().maybe_to_stream().decap().frames for idx in high_pred_exFrm.index]

low_pred_exFrm = dirFrm[dirFrm.is_chipped].sort_values('pred_is_chipped').iloc[:8]
low_pred_exFrms = [retPnts[idx][:2].stream(confs = confs).join().maybe_to_stream().decap().frames for idx in low_pred_exFrm.index]

###################################
#### FRAMES RASTER WITH TABLE #####
###################################
from ATMANpy.types.Overlays import FrameRaster, FramesRaster, frame_cascade, reverse_cascade
from ATMANpy.types.Frames import BGRFrame

STATIC_FRAMES = 25
TRANSITION_FRAMES = 100
all_frames = []
script_lines = []

blankFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))

# script_line =   """
#                     At the most basic level, there are two ways to return a first-serve. 
#                     A player can either adjust his grip to his or her preferred backhand or forehand grip, 
#                     and attempt to return the ball with topspin,
#                     or he or she can keep a neutral, continental grip, and chip the ball back into play.
#                     The resulting chip will have underspin, but, as we will see in the course of this analysis, 
#                     it will have typically less RPM, greater net clearance, and lower speed than a slice employed in a rally.
#                 """

script_line =   """
                    At the most basic level, there are two ways to return a first-serve: 
                    a player can return the ball with topspin, or chip the ball back into play with backspin. 
                """

titleStr = 'On the Effectiveness of the Chipped Return'
titleFrm = blankFrm.put(titleStr, size = 2, where = (600, 540))

titleFrms = BGRFrames.from_frame(titleFrm, STATIC_FRAMES)
all_frames.append(titleFrms)
script_lines.append(script_line)

script_line =   """
                    The chipped return is infrequently employed; 
                    in our sample of 1,451 first-serve returns observed in 20 matches of the 2021 Brest Challenger Tournament...
                """
fadeFrms = titleFrms[:50].dim_progressively()
blackFrms = BGRFrames.from_frame(BGRFrame.make_black(), 25)

tabFrm = overlay_center(BGRFrame.make_black(), null_table.render())
tabFrms = BGRFrames.from_frame(tabFrm, 25)

jointFrms = BGRFrames.from_frames([fadeFrms, blackFrms, tabFrms])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    72% of first serves are returned with topspin, ...
                """
tabFrm = overlay_center(BGRFrame.make_black(), tops_table.render())
tabFrms = BGRFrames.from_frame(tabFrm, 25)
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   """
                    ... and only 28% of first-serves are returned as a chip, 
                    and even fewer, as we will see, are chipped back by choice.
                """
tabFrm = overlay_center(BGRFrame.make_black(), chip_table.render())
tabFrms = BGRFrames.from_frame(tabFrm, 25)
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   """
                    However, some highly successful players consistently use the chipped return as a tool 
                    to either neutralize the rally with a deep, floating chip, 
                    or draw the server into the court with a shallow, low-bouncing chip.
                """
stillFrms = BGRFrames.from_frame(tabFrm, 200)
fadeFrms = BGRFrames.from_frame(tabFrm, 100).dim_progressively()
jointFrms = BGRFrames.from_frames([stillFrms, fadeFrms])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    The primary goal of this analysis is to take a good look at these chip return varieties, 
                    and try to understand their effectiveness compared to the topspin return, as measured by the expected probability to win the point.
                """

someStr = "The primary goal of this analysis is to\nunderstand the effectiveness of\nthe chip return compared to the topspin return,\nas measured by\nthe expected probability to win the point."

import cv2 as cv

def typewriter_effect(someStr, frames_per_char = 2,
                      text_where = (800, 500), text_color = const.BGR['leafy_green'], text_size = 1, text_font = cv.FONT_HERSHEY_PLAIN,
                      align = 'left', manual_aligners = None):
    emptyFrm = BGRFrame.make_black()
    
    lines = someStr.split('\n')

    ### figure out how to center text
    if align == 'left':
        spaces = [0 for l in lines]
    elif align == 'center':
        lens = [len(l) for l in lines]
        m = max(lens)
        spaces = [(m - l) // 2 for l in lens]
    elif align == 'right':
        lens = [len(l) for l in lines]
        m = max(lens)
        spaces = [(m - l) for l in lens]

    if manual_aligners is not None:
        spaces = [s + m for s, m in zip(spaces, manual_aligners)]

    out_frames = []
    prev_lines = ['']
    for idx in range(len(lines)):
        lineStr = lines[idx]
        num_spaces = spaces[idx]

        remaining_lines = len(lines) - len(prev_lines) + 1
        # prevStr = '\n' * remaining_lines + '\n'.join(prev_lines) 
        prevStr = '\n'.join(prev_lines)

        curr_chars = []
        for curr_char in lineStr:
            curr_chars.append(curr_char)

            currStr = ' ' * num_spaces + ''.join(curr_chars)
            currFrm = emptyFrm.put(prevStr + '\n' + currStr, where = text_where, color = text_color, size = text_size, font = text_font)
            currFrms = BGRFrames.from_frame(currFrm, frames_per_char)
            out_frames.append(currFrms)
        
        prev_lines.append(currStr)
    
    return BGRFrames.from_frames(out_frames)

typeFrms = typewriter_effect(someStr, frames_per_char = 1, 
                             text_where = (520, 420), text_color = const.BGR['bright_grey'], text_size = 1, text_font = cv.FONT_HERSHEY_TRIPLEX,
                             align = 'center', manual_aligners = [2, 2, 0, 2, 2])
# DecapStream(typeFrms, 0, 100).show()
# typeFrms[-1].show()

fadeFrms = BGRFrames.from_frame(typeFrms[-1], 50).dim_progressively()
jointFrms = BGRFrames.from_frames([typeFrms, fadeFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   "At first glance, it doesn't look good for the chipped return."

tabFrm = overlay_center(BGRFrame.make_black(), sample_and_win_table_null.render())
tabFrms = BGRFrames.from_frame(tabFrm, 25)
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   "On average, in our sample, the returner wins around 40% of points whenever he returns topspin, ..."

tabFrm = overlay_center(BGRFrame.make_black(), sample_and_win_table_tops.render())
tabFrms = BGRFrames.from_frame(tabFrm, 25)
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   "... versus only around 20% of points when he chips the ball back."

tabFrm = overlay_center(BGRFrame.make_black(), sample_and_win_table_chip.render())
tabFrms = BGRFrames.from_frame(tabFrm, 25)
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   """
                    However, this view is slightly unfair toward the chipped return. 
                    To understand why, carefully consider these eight returns, and what they have in common.
                """
tabFrm = overlay_center(BGRFrame.make_black(), sample_and_win_table_all.render())
tabFrms = BGRFrames.from_frame(tabFrm, 25)
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line = ""

high_pred_exFrms_tight = [myFrms[:-15] for myFrms in high_pred_exFrms] ## shorten the example frames a bit, to tighten the flow
silentFrms = BGRFrames.from_frames(high_pred_exFrms_tight[:3]).rescale(3)
all_frames.append(silentFrms)
script_lines.append(script_line)

script_line =   """
                    All these returns have three things in common. They're played off the backhand, on the stretch, as a chip!
                    In fact, there is virtually no way to return these with topspin.
                    On these serves, the returner has no choice: he is forced to block the ball back in, 
                    usually resulting in a short chip that is easy to attack if it lands fair at all.
                    
                    For the lack of choice, there is nothing to discuss:
                    The serve is almost an ace - do what you can, chip it back in, and pray for the best.
                    
                    What can be discussed are returns that are chipped by choice: given that the returner could either play a topspin or a chip return, 
                    we want to assess the quality of said choice. 
                    
                    Therefore, we shall have to put in some work to find out on which serves the returner actually had a choice.
                """

voicedFrms = BGRFrames.from_frames(high_pred_exFrms_tight[3:]).rescale(3)
all_frames.append(voicedFrms)
script_lines.append(script_line)

#################################
##### PROPENSITY SCORE MODEL ####
#################################
 
### Propensity Wording:
# To be able to differentiate between forced chips and choice chips, we employ a technique called Propensity Score Matching, or PSM for short.
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

script_line =   """
                    To be able to differentiate between forced chips and choice chips, 
                    we employ a technique called Propensity Score Matching, or PSM for short.
                """

ball_color = const.BGR['ball_yellow']
which_example = 745

high_pred_exSer = dirFrm.sort_values('pred_is_chipped').iloc[which_example]
which_example_idx = high_pred_exSer.name
srvShot = retPnts_pre[int(which_example_idx)][0]
srvFrms = srvShot.stream(confs = confs).decap().frames.rescale(3)

srvFrms_pre = srvShot.stream(confs = confs).widen(50).decap().frames[:50].rescale(3)

shot_splines = srvShot.feat_spline(which_endpoint = 'penultimate').values()
serveScns = [Scene([spln.recolor(ball_color).resize(1.5)]) for spln in shot_splines if spln is not None]
serveScns = [Scene.make_empty(), *serveScns]
len(serveScns)

### interpolate between last voicedFrms and first srvFrms
itplFrms = vs.interpolate_frames(voicedFrms[-1], srvFrms_pre[0], 50)

### get the stats
speed = high_pred_exSer['serve_speed']
depth = high_pred_exSer['serve_depth']
height = high_pred_exSer['serve_height']
is_back = high_pred_exSer['ret_is_back']
ret_dist = high_pred_exSer['ret_dist']
pred = high_pred_exSer['pred_is_topspin']

### read players to be able to stagger their shapes
from ATMANpy.types.Features import StreamFeatures
from ATMANpy.types.Frames import MaskFrame
from ATMANpy.types.Contours import Contours

mtc_idx = srvShot.get_match_index()
vid, sto = confs[mtc_idx]['video_id'], confs[mtc_idx]['store_name']
plSF = StreamFeatures.from_store(vid, sto, featnameList = ['botPlayerCnt', 'topPlayerCnt'])
mySF = plSF[srvShot.get_frames()[:-1]]

prevMask = MaskFrame.false(srvFrms[-1])
prevFrm = None
out_frames = []
for frameNum, frame in zip(mySF, srvFrms):
    featDict = mySF[frameNum]
    botCnt, topCnt = featDict['botPlayerCnt'].rescale(3), featDict['topPlayerCnt'].rescale(3)    
    
    darkMask = MaskFrame.make_black()
    playMask = darkMask.within_contour(topCnt) | darkMask.within_contour(botCnt)
    maskFrm = frame.mask(playMask)

    if prevFrm is not None:
        currFrm = prevFrm.mask(prevMask) + frame.mask(~prevMask)
    else:
        currFrm = frame
    
    out_frames.append(currFrm)

    prevMask = prevMask | playMask
    prevFrm = currFrm
    
outFrms = BGRFrames.from_single_frames(out_frames)    
# DecapStream(outFrms, 0, 100).show()
# outFrms[-1].show()

arg_res = 20

animFrms, animDict = vs.animate_sequence(serveScns, start_cam = brestCam_full, stop_cam = brestCam_full, 
                                animate_over = outFrms, cumulative_foreground_transparence = 0.9, cumulate_argument = True,
                                argument_resolution = arg_res, foreground_resolution = arg_res)
DecapStream(animFrms, 0, 100).show(ms_between_frames = 40)
# animFrms[-1].show()

### dim the frame, without dimming the features
prevScn = animDict['previous_foreground_scene']

dim_frames = 25
dim_backFrms = BGRFrames.from_frame(outFrms[-1], dim_frames).dim_progressively(1, 0.5)
dimFrms, animDict = vs.animate_sequence([prevScn] * dim_frames, start_cam = brestCam_full, stop_cam = brestCam_full, 
                                        argument_resolution = arg_res, foreground_resolution = arg_res,
                                        animate_over = dim_backFrms)
DecapStream(dimFrms, 0, 100).show(ms_between_frames = 40)
# dimFrms[-1].show()

jointFrms = BGRFrames.from_frames([itplFrms, srvFrms_pre, animFrms, dimFrms])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    The PSM looks at the speed of the serve...
                """

### speed arrow:
from ATMANpy.types.ShapesTennis3D import Arrow
arr_color = const.BGR['sign_orange']
text_color = const.BGR['salmon_pink']
arg_lum = 1.25

arr_params = {'hat_length': 10, 'thickness': 1, 'color': arr_color}

speedLn = serveScns[1][0]
speedLn = speedLn.lengthen_by(2, which = 'right')
speedArr = Arrow.make(speedLn.leftPt, speedLn.rightPt, **arr_params)

speedPt = (speedLn.leftPt + speedLn.rightPt) / 2
speedPt = speedPt.annotate(f'Speed: {speed:.0f} km/h').recolor(text_color)
speedPt = speedPt.right(10).down(5)

statFrms, animDict = vs.animate_static(speedArr, static_cam = brestCam_full, animate_over = dimFrms[-1],
                                        annotated_points = [speedPt], 
                                        argument_resolution = arg_res, foreground_resolution = arg_res,
                                        argument_luminosity = arg_lum, num_frames = 25,
                                        **animDict)
# statFrms[-1].show()

all_frames.append(statFrms)
script_lines.append(script_line)

script_line =   """
                    ... the height of the bounce...
                """

### height arrow:
from ATMANpy.types.Shapes3D import Point3D, Line3D
x, y, z = srvShot.get_apex_post_bounce()
postPnt = Point3D((x, y, z))
postPnt_ground = Point3D((x, y, 0))

heightLn = Line3D(postPnt_ground, postPnt).shorten_by(0.15, which = 'both')
postPnt_ground, postPnt = heightLn.leftPt, heightLn.rightPt

heightArr = Arrow.make(postPnt_ground, postPnt, **arr_params)

heightPt = (postPnt + postPnt_ground) / 2
heightPt = heightPt.annotate(f'Bounce Height: {height:.1f} m').recolor(text_color)
heightPt = heightPt.right(10).up(7.5)

statFrms, animDict = vs.animate_static(heightArr, static_cam = brestCam_full, animate_over = dimFrms[-1],
                                annotated_points = [heightPt], argument_resolution = arg_res, foreground_resolution = arg_res,
                                argument_luminosity = arg_lum, num_frames = 25,
                                **animDict)
# statFrms[-1].show()

all_frames.append(statFrms)
script_lines.append(script_line)


script_line =   """
                    ... how deep the returner is positioned ...
                """

### depth arrow:
x, y = srvShot.recPlayer.get_initial_position()
y_baseline = CourtLines.make()['Bottom Baseline'].leftPt.y

depthPnt = Point3D((x, y, 0))
depthPnt_base = Point3D((x, y_baseline, 0))

depthLn = Line3D(depthPnt_base, depthPnt).shorten_by(0.15, which = 'both')
depthPnt_base, depthPnt = depthLn.leftPt, depthLn.rightPt

depthArr = Arrow.make(depthPnt_base, depthPnt, **arr_params)

depthPt = (depthPnt + depthPnt_base) / 2
depthPt = depthPt.annotate(f'Return Depth: {depth:.1f} m').recolor(text_color)
depthPt = depthPt.right(10)

statFrms, animDict = vs.animate_static(depthArr, static_cam = brestCam_full, animate_over = dimFrms[-1],
                                annotated_points = [depthPt], argument_resolution = arg_res, foreground_resolution = arg_res,
                                argument_luminosity = arg_lum, num_frames = 25,
                                **animDict)
# statFrms[-1].show()

all_frames.append(statFrms)
script_lines.append(script_line)


script_line =   """
                    ... whether the serve is to the backhand or forehand ...
                """
### bh arrow
x, y = srvShot.recPlayer.get_initial_position()

retPnt = Point3D((x, y, 0)).back(25)
retPnt_bh = retPnt.left(70)

retLn = Line3D(retPnt, retPnt_bh).shorten_by(0.1, which = 'both')
retPnt, retPnt_bh = retLn.leftPt, retLn.rightPt

retArr = Arrow.make(retPnt, retPnt_bh, **arr_params)

retPt = (retPnt + retPnt_bh) / 2
retPt = retPt.annotate(f'Hit to Backhand').recolor(text_color)
retPt = retPt.back(10).left(15)

statFrms, animDict = vs.animate_static(retArr, static_cam = brestCam_full, animate_over = dimFrms[-1],
                                annotated_points = [retPt], argument_resolution = arg_res, foreground_resolution = arg_res,
                                argument_luminosity = arg_lum, num_frames = 25,
                                **animDict)
# statFrms[-1].show()

all_frames.append(statFrms)
script_lines.append(script_line)

script_line =   """
                    ... and how far the returner has to stretch to get to the ball.
                """

### dist arrow
x, y = srvShot.recPlayer.get_initial_position()
x2, y2 = srvShot.get_ball_last_position()[:2]

retPnt = Point3D((x, y, 0)).back(10)
hitPnt = Point3D((x2, y2, 0)).back(10)

stretch_dist = retPnt.distance_to(hitPnt) / const.SCALE * const.FOOT_TO_METER

stretchLn = Line3D(retPnt, hitPnt).shorten_by(0.1, which = 'both')
retPnt, hitPnt = stretchLn.leftPt, stretchLn.rightPt

stretchArr = Arrow.make(retPnt, hitPnt, **arr_params)

stretchPt = (retPnt + hitPnt) / 2
stretchPt = stretchPt.annotate(f'Stretch Distance: {stretch_dist:.1f} m').recolor(text_color)
stretchPt = stretchPt.back(15).left(55)

statFrms, animDict = vs.animate_static(stretchArr, static_cam = brestCam_full, animate_over = dimFrms[-1],
                                annotated_points = [stretchPt], argument_resolution = arg_res, foreground_resolution = arg_res,
                                argument_luminosity = arg_lum, num_frames = 25,
                                **animDict)
# statFrms[-1].show()

all_frames.append(statFrms)
script_lines.append(script_line)

script_line =   """
                    From this data, the PSM predicts the probability of the ball being chipped back by the returner, 
                    as opposed to being returned with topspin.
                """

### fade into virtual court
back_res = 5
arrScn_pre = Scene([speedArr, depthArr, heightArr, retArr, stretchArr])
arrPts_pre = [speedPt, depthPt, heightPt, retPt, stretchPt]
virtFrms, animDict2 = vs.animate_static(arrScn_pre, static_cam = brestCam_full, 
                                        background_scene = Scene.make_court(),
                                        annotated_points = arrPts_pre, 
                                        argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                        )

itplFrms = vs.interpolate_frames(statFrms[-1], virtFrms[0], 50)
DecapStream(itplFrms, 0, 100).show()

#### make into diagram
from ATMANpy.auxil.court import birds_eye_service_box_cam
bot_adCam = birds_eye_service_box_cam(which = 'Ad', side = 'Bot', height = 180)
bot_dcCam = birds_eye_service_box_cam(which = 'Deuce', side = 'Bot', height = 180)

arr_len = 22
arr_off = 10
arr_side = 15
arr_text = 25
rot_angle = np.pi / 2

p1 = Point3D((200, 645, 0))
p2 = p1.back(arr_len)
speedArr_diag = Arrow.make(p2, p1.right(arr_side), **arr_params, rot_angle = rot_angle)

p3 = p2.back(arr_off)
p4 = p3.back(arr_len)
heightArr_diag = Arrow.make(p3, p4.right(arr_side), **arr_params, rot_angle = rot_angle)

p5 = p4.back(arr_off)
p6 = p5.back(arr_len)
depthArr_diag = Arrow.make(p5, p6.right(arr_side), **arr_params, rot_angle = rot_angle) 

p7 = p6.back(arr_off)
p8 = p7.back(arr_len)
stretchArr_diag = Arrow.make(p8, p7.right(arr_side), **arr_params, rot_angle = rot_angle)

p9 = p8.back(arr_off)
p10 = p9.back(arr_len)
retArr_diag = Arrow.make(p10, p9.right(arr_side), **arr_params, rot_angle = rot_angle)

speedPt_post = (p1 + p2) / 2
heightPt_post = (p3 + p4) / 2
depthPt_post = (p5 + p6) / 2
stretchPt_post = (p7 + p8) / 2
retPt_post = (p9 + p10) / 2

speedPt_post = speedPt_post.annotate(speedPt.text).right(arr_text).recolor(text_color)
heightPt_post = heightPt_post.annotate(heightPt.text).right(arr_text).recolor(text_color)
depthPt_post = depthPt_post.annotate(depthPt.text).right(arr_text).recolor(text_color)
stretchPt_post = stretchPt_post.annotate(stretchPt.text).right(arr_text).recolor(text_color)
retPt_post = retPt_post.annotate(retPt.text).right(arr_text).recolor(text_color)

arrScn_post = Scene([speedArr_diag, depthArr_diag, heightArr_diag, retArr_diag, stretchArr_diag])

move_frames = 50
itplScns = vs.interpolate_scenes(arrScn_pre, arrScn_post, move_frames)

moveFrms, _ = vs.animate_sequence(itplScns, start_cam = brestCam_full, stop_cam = bot_adCam, background_scene = Scene.make_court(),
                                  argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                  interpolate_func = vs.interpolate_camera_naive,
                                )
# DecapStream(moveFrms, 0, 100).show()

### overlay the text.
from ATMANpy.types.Visual3D import View
itplCams = vs.interpolate_camera_naive(brestCam_full, bot_adCam, move_frames)

out_frames = []
for i in range(move_frames):
    lmbd = i / (move_frames - 1)
    myCam = itplCams[i]
    myArr = moveFrms[i].array

    speedPt_itpl = speedPt * (1 - lmbd) + speedPt_post * lmbd
    heightPt_itpl = heightPt * (1 - lmbd) + heightPt_post * lmbd
    depthPt_itpl = depthPt * (1 - lmbd) + depthPt_post * lmbd
    stretchPt_itpl = stretchPt * (1 - lmbd) + stretchPt_post * lmbd
    retPt_itpl = retPt * (1 - lmbd) + retPt_post * lmbd
    
    speedPt_itpl = speedPt_itpl.annotate(speedPt.text).recolor(text_color)
    heightPt_itpl = heightPt_itpl.annotate(heightPt.text).recolor(text_color)
    depthPt_itpl = depthPt_itpl.annotate(depthPt.text).recolor(text_color)
    stretchPt_itpl = stretchPt_itpl.annotate(stretchPt.text).recolor(text_color)
    retPt_itpl = retPt_itpl.annotate(retPt.text).recolor(text_color)

    myView = View.make(myCam, Scene.make_empty(), annotatedPoints = [speedPt_itpl, heightPt_itpl, depthPt_itpl, stretchPt_itpl, retPt_itpl])
    myFrm = myView.snap(over_array = myArr)
    out_frames.append(myFrm)

outFrms = BGRFrames.from_single_frames(out_frames)
# DecapStream(outFrms, 0, 100).show()
# outFrms[-1].show()

probPt1 = depthPt_post.right(85).retext('Probability to Chip:').recolor(const.BGR['light_brown']).resize(2)
probPt2 = depthPt_post.right(146).retext(f'{1 - pred:.1%}').recolor(const.BGR['light_brown']).resize(2)

probFrms, animDict = vs.animate_static(arrScn_post, static_cam = bot_adCam, background_scene = Scene.make_court(), 
                                argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                annotated_points = [speedPt_post, heightPt_post, depthPt_post, stretchPt_post, retPt_post, probPt1, probPt2],
                                )
# probFrms[-1].show()

itplFrms2 = vs.interpolate_frames(outFrms[-1], probFrms[0], 50)
# DecapStream(itplFrms2, 0, 100).show()

jointFrms = BGRFrames.from_frames([itplFrms, outFrms, itplFrms2, probFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    The arrows here indicate how each factor influences the predicted probability to chip.
                    A chip return is LIKELIER if...
                """

all_frames.append(probFrms)
script_lines.append(script_line)

brighten_factor = 2
bright_text_color = tuple([min(c, 255) for c in np.array(text_color) * brighten_factor])

script_line =   """
                    ... the serve is faster ...
                """

probFrms1, _ = vs.animate_static(speedArr_diag.dim(brighten_factor), static_cam = bot_adCam, background_scene = Scene.make_court(), 
                                argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                annotated_points = [speedPt_post.recolor(bright_text_color), probPt1, probPt2],
                                **animDict
                                )
# probFrms1[-1].show()

all_frames.append(probFrms1)
script_lines.append(script_line)

script_line =   """
                    ... the ball bounces lower ...
                """

probFrms2, _ = vs.animate_static(heightArr_diag.dim(brighten_factor), static_cam = bot_adCam, background_scene = Scene.make_court(), 
                                argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                annotated_points = [heightPt_post.recolor(bright_text_color), probPt1, probPt2],
                                **animDict
                                )
# probFrms2[-1].show()

all_frames.append(probFrms2)
script_lines.append(script_line)

script_line =   """
                    ... the returner is closer to the baseline ...
                """

probFrms3, _ = vs.animate_static(depthArr_diag.dim(brighten_factor), static_cam = bot_adCam, background_scene = Scene.make_court(), 
                                argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                annotated_points = [depthPt_post.recolor(bright_text_color), probPt1, probPt2],
                                **animDict
                                )
# probFrms3[-1].show()

all_frames.append(probFrms3)
script_lines.append(script_line)

script_line =   """
                    ... the returner has to stretch farther to reach the serve ...
                """

probFrms4, _ = vs.animate_static(stretchArr_diag.dim(brighten_factor), static_cam = bot_adCam, background_scene = Scene.make_court(), 
                                argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                annotated_points = [stretchPt_post.recolor(bright_text_color), probPt1, probPt2],
                                **animDict
                                )
# probFrms4[-1].show()

all_frames.append(probFrms4)
script_lines.append(script_line)


script_line =   """
                    ... and finally if the serve is to the backhand.
                """

probFrms5, _ = vs.animate_static(retArr_diag.dim(brighten_factor), static_cam = bot_adCam, background_scene = Scene.make_court(), 
                                argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                annotated_points = [retPt_post.recolor(bright_text_color), probPt1, probPt2],
                                **animDict
                                )
# probFrms5[-1].show()

all_frames.append(probFrms5)
script_lines.append(script_line)


script_line =   """
                    So how does this propensity model help us differentiate between chips that are forced and chips that are done by choice?
                """

itplFrms = vs.interpolate_frames(probFrms5[-1], probFrms[0], 50)
# DecapStream(itplFrms, 0, 100).show()

moveFrms, _ = vs.animate_cam_motion(None, start_cam = bot_adCam, stop_cam = bot_dcCam, num_frames = 25, interpolate_func = vs.interpolate_camera_naive,
                                    background_scene = Scene.make_court(),
                                    argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                    **animDict)
# DecapStream(moveFrms, 0, 100).show()
# moveFrms[-1].show()

number_line_color = const.BGR['bright_grey']
grow_frames = 25

p1, p2 = stretchPt_post.right(82), stretchPt_post.right(166)

p1_annot = p1.retext(f'{0:.0%}').recolor(number_line_color).back(10).left(2).resize(1)
p2_annot = p2.retext(f'{1:.0%}').recolor(number_line_color).back(10).left(6).resize(1)

numlnArr_null = Arrow.make(p1, p1.right(1), color = number_line_color, rot_angle = np.pi / 2, hat_length = 1, thickness = 1)
numlnArr = Arrow.make(p1, p2, color = number_line_color, rot_angle = np.pi / 2, hat_length = 10, thickness = 1)

nullScn, arrScn = Scene([numlnArr_null]), Scene([numlnArr])

itplScns = vs.interpolate_scenes(nullScn, arrScn, grow_frames)

animDict_no_prob_1 = {**animDict}
animDict_no_prob_1['previous_annotated_points'] = animDict_no_prob_1['previous_annotated_points'][:-1]

growFrms, _ = vs.animate_sequence(  itplScns, start_cam = bot_dcCam, stop_cam = bot_dcCam, background_scene = Scene.make_court(),
                                    argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                    annotated_points = [p1_annot],
                                    **animDict_no_prob_1)
# DecapStream(growFrms, 0, 100).show()

probPt2_target = probPt2.back(40).left(10)

out_frames = []
for i in range(grow_frames):
    lmbd = i / (grow_frames - 1)
    probPt2_itpl = probPt2 * (1 - lmbd) + probPt2_target * lmbd
    probPt2_itpl = probPt2_itpl.annotate(probPt2.text).recolor(probPt2.color).resize(probPt2.size)

    myFrm = growFrms[i]
    myView = View.make(bot_dcCam, Scene.make_empty(), annotatedPoints = [probPt2_itpl])
    outFrm = myView.snap(over_array = myFrm.array)
    out_frames.append(outFrm)

outFrms = BGRFrames.from_single_frames(out_frames)
# DecapStream(outFrms, 0, 100).show()
# outFrms[-1].show()

statFrms, animDict = vs.animate_static(numlnArr, static_cam = bot_dcCam, background_scene = Scene.make_court(),
                                argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                annotated_points = [p1_annot, p2_annot, probPt2_itpl],
                                **animDict_no_prob_1)
# statFrms[-1].show()

jointFrms = BGRFrames.from_frames([moveFrms, outFrms, statFrms])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    If the PSM predicts a high propensity to chip, 
                    then it effectively believes that there was no way that specific serve could have been returned with topspin.
                    It is a forced chip return.
                """

forcePt = p2.front(10).retext('Forced Chip').rethicken(2).left(13).recolor(number_line_color)

statFrms_force, animDict = vs.animate_static(None, static_cam = bot_dcCam, background_scene = Scene.make_court(),
                                            argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                            annotated_points = [forcePt],
                                            **animDict)

blinkFrms = vs.blinking(statFrms_force[-1], statFrms[-1], 2, 250)
# DecapStream(blinkFrms, 0, 100).show()
# blinkFrms[-1].show()

all_frames.append(blinkFrms)
script_lines.append(script_line)

script_line =   """
                    On the other hand, Let's say we observe a chip in a situation where the propensity of playing a chip is low, 
                    that is, where we would usually see a topspin return.
                    In this case, the chip return is the returner's strong choice: he could have played topspin, but he went for the chip.
                """

choicePt = p1.front(10).retext('Choice Chip').rethicken(2).left(3).recolor(number_line_color)

statFrms_choice, animDict = vs.animate_static(None, static_cam = bot_dcCam, background_scene = Scene.make_court(),
                                            argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                            annotated_points = [choicePt],
                                            **animDict)

blinkFrms = vs.blinking(statFrms_choice[-1], statFrms_force[-1], 2, 250)
# DecapStream(blinkFrms, 0, 100).show()

all_frames.append(blinkFrms)
script_lines.append(script_line)

script_line =   """
                    To illuminate this, consider the 8 lowest propensity chipped returns.
                """
### move out of the virtual view, and blend into court

moveFrms_blank, _ = vs.animate_cam_motion(None, start_cam = bot_dcCam, stop_cam = brestCam_full, background_scene = Scene.make_court(),
                                          argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                          num_frames = 60, interpolate_func = vs.interpolate_camera_naive, start_later = 20)

moveFrms_prev, _ = vs.animate_cam_motion(None, start_cam = bot_dcCam, stop_cam = brestCam_full, background_scene = Scene.make_court(),
                                          argument_resolution = arg_res, foreground_resolution = arg_res, background_resolution = back_res,
                                          num_frames = 60, interpolate_func = vs.interpolate_camera_naive, start_later = 20,
                                          **animDict)

moveFrms = moveFrms_prev.dim_progressively(1, 0, exp = 0.5) + moveFrms_blank.dim_progressively(0, 1, exp = 0.5)

DecapStream(moveFrms, 0, 100).show()

trans_frames = 50
first_idx = int(low_pred_exFrm.index[0])
firstFrms = retPnts_pre[first_idx][:2].stream(confs = confs).join().maybe_to_stream().widen(trans_frames).decap().frames[:trans_frames].rescale(3)

stillFrms = BGRFrames.from_frame(moveFrms[-1], trans_frames)

transFrms = stillFrms.dim_progressively(1, 0) + firstFrms.dim_progressively(0, 1)

lo_pred_exFrms_tight = [myFrms[:-15] for myFrms in low_pred_exFrms] ## shorten the example frames a bit, to tighten the flow
silentFrms = BGRFrames.from_frames(lo_pred_exFrms_tight[:3]).rescale(3)

jointFrms = BGRFrames.from_frames([moveFrms, transFrms, silentFrms])
DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                On these returns, it does indeed seem that the returning player CHOSE to chip the ball back, 
                where he could very well have played a topspin return.
                """

voicedFrms = BGRFrames.from_frames(lo_pred_exFrms_tight[3:]).rescale(3)
all_frames.append(voicedFrms)
script_lines.append(script_line)

script_line =   """
                    We can now use these propensities to focus on chipped returns where the returner actually had a choice, 
                    and thereby assess the quality of this choice.
                """
fadeFrms = BGRFrames.from_frame(voicedFrms[-1], 100).dim_progressively(1, 0)
all_frames.append(fadeFrms)
script_lines.append(script_line)

script_line =   """
                    As we saw earlier, the probability to win a point following a chipped return is just under 20%. 
                    If we concentrate on choice chips, we expect that probability to increase, since we discount
                    those serves that are returned on the stretch, under great difficulty.
                """

tabFrm = naive_table.render()
tabFrm = overlay_center(BGRFrame.make_black(), tabFrm)
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)

all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   """
                    Indeed, the returner's win probability increases, if only modestly, to 23.3%.  
                """

tabFrm = smart_table.render()
tabFrm = overlay_center(BGRFrame.make_black(), tabFrm)
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)

all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   """
                    Unfortunately, even after adjusting for choice chips, the data is not yet ready to reveal a true, unbiased picture
                    of return win probabilities. 

                    Why? Well, in a perfect world, we'd ask each player to hit 100 serves to each other player in our sample.
                    This way, we'd observe equally many serves and returns for each single player.
                    However, our data is generated not by experiment, but in the course of actual tennis matches. 
                    In a tennis match, the amount of returns observed by player is systematically biased: 
                    what happens is that we observe more returns from the better player.
                """
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   "To illuminate why, imagine player A has a great serve and average return; player B has an average return and serve."

myRst = FrameRaster(2, 2, borders = None)
myRst.add(ex1Frm, 0, 0)
rstFrm = myRst.render()
rstFrms = BGRFrames.from_frame(rstFrm, STATIC_FRAMES)
all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   "We expect player A to hold faster (due to his strong serve), and therefore play relatively fewer points on serve and relatively more on the return."

myRst.add(ex1Frm_with_text, 0, 0)
rstFrm = myRst.render()
rstFrms = BGRFrames.from_frame(rstFrm, STATIC_FRAMES)
all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   "Similarly, if A had a great return and an average serve..."

myRst.add(ex2Frm, 1, 1)
rstFrm = myRst.render()
rstFrms = BGRFrames.from_frame(rstFrm, STATIC_FRAMES)
all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   "... we'd expect him to hang in for longer in return games, thus playing relatively more points on return, again."

myRst.add(ex2Frm_with_text, 1, 1)
rstFrm = myRst.render()
rstFrms = BGRFrames.from_frame(rstFrm, STATIC_FRAMES)
all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   """
                    Yet another way to show this is to look at what happens to win% if rallys go on for long. 
                    When a point goes on for a long time, the impact of the serve and return is nullified.
                    If both players were equally matched, we'd expect the win probability on long points to be at 50%.

                    So let us take a look at the returner's win probability given the length of rallys.
                """

all_frames.append(rstFrms)
script_lines.append(script_line)

graphScns_itpl = vs.interpolate_scenes(bar_graphScn_null, bar_graphScn, num_frames = TRANSITION_FRAMES)
myFrms, _ = vs.animate_sequence(graphScns_itpl, start_cam = myGraph.cam, stop_cam = myGraph.cam,
                                annotated_points = bar_annotatedPnts_start, argument_resolution = BAR_RESOLUTION)

ovrFrms = overlay_center(BGRFrames.from_frame(BGRFrame.make_black(), 100), myFrms)

script_line = ""
all_frames.append(ovrFrms)
script_lines.append(script_line)

# vis_ut.show(all_frames)

script_line =  """
                    We observe that the return player has greater win probability in long rallys, 
                    indicating that the better player indeed plays more returns in a match, simply due to the nature of tennis scoring.

                    To remove this bias, and approximate the experimental setting, 
                    we introduce a balancing factor that gives more weight to returns made by the player for whom we observe fewer returns, overall. 
                    Ostensibly, this is the weaker player.
                """

myFrms, _ = vs.animate_static(graphScns_itpl[-1], static_cam = myGraph.cam, annotated_points = bar_annotatedPnts, 
                                argument_resolution = BAR_RESOLUTION, num_frames = STATIC_FRAMES)
ovrFrms = overlay_center(BGRFrames.from_frame(BGRFrame.make_black(), 25), myFrms)

all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    The balance adjustment does not greatly impact either win percentage beyond a slight reduction. A reduction is to be expected, since we're
                    counteracting the impact of the better player being on return more often solely by virtue of the data generating process.
                """

tabFrm = smart_bal_table.render()
tabFrm = overlay_center(BGRFrame.make_black(), tabFrm)
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   """
                    Our third and final adjustment is connected to something we have just mentioned. 
                    The longer a rally goes on, the less the serve and the return matter to the outcome of the point.
                    Therefore, the longer a rally goes on, the less we ought to attribute whether a point has been won or lost 
                    to the quality of the serve and the return. 
                    Theoretically, if the rally were to go on for almost ever, we shouldn't credit the return with winning (or losing) the rally, 
                    but only with neutralizing the serve and bringing the probability of winning the point to 50%.
                    This is exactly what we do with our attribution adjustment. 
                    If the point is over fast, the return is credited for winning or losing the point to a greater extent;
                    if the point ends only after a long rally, the return is only credited with neutralizing the serve.
                """
all_frames.append(tabFrms)
script_lines.append(script_line)

script_line =   """
                    After the attribution adjustment, chipped returns gain around 4% in win probability, whereas the topspin return's success rate is slightly reduced.

                    Why there's an increase for chips and none for topspin returns is a combination of many factors, 
                    the most intuitively accessible of which is as follows:
                """

tabFrm = smart_bal_len_table.render()
tabFrm = overlay_center(BGRFrame.make_black(), tabFrm)
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
all_frames.append(ovrFrms)
script_lines.append(script_line)


script_line =   """
                    For the aggressively minded tennis player, there are a variety of ways to end the point on the third shot,
                    either by serve and volley, or by baseline winner; these patterns can be subsumed under the term serve + 1.
                """

graphFrm = overlay_center(BGRFrame.make_black(), avg_graphFrm0)
graphFrms = BGRFrames.from_frame(graphFrm, STATIC_FRAMES)
all_frames.append(graphFrms)
script_lines.append(script_line)

script_line =   """
                    On a chipped return, the server's probability to make the (serve + 1) winner is "only" 62%. 
                    On aggregate, the server's patience would pay off: if he were to play a set-up shot instead of an immediate winner, 
                    let the returner scramble, and attempt a winner on the fifth shot of the rally, his success rate would rise to 82%.
                """

graphFrm = overlay_center(BGRFrame.make_black(), avg_graphFrm1)
graphFrms = BGRFrames.from_frame(graphFrm, STATIC_FRAMES)
all_frames.append(graphFrms)
script_lines.append(script_line)

script_line =   """
                    In other words, on a chipped return, a serve + 2 is more successful on average than a serve + 1. 
                    We will illuminate some of the causes for this later.
                """

graphFrm = overlay_center(BGRFrame.make_black(), avg_graphFrm2)
graphFrms = BGRFrames.from_frame(graphFrm, STATIC_FRAMES)
all_frames.append(graphFrms)
script_lines.append(script_line)

script_line =   """
                    The bad news for the server is that the more promising serve + 2 pattern is attempted only about half as often as the serve + 1. 
                    The typical server is, in a way, slightly too impatient on chipped returns.
                    Since the attribution adjustment correctly values the return's impact on the outcome of short rallyes higher than on long rallyes, 
                    the server's "impatience" is reflected in an increase in the success rate of the chip return after the attribution adjustment.
                """

graphFrm = overlay_center(BGRFrame.make_black(), shr_graphFrm)
graphFrms = BGRFrames.from_frame(graphFrm, STATIC_FRAMES)
all_frames.append(graphFrms)
script_lines.append(script_line)

script_line =   """
                    After all adjustments are made, chipped returns are still 12% worse than topspin returns in terms of winning percentage.
                    The question then is: how good must a chip return be to overcome those 12%? Where should it be placed; how should it be played?
                    Is it better to play a high-arcing chip or a low-bouncing chip? 
                    
                    In the following, we will attempt to answer these questions.
                """

tabFrm = smart_bal_len_table.render()
tabFrm = overlay_center(BGRFrame.make_black(), tabFrm)
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
all_frames.append(tabFrms)
script_lines.append(script_line)

### Saving the separate MultiFrame objects and the Joint Script
assert len(script_lines) == len(all_frames)

show(all_frames)

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
