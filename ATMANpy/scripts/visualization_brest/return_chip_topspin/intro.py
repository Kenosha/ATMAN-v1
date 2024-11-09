
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

import ATMANpy.auxil.constants as const

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Point, NullPoint, read_in_data_tables, read_in_all_placeholder_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.Overlays import overlay
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
nullTable = read_in_all_placeholder_tables(confs)

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
                'header_text': 'Serve Success and Frequency on Choice Chipped Returns', 'header_size': 2, 'header_horizontal_offset': 5,
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
avg_graphFrm0.show()

myGraph.add(vChart1)
avg_graphFrm1 = myGraph.render()
avg_graphFrm1 = avg_graphFrm1.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
avg_graphFrm1 = overlay(avg_graphFrm1, legFrm1, top_left_point = tlPt)
avg_graphFrm1.show()

myGraph.add(vChart2)
avg_graphFrm2 = myGraph.render()
avg_graphFrm2 = avg_graphFrm2.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
avg_graphFrm2 = avg_graphFrm2.put('Serve+2', color = const.BGR['intense_red'], where = (716, 425), size = 1)
avg_graphFrm2 = overlay(avg_graphFrm2, legFrm1, top_left_point = tlPt)
avg_graphFrm2.show()

myGraph.add(shrChart)
shr_graphFrm = myGraph.render()
shr_graphFrm = shr_graphFrm.put('Serve+1', color = const.BGR['intense_red'], where = (460, 425), size = 1)
shr_graphFrm = shr_graphFrm.put('Serve+2', color = const.BGR['intense_red'], where = (716, 425), size = 1)
shr_graphFrm = overlay(shr_graphFrm, legFrm2, top_left_point = tlPt)
shr_graphFrm.show()

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
ex1Frm.show()

player_a = {'Serve': 'Average', 'Return': 'Great'}
player_b = {'Serve': 'Average', 'Return': 'Average'}
table_data = {'Player A': player_a, 'Player B': player_b}

ex2_table = Table(table_data, title = 'Example 2', **table_kwargs_examples)
ex2Frm = ex2_table.render()
ex2Frm.show()

myStr = '-> Player A plays more points on return!'

ex1Frm_with_text = ex1Frm.put(myStr, where = (118, 500), size = 2, color = const.BGR['bright_grey'])
ex1Frm_with_text.show()

ex2Frm_with_text = ex2Frm.put(myStr, where = (118, 500), size = 2, color = const.BGR['bright_grey'])
ex2Frm_with_text.show()


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
barGraph.render().show()

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

View.make(myGraph.cam, bar_graphScn_null, annotatedPoints = bar_annotatedPnts_start, resolution = BAR_RESOLUTION).snap().show()
View.make(myGraph.cam, bar_graphScn, annotatedPoints = bar_annotatedPnts, resolution = BAR_RESOLUTION).snap().show()


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

tdict = {'Unadjusted': t0}
cdict = {'Unadjusted': c0}
table_data = {index_name_t: tdict, index_name_c: cdict}

initial_naive_table = Table(table_data, **table_kwargs_small, title = title, header_text_size = 0)
# initial_naive_table.render().show()

naive_table = Table(table_data, **table_kwargs_large, title = title)
naive_table.render().show()

tdict = {'Unadjusted': t0, '+Choice Adj.': t1}
cdict = {'Unadjusted': c0, '+Choice Adj.': c1}
table_data = {index_name_t: tdict, index_name_c: cdict}

smart_table = Table(table_data, **table_kwargs_large, title = title)
smart_table.render().show()

tdict = {'Unadjusted': t0, '+Choice Adj.': t1, '+Balance Adj.': t2}
cdict = {'Unadjusted': c0, '+Choice Adj.': c1, '+Balance Adj.': c2}
table_data = {index_name_t: tdict, index_name_c: cdict}

smart_bal_table = Table(table_data, **table_kwargs_large, title = title)
smart_bal_table.render().show()

tdict = {'Unadjusted': t0, '+Choice Adj.': t1, '+Balance Adj.': t2, '+Attrib. Adj.': t3}
cdict = {'Unadjusted': c0, '+Choice Adj.': c1, '+Balance Adj.': c2, '+Attrib. Adj.': c3}
table_data = {index_name_t: tdict, index_name_c: cdict}

smart_bal_len_table = Table(table_data, **table_kwargs_large, title = title)
smart_bal_len_table.render().show()

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

str_chips = f'{num_chip:>4} ({num_chip/num_tot:.0%})'
str_tops = f'{num_tops:>4} ({num_tops/num_tot:.0%})'

tdict = {'Sample Size': str_tops}
cdict = {'Sample Size': str_chips}
totDict = {'Sample Size': str(num_tot)}
table_data = {index_name_t: tdict, index_name_c: cdict, index_name_a: totDict}

sample_table = Table(table_data, **table_kwargs_small, null_columns = 1)
sample_table.render().show()

tdict = {'Sample Size': str_tops, 'Returner Win%': t0}
cdict = {'Sample Size': str_chips, 'Returner Win%': c0}
totDict = {'Sample Size': str(num_tot), 'Returner Win%': (num_chip * c0 + num_tops * t0) / num_tot}
table_data = {index_name_t: tdict, index_name_c: cdict, index_name_a: totDict}

sample_and_win_table = Table(table_data, **table_kwargs_small)
sample_and_win_table.render().show()

###############################################
###### GET HIGH AND LOW PROPENSITY CHIPS ######
###############################################
dirFrm = regFrm[(regFrm['is_top_to_bot'])]

high_pred_exFrm = dirFrm.sort_values('pred_is_chipped').iloc[-8:]
high_pred_exFrms = [retPnts[idx][:2].stream(confs = confs).decap().frames for idx in high_pred_exFrm.index]

low_pred_exFrm = dirFrm[dirFrm.is_chipped].sort_values('pred_is_chipped').iloc[:8]
low_pred_exFrms = [retPnts[idx][:2].stream(confs = confs).decap().frames for idx in low_pred_exFrm.index]


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

script_line =   """
                    At the most basic level, there are two ways to return a first-serve. 
                    A player can either adjust his grip to his or her preferred backhand or forehand grip, 
                    and attempt to return the ball with topspin,
                    or he or she can keep a neutral, continental grip, and chip the ball back into play.
                    The resulting chip will have underspin, but, as we will see in the course of this analysis, 
                    it will have typically less RPM, greater net clearance, and lower speed than a slice employed in a rally.
                """

titleStr = 'On the Effectiveness of the Chipped Return'
titleFrm = blankFrm.put(titleStr, size = 2, where = (600, 540))

titleFrms = BGRFrames.from_frame(titleFrm, STATIC_FRAMES)
all_frames.append(titleFrms)
script_lines.append(script_line)

script_line =   """
                    The chipped return is infrequently employed by choice; 
                    in our sample of 1,451 first-serve returns, only 28% of first-serves are returned as a chip, 
                    and even fewer, as we will see, by choice.
                    However, some highly successful players consistently use it as a tool 
                    to either neutralize the rally with a deep, floating chip, or draw the server into the court with a shallow, low-bouncing chip.

                    The primary goal of this analysis is to take a good look at these chip return varieties, 
                    and try to understand their effectiveness compared to the topspin return, as measured by the expected probability to win the point.
                """

baseRst = FrameRaster(3, 3, borders = None)
baseRst.add(sample_table.render(), 1, 1)
baseFrm = baseRst.render()
# baseFrm.show()
baseFrms = BGRFrames.from_frame(baseFrm, repetitions = STATIC_FRAMES)
all_frames.append(baseFrms)
script_lines.append(script_line)

script_line =   """
                    At first glance, it doesn't look good for the chipped return. 
                    On average, in our sample, the returner wins around 40% of points whenever he returns topspin, 
                    versus only around 20% of points when he chips the ball back.

                    However, this view is slightly unfair toward the chipped return. 
                    To understand why, carefully consider these eight returns, and what they have in common.
                """
baseRst.add(sample_and_win_table.render(), 1, 1)
baseFrm = baseRst.render()
# baseFrm.show()
baseFrms = BGRFrames.from_frame(baseFrm, repetitions = STATIC_FRAMES)
all_frames.append(baseFrms)
script_lines.append(script_line)

script_line = ""
baseRst = FramesRaster(3, 3, borders = None)
baseRst.add(sample_and_win_table.render(), 1, 1)
finalRst, cascFrms = frame_cascade(high_pred_exFrms, baseRst)
# DecapStream(cascFrms, 0, 100).show()
# cascFrms[-1].show()
all_frames.append(cascFrms)
script_lines.append(script_line)

script_line =   """
                    All these returns have three things in common. They're played off the backhand, on the stretch, as a chip!
                    In fact, there is virtually no way to return these with topspin; 
                    and not just because a great majority of players play a two-handed topspin backhand, and the serve can only be reached with one hand.
                    On these serves, the returner has no choice: he is forced to block the ball back in, 
                    usually resulting in a short chip that is easy to attack if it lands fair at all.
                    
                    On these returns, then, there is nothing to discuss, for the lack of choice. 
                    The serve is almost an ace - do what you can, chip it back in, and pray for the best.
                    
                    What can be discussed are returns that are chipped by choice: given that the returner could either play a topspin or a chip return, 
                    we want to assess the quality of said choice. 
                    We want to find out how one should chip, or how one should play topspin, to maximize one's chances of winning the point. 
                    
                    Therefore, the ability to make a choice matters, and we shall have to put in some work to find out on which serves the returner actually had a choice.
                """
staticFrms = BGRFrames.from_frame(cascFrms[-1], STATIC_FRAMES)
all_frames.append(staticFrms)
script_lines.append(script_line)

script_line = ""
revRst, revFrms = reverse_cascade(finalRst, keep_which = [(1, 1)], acceleration = 1.2, frames_per_element = 15)
# DecapStream(revFrms, 0, 100).show()
# revFrms[-1].show()

all_frames.append(revFrms)
script_lines.append(script_line)

#################################
##### PROPENSITY SCORE MODEL ####
#################################
 
RESOLUTION = 10
renameDict = {  'serve_speed': 'Serve Speed', 'serve_depth': 'Return Depth', 'serve_height': 'Bounce Height', 
                'ret_is_back': 'Returner uses Backhand', 'serve_dist_sq': 'Stretch Distance'}
fitted_model = logReg
max_line_thickness = 2.5
thickness_taper = 0.05
shape_offset = 0.1
model_name = 'Propensity to Chip'


outDiag = vis_ut.model_to_diagram(fitted_model, renameDict = renameDict, model_name = 'Propensity to Chip', 
                                    max_line_thickness = 3, resolution = RESOLUTION, shape_offset = shape_offset, 
                                    shape_color = const.BGR['marine_blue'], text_color = const.BGR['bright_grey'])

for col in range(5):
    outDiag[0, col][0].text_vertical_offset = -8

outDiag[0, 0][0].text_horizontal_offset = 6
outDiag[0, 1][0].text_horizontal_offset = 4
outDiag[0, 2][0].text_horizontal_offset = 9
outDiag[0, 3][0].text_horizontal_offset = 9
outDiag[0, 4][0].text_horizontal_offset = 10
# outDiag[0, 5][0].text_horizontal_offset = 4
# outDiag[0, 6][0].text_horizontal_offset = 5

outDiag[1, 0][0].text_skip_after_how_many_symbols = None
outDiag[1, 0][0].text_horizontal_offset = 40
outDiag[1, 0][0].text_vertical_offset = 0
outDiag[1, 0][0].text_size = 2
outDiag[1, 0][0].text_thickness = 2

### staggered diagram
def remove_all_diagram_shapes_after_index(myDiag, idx):
    myDiag.shapes[0] = [shape for i, shape in enumerate(myDiag.shapes[0]) if i < idx]
    myDiag.arrows[0] = [arrow for i, arrow in enumerate(myDiag.arrows[0]) if i < idx]
    return myDiag

outDiags = [remove_all_diagram_shapes_after_index(outDiag.deepcopy(), idx) for idx in range(6)]

for myDiag in outDiags:
    diagFrm = myDiag.render()
    diagFrm.show()

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
blankFrms = BGRFrames.from_frame(blankFrm, STATIC_FRAMES)
all_frames.append(blankFrms)
script_lines.append(script_line)

diagRaster = FrameRaster(3, 3, borders = None)

script_line = """
                PSM estimates, from the data at hand, the propensity to chip a ball, given a set of explanatory variables. 
                In our case, these explanatory variables pertain to the characteristics of the serve relative to the returner.
                How this Propensity Model is useful for our purposes will become apparent in a moment; 
                For now, imagine the PSM as simply a device that looks at the characteristics of a serve and predicts the probability 
                that the return will be chipped back, as opposed to returned with topspin.
            """
diagFrm = outDiags[0].render()
diagRaster.add(diagFrm, 1, 1)
diagFrm0 = diagRaster.render()
# diagFrm0.show()
diagFrms0 = BGRFrames.from_frame(diagFrm0, STATIC_FRAMES)
all_frames.append(diagFrms0)
script_lines.append(script_line)


script_line =   """
                    The best predictor of whether a ball is chipped back is how far the returner has to stretch to get to it. 
                    The green arrow indicates that the relationship is positive: 
                    The farther the stretch, the likelier the return is to be chipped back.
                """
diagFrm = outDiags[1].render()
diagRaster.add(diagFrm, 1, 1)
diagFrm1 = diagRaster.render()
# diagFrm1.show()
diagFrms1 = BGRFrames.from_frame(diagFrm1, STATIC_FRAMES)
all_frames.append(diagFrms1)
script_lines.append(script_line)


script_line =   """
                    The second most important predictor of whether the ball will be chipped back is whether the ball goes to the returner's backhand or forehand.
                    If the serve goes to the backhand, the return is likelier to be a chip, as opposed to a serve to the returner's forehand, which is more often returned with topspin.
                    Note that these two predictors are exactly what the eight returns we showed just now had most obviously in common. 
                    Beyond these two, however, there are other predictors. In diminishing order of importance, these are...
                """
diagFrm = outDiags[2].render()
diagRaster.add(diagFrm, 1, 1)
diagFrm2 = diagRaster.render()
# diagFrm2.show()
diagFrms2 = BGRFrames.from_frame(diagFrm2, STATIC_FRAMES)
all_frames.append(diagFrms2)
script_lines.append(script_line)

script_line =   """
                    A serve that is returned from a deeper position is LESS likely to be a chip; 
                    the red arrow indicates that the relationship is inverse. 
                    From a deep return position, the returner has more time to set up for a topspin return.
                """
diagFrm = outDiags[3].render()
diagRaster.add(diagFrm, 1, 1)
diagFrm3 = diagRaster.render()
# diagFrm3.show()
diagFrms3 = BGRFrames.from_frame(diagFrm3, STATIC_FRAMES)
all_frames.append(diagFrms3)
script_lines.append(script_line)

script_line =   """
                    A low-bouncing serve is more likely to be chipped back than a high-bouncing serve...
                """
diagFrm = outDiags[4].render()
diagRaster.add(diagFrm, 1, 1)
diagFrm4 = diagRaster.render()
# diagFrm4.show()
diagFrms4 = BGRFrames.from_frame(diagFrm4, STATIC_FRAMES)
all_frames.append(diagFrms4)
script_lines.append(script_line)

script_line =   """
                    And, finally, a fast serve is more likely to be chipped back than a slow serve.
                """
diagFrm = outDiags[5].render()
diagRaster.add(diagFrm, 1, 1)
diagFrm5 = diagRaster.render()
# diagFrm5.show()
diagFrms5 = BGRFrames.from_frame(diagFrm5, STATIC_FRAMES)
all_frames.append(diagFrms5)
script_lines.append(script_line)

script_line =   """
                    So how is this propensity model useful in differentiating between chips that are forced and chips that are done by choice?
                """
all_frames.append(diagFrms5)
script_lines.append(script_line)

num_frames = 5
reversed_frame_list = [ diagFrms5[:num_frames], diagFrms4[:num_frames], 
                        diagFrms3[:num_frames], diagFrms2[:num_frames], 
                        diagFrms1[:num_frames], diagFrms0[:num_frames]]

for myFrms in reversed_frame_list:
    all_frames.append(myFrms)
    script_lines.append("")

script_line = ""
revFrms = BGRFrames.from_frames(reversed_frame_list)

script_line = "Knowing that the propensity score represents the estimated probability of chipping the ball back..."
stillFrms = BGRFrames.from_frame(revFrms[-1], STATIC_FRAMES)
all_frames.append(stillFrms)
script_lines.append(script_line)

##############################
### NUMBER LINE PROPENSITY ###
##############################e
from ATMANpy.types.Visual3D import Scene, View
from ATMANpy.types.Shapes3D import Region3D
import ATMANpy.funcs.vis_funcs as vs

myReg, annPnt = outDiag[1, 0][0].render()
annotated_points = [annPnt]
relative_thickness = 0.1
resolution = 10
num_frames = 25

transFrms, arrScn, arrCam, ptDict = vis_ut.region_to_line_to_arrow(myReg, annPnt, outDiag, relative_line_thickness = 0.02, relative_line_length = 0.9,
                                                                    resolution = RESOLUTION, num_frames = 25, ann_right_px = -45, ann_left_px = 0, ann_vert_px = 15)

script_line = ""
myRst = FramesRaster(3, 3, borders = None)
myRst.add(transFrms, 1, 1)
transFrms_rast = myRst.render()
all_frames.append(transFrms_rast)
script_lines.append(script_line)

script_line = " ... we calculate it for any given return on which a chip was in fact observed."
arrFrm0 = View.make(arrCam, arrScn, annotatedPoints = [ptDict['m_text'], ptDict['l_num'], ptDict['r_num']]).snap()
arrFrms0 = BGRFrames.from_frame(arrFrm0, STATIC_FRAMES)
myRst.add(arrFrms0, 1, 1)
statFrms_rast = myRst.render()
all_frames.append(statFrms_rast)
script_lines.append(script_line)

script_line =   """
                    If the PSM predicts a high propensity to chip, 
                    then it effectively believes that there was no way that specific serve could have been returned with topspin.
                    It is a forced chip return.
                """

arrFrm0_noright = View.make(arrCam, arrScn, annotatedPoints = [ptDict['m_text'], ptDict['l_num']]).snap()
arrFrms_blink0 = vs.blinking(arrFrm0, arrFrm0_noright, blinks_per_second = 2, duration_in_frames = 300)

myRst = FramesRaster(3, 3, borders = None)
myRst.add(arrFrms_blink0, 1, 1)
blinkFrms_rast = myRst.render()
all_frames.append(blinkFrms_rast)
script_lines.append(script_line)

script_line =   """
                    In fact, the eight returns we just saw are the eight highest propensity returns in our sample.    
                """
arrFrm1 = View.make(arrCam, arrScn, annotatedPoints = [ptDict['m_text'], ptDict['l_num'], ptDict['r_num'], ptDict['r_text']]).snap()
arrFrms1 = BGRFrames.from_frame(arrFrm1, STATIC_FRAMES)
myRst.add(arrFrms1, 1, 1)
statFrms_rast = myRst.render()
all_frames.append(statFrms_rast)
script_lines.append(script_line)


script_line =   """
                    On the other hand, on a low propensity chipped return, the model believes that playing a chip in such a situation is very unlikely;
                    in most cases, the returner would have opted for a topspin return.
                    Therefore, observing a chipped return in such a low propensity situation expresses the returner's strong choice to chip the ball back.
                """
arrFrm1_noleft = View.make(arrCam, arrScn, annotatedPoints = [ptDict['m_text'], ptDict['r_num'], ptDict['r_text']]).snap()
arrFrms_blink1 = vs.blinking(arrFrm1, arrFrm1_noleft, blinks_per_second = 2, duration_in_frames = 300)

myRst = FramesRaster(3, 3, borders = None)
myRst.add(arrFrms_blink1, 1, 1)
blinkFrms_rast = myRst.render()
all_frames.append(blinkFrms_rast)
script_lines.append(script_line)

script_line =   """
                    To illuminate this, consider the 8 lowest propensity chipped returns.
                """
arrFrm2 = View.make(arrCam, arrScn, annotatedPoints = [ptDict['m_text'], ptDict['l_num'], ptDict['r_num'], ptDict['l_text'], ptDict['r_text']]).snap()
arrFrms2 = BGRFrames.from_frame(arrFrm2, STATIC_FRAMES)
myRst.add(arrFrms2, 1, 1)
statFrms_rast = myRst.render()
all_frames.append(statFrms_rast)
script_lines.append(script_line)



jntFrms = BGRFrames.from_frames(all_frames)
# DecapStream(jntFrms, 0, 100).show()

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


stillFrm = arrFrms2[-1]
propRst = baseRst.deepcopy()
propRst.add(stillFrm, 1, 1)

finalRst, cascFrms = frame_cascade(low_pred_exFrms, propRst)

script_line = ""
all_frames.append(cascFrms)
script_lines.append(script_line)

script_line =   """
                On these returns, it does indeed seem that the returning player CHOSE to chip the ball back, 
                where he could very well have played a topspin return.

                We can now use these propensities to focus on chipped returns where the returner actually had a choice, 
                and thereby assess the quality of this choice.
                """
cascFrms = BGRFrames.from_frame(cascFrms[-1], STATIC_FRAMES)
all_frames.append(cascFrms)
script_lines.append(script_line)


# (show table with choice returns)

script_line = ""
revRst, revFrms = reverse_cascade(finalRst, keep_which = [(1, 1)], acceleration = 1.2, frames_per_element = 15)
all_frames.append(revFrms)
script_lines.append(script_line)

script_line = ""
emptyFrm = revFrms[-1].zero()
emptyFrms = BGRFrames.from_frame(emptyFrm, STATIC_FRAMES)
all_frames.append(emptyFrms)
script_lines.append(script_line)

# jntFrms = BGRFrames.from_frames(all_frames)
# DecapStream(jntFrms, 0, 100).show()
blankFrms_short = BGRFrames.from_frame(blankFrm, STATIC_FRAMES)
blankFrms_long = BGRFrames.from_frame(blankFrm, TRANSITION_FRAMES)

script_line =   """
                    As we saw earlier, the probability to win a point following a chipped return is just under 20%. 
                    If we concentrate on choice chips, we expect that probability to increase, since we discount
                    those serves that are returned on the stretch, under great difficulty.
                """

tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, tabFrms, top_left_point = table_offsetPt)

all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    When concentrating on choice returns, the probability does indeed increase; 
                    however, it only increases modestly, to 23.3%.  
                """
tabFrm = smart_table.render()
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, tabFrms, top_left_point = table_offsetPt)

all_frames.append(ovrFrms)
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
all_frames.append(ovrFrms)
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

ovrFrms = overlay(blankFrms_long, myFrms, top_left_point = graph_offsetPt)

script_line = ""
all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =  """
                    We observe that the return player has greater win probability in long rallys, 
                    indicating that the better player indeed plays more returns in a match, simply due to the nature of tennis scoring.

                    To remove this bias, and approximate the experimental setting, 
                    we introduce a balancing factor that gives more weight to returns made by the player for whom we observe fewer returns, overall. 
                    Ostensibly, this is the weaker player.
                """

myFrms, _ = vs.animate_static(graphScns_itpl[-1], static_cam = myGraph.cam, annotated_points = bar_annotatedPnts, 
                                argument_resolution = BAR_RESOLUTION, num_frames = STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, myFrms, top_left_point = graph_offsetPt)

all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    The balance adjustment does not greatly impact either win percentage beyond a slight reduction. This is to be expected, since we're
                    counteracting the impact of the better player being on return more often solely by virtue of the data generating process.
                """

tabFrm = smart_bal_table.render()
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, tabFrms, top_left_point = table_offsetPt)
all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    Another adjustment we have to undertake (and I promise this is the last one) is connected to something we have just mentioned. 
                    The longer a rally goes on, the less the serve and the return matter to the outcome of the point.
                    Therefore, the longer a rally goes on, the less we ought to attribute whether a point has been won or lost 
                    to the quality of the serve and the return. 
                    Theoretically, if the rally were to go on for ever, we shouldn't credit the return with winning (or losing) the rally, 
                    but only with neutralizing the serve and bringing the probability of winning the point to 50%.
                    This is exactly what we do with our attribution adjustment. 
                    If the point is over fast, the return is credited for winning or losing the point to a greater extent;
                    if the point ends only after a long rally, the return is only credited with neutralizing the serve.
                """
all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    After the attribution adjustment, chipped returns gain around 4% in win probability, whereas the topspin return's success rate is slightly reduced.

                    Why there's an increase for chips and none for topspin returns is somewhat unclear to me, but it seems to be a combination of many factors, 
                    the most intuitively accessible of which is as follows:
                """

tabFrm = smart_bal_len_table.render()
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, tabFrms, top_left_point = table_offsetPt)
all_frames.append(ovrFrms)
script_lines.append(script_line)


script_line =   """
                    For the aggressively minded tennis player, there are a variety of ways to end the point on the third shot,
                    either by serve and volley, or by baseline winner; these patterns can be subsumed under the term serve + 1.
                """

graphFrms = BGRFrames.from_frame(avg_graphFrm0, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, graphFrms, top_left_point = graph_offsetPt)
all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    On a chipped return, the server's probability to make the (serve + 1) winner is "only" 62%. 
                    On aggregate, the server's patience would pay off: if he were to play a set-up shot instead of an immediate winner, 
                    let the returner scramble, and attempt a winner on the fifth shot of the rally, his success rate would rise to 82%.
                """

graphFrms = BGRFrames.from_frame(avg_graphFrm1, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, graphFrms, top_left_point = graph_offsetPt)
all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    In other words, on a chipped return, a serve + 2 is more successful on average than a serve + 1. 
                    We will illuminate some of the causes for this later.
                """

graphFrms = BGRFrames.from_frame(avg_graphFrm2, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, graphFrms, top_left_point = graph_offsetPt)
all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    The bad news for the server is that the more promising serve + 2 pattern is attempted only about half as often as the serve + 1. 
                    The typical server is, in a way, slightly too impatient on chipped returns.
                    Since the attribution adjustment correctly values the return's impact on the outcome of short rallyes higher than on long rallyes, 
                    the server's "impatience" is reflected in an increase in the success rate of the chip return after the attribution adjustment.
                """

graphFrms = BGRFrames.from_frame(shr_graphFrm, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, graphFrms, top_left_point = graph_offsetPt)
all_frames.append(ovrFrms)
script_lines.append(script_line)

script_line =   """
                    So then, alright: This is our bottom line:
                    After all adjustments are made, chipped returns are still 12% worse than topspin returns in terms of winning percentage.
                    The question then is: how good must a chip return be to overcome those 12%? Where should it be placed; how should it be played?
                    Is it better to play a high-arcing chip or a low-bouncing chip? 
                    
                    In the following, we will attempt to answer these questions.
                """

tabFrm = smart_bal_len_table.render()
tabFrms = BGRFrames.from_frame(tabFrm, STATIC_FRAMES)
ovrFrms = overlay(blankFrms_short, tabFrms, top_left_point = table_offsetPt)
all_frames.append(ovrFrms)
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
