import os 
import numpy as np
import pandas as pd
      
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames, MaskFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Visual3D import Camera, View, Scene

from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Shapes3D import Region3D, Line3D

from ATMANpy.types.Abstract import NullPoint, Points, Shots, read_in_data_tables, read_in_placeholder_tables, read_in_colinear_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.return_chip_topspin.utils as vis_ut

from ATMANpy.types.ShapesTennis3D import Arrow
from ATMANpy.types.Shapes3D import Circle3D

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'analyst_icon'
SAVE_FOLDER = 'website_snippets'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]


from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

#############################
#### READ IN CONFIGS ########
#############################
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import collect_video_frames

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
coliTable = read_in_colinear_tables(confs)

### replace points suspected of co-linearity with the co-linear stand-in
to_replace = [(0, 110, 1), (0, 112, 2)] ### match index, point index, shot index

replTable = dataTable.copy()

for mtc_idx, pnt_idx, sht_idx in to_replace:
    mtc_coliTable = coliTable.loc[mtc_idx]
    mtc_replTable = replTable.loc[mtc_idx]

    sht_coliTable = mtc_coliTable[(mtc_coliTable['point_idx'] == pnt_idx) & (mtc_coliTable['shot_idx'] == sht_idx)]
    mtc_replTable.loc[sht_coliTable.index] = sht_coliTable

dataPnts = Points.from_datatable(replTable, split_by_match = False)
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

##############################
#### EXPLANATION FOREHAND ####
##############################

from statsmodels.discrete.discrete_model import Logit

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

# -> why is there a preference for a return to the forehand side 
DECAY_RATE = 0.9
regFrm = vis_ut.make_regressor_frame(retPnts, logReg, winSer, idxFrm_new, decay_rate = DECAY_RATE)

in_col = 'return_is_fine' #'did_bounce_in'
retPnts_in = retPnts.filter(regFrm[in_col].values)
regFrm_in = regFrm[regFrm[in_col]]

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

### Weighted: What Percent Shots are Hit Forehand?
foreFrm = subFrm.groupby(['return_is_chipped', 'return_area']).apply(weighted_fore_percentage)
foreFrm = foreFrm.unstack('return_is_chipped')

### Win% if server runs around 
fore_srv_winFrm = subFrm.groupby(['return_area', 'third_fore']).apply(weighted_win_percentage_nochip, attrib = True)
fore_srv_winFrm = fore_srv_winFrm.unstack('third_fore')
fore_ret_winFrm = 1 - fore_srv_winFrm

#########################
#### MAKE ALL GRAPHS ####
#########################

BALL_COLOR = const.BGR['ball_yellow']
CHIP_LO_COLOR = const.BGR['marine_blue']
CHIP_HI_COLOR = const.BGR['bright_blue']
TOPS_COLOR = const.BGR['light_brown']

from ATMANpy.funcs.util_funcs import interpolate_tuples
CHIP_COLOR = interpolate_tuples(CHIP_LO_COLOR, CHIP_HI_COLOR)

from ATMANpy.auxil.court import birds_eye_half_court_cam
topCam = birds_eye_half_court_cam(side = 'Top', zoom = 1.5).up(10)

from ATMANpy.types.Overlays import Graph, LineChart, ScatterChart

# a) win probability on chip/tops
tops_rets = chip_ret_winFrm[False].values.tolist()
chip_rets = chip_ret_winFrm[True].values.tolist()

topsChart = LineChart([0.5, 1.5, 2.5, 3.5], tops_rets, color = TOPS_COLOR, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'above',
                        annotation_size = 2)
chipChart = LineChart([0.5, 1.5, 2.5, 3.5], chip_rets, color = CHIP_LO_COLOR, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'below',
                        annotation_size = 2)

myGraph1 = Graph([topsChart, chipChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'Ret. Win%', label_size = 2, tick_label_size = 2, 
                x_label_dist = 10, x_tick_label_left_shift = 12)
myGraph1.render().show()

# b) fore probability on chip/tops
tops_rets = foreFrm[False].values.tolist()
chip_rets = foreFrm[True].values.tolist()

topsChart = LineChart([0.5, 1.5, 2.5, 3.5], tops_rets, color = TOPS_COLOR, thickness = 0.5,
                      annotate_line_heights = True, annotate_line_heights_where = 'below',
                      annotation_size = 2)
chipChart = LineChart([0.5, 1.5, 2.5, 3.5], chip_rets, color = CHIP_LO_COLOR, thickness = 0.5,
                      annotate_line_heights = True, annotate_line_heights_where = 'above',
                      annotation_size = 2)

myGraph2 = Graph([topsChart, chipChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'S+1 Fore%', label_size = 2, tick_label_size = 2,
                x_label_dist = 10, x_tick_label_left_shift = 12)
myGraph2.render().show()

# c) win probability on fore/back
back_rets = (1 - fore_ret_winFrm[False].values).tolist()
fore_rets = (1 - fore_ret_winFrm[True].values).tolist()

# -> prolong the backhands a bit so they don't look so short on the x-axis
back_rets_extra_val = back_rets[2] - (back_rets[3] - back_rets[2]) / 2
back_rets_extra = [back_rets_extra_val, back_rets_extra_val, *back_rets[2:]]

back_color = const.BGR['minty_green']
fore_color = const.BGR['rose_orange']

backChart = LineChart([1.99, 2, 2.5, 3.5], back_rets_extra, color = back_color, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'below',
                        annotation_size = 2)
foreChart = LineChart([0.5, 1.5, 2.5, 3.5], fore_rets, color = fore_color, thickness = 0.5,
                        annotate_line_heights = True, annotate_line_heights_where = 'above',
                        annotation_size = 2)

myGraph3 = Graph([backChart, foreChart], cam = topCam, dimensions = (1920, 1080), make_default_x_labels = False,
                y_label_format = '>5.0%', y_label_dist = 14, offset = 0.38, header_text = "", header_height = 0, 
                y_label = 'Srv. Win%', label_size = 2, tick_label_size = 2,
                x_label_dist = 10, x_tick_label_left_shift = 12)
# myGraph3.render().show()

backFrm = BGRFrame.make_black() + const.BGR['darker_grey']
backFrm_leg = BGRFrame.make_black(rows = 200, cols = 400) + const.BGR['darker_grey']

##########################
#### MAKE THE LEGENDS ####
##########################

from ATMANpy.types.Overlays import Legend, LegendRow, overlay

# a) chip / tops
legRows = [LegendRow('... Topspin', TOPS_COLOR, text_size = 2),
           LegendRow('... Chip', CHIP_COLOR, text_size = 2)]
myLeg_chip_tops = Legend(legRows, header_text = 'Return is ...', header_offset = 2.7, header_thickness = 1,
                            legend_dimensions = (400, 200), resolution = 2000, borders = None)
legFrm_chip_tops = myLeg_chip_tops.render(over_array = backFrm_leg.array)
# legFrm_chip_tops.show()

# b) fore / back
legRows = [LegendRow('... Forehand', fore_color, text_size = 2),
           LegendRow('... Backhand', back_color, text_size = 2)]
myLeg_chip_tops = Legend(legRows, header_text = 'Serve + 1 is ...', header_offset = 2.7, header_thickness = 1,
                            legend_dimensions = (400, 200), resolution = 2000, borders = None)
legFrm_fore_back = myLeg_chip_tops.render(over_array = backFrm_leg.array)
# legFrm_fore_back.show()

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

##########################
### ANIMATE THE GRAPHS ###
##########################

courtScn = Scene.make_court()


from ATMANpy.types.Overlays import make_text_in

### cam
from ATMANpy.auxil.court import cnt_2D, top_cnt_2D
from ATMANpy.types.Shapes3D import Point3D
p_orig, p_target = Point3D.from_point2D(cnt_2D, z = 100), Point3D.from_point2D(top_cnt_2D, z = 20)
heatCam = Camera.from_to(p_orig, p_target, f = 1000, sensor_dims = (960, 540)).zoom(0.5)

from ATMANpy.types.Shapes3D import Region3D
from ATMANpy.auxil.court import top_a, top_b, top_c, top_d
top_a_3D, top_b_3D, top_c_3D, top_d_3D = Region3D.from_region2D(top_a, z = 0), Region3D.from_region2D(top_b, z = 0), Region3D.from_region2D(top_c, z = 0), Region3D.from_region2D(top_d, z = 0)

regLines3D = [myLn for myReg in [top_a_3D, top_b_3D, top_c_3D, top_d_3D] for myLn in myReg.lineList]
regLines3D = [myLn.recolor(const.BGR['leafy_green']).resize(0.5) for myLn in regLines3D]

abcd_params = {'color': const.BGR['leafy_green'], 'size': 2, 'horizontal_offset': 25}
aPt, bPt, cPt, dPt = make_text_in(top_a_3D, 'A', **abcd_params), make_text_in(top_b_3D, 'B', **abcd_params), make_text_in(top_c_3D, 'C', **abcd_params), make_text_in(top_d_3D, 'D', **abcd_params)

# annotated points
from ATMANpy.auxil.court import top_bsl_deu_2D, top_bsl_ad_2D
fhPt_annot = Point3D.from_point2D(top_bsl_deu_2D, z = 0).front(16).up(10).right(5).annotate("Server's\nForehand")
bhPt_annot = Point3D.from_point2D(top_bsl_ad_2D, z = 0).front(16).up(10).left(35).annotate("Server's\nBackhand")

fhPt_annot_fore = fhPt_annot.resize(2).right(16)
bhPt_annot_fore = bhPt_annot.resize(2).left(16)

annotated_points = [fhPt_annot_fore, bhPt_annot_fore, aPt, bPt, cPt, dPt]

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
graphScn1 = Scene(jntLns)
jntPts1 = [*annotated_points, *graph_annPnts]

graphScn, graph_annPnts = myGraph2.preview()
jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
graph_annPnts[-5] = graph_annPnts[-5].back(18)
graph_annPnts[-4] = graph_annPnts[-4].back(18)
graph_annPnts[-3] = graph_annPnts[-3].back(5)
graph_annPnts[2] = graph_annPnts[2].front(12)
graph_annPnts[3] = graph_annPnts[3].front(12)
graph_annPnts[4] = graph_annPnts[4].front(-4)
graphScn2 = Scene(jntLns)
jntPts2 = [*annotated_points, *graph_annPnts]

# View.make(topCam, graphScn, annotatedPoints = graph_annPnts).snap().show()

graphScn, graph_annPnts = myGraph3.preview()
graph_annPnts.pop(2)
graph_annPnts.pop(2)
jntLns = [*regLines3D_trunc, *graphScn.shapes[2:]]
graphScn3 = Scene(jntLns)
jntPts3 = [*annotated_points, *graph_annPnts]

### procedure: 
# do the fading transitions with just the annotated points first, 
# and then animate the lines over it


motionFrms_into_virtual, _ = vs.animate_sequence([regLines3D for _ in range(25)], start_cam = heatCam.resize(2), stop_cam = topCam, **prmDict, 
                                                    background_scene = courtScn, interpolate_func = vs.interpolate_camera_naive,
                                                    annotated_points = annotated_points,
                                                    animate_over = backFrm)

motionFrms_into_graphed, _ = vs.animate_sequence([graphScn1 for _ in range(25)], start_cam = heatCam.resize(2), stop_cam = topCam, **prmDict, 
                                                    background_scene = None, interpolate_func = vs.interpolate_camera_naive,
                                                    annotated_points = jntPts1,
                                                    animate_over = backFrm)
motionFrms_into_graphed = overlay(motionFrms_into_graphed, legFrm_chip_tops, top_left_point = tlPt)

motionFrms_into = motionFrms_into_graphed.dim_progressively(0, 1) + motionFrms_into_virtual.dim_progressively(1, 0)


DecapStream(motionFrms_into, 0, 100).show()

staticFrms, _= vs.animate_static(regLines3D, static_cam = topCam, **prmDict, 
                                      background_scene = courtScn,
                                      annotated_points = annotated_points, num_frames = 25,
                                      animate_over = backFrm)


# a) fading transitions with just point
staticFrms1, _ = vs.animate_static(Scene.make_empty(), static_cam = topCam, **prmDict, 
                                    background_scene = courtScn_subset,
                                    annotated_points = jntPts1, num_frames = 45,
                                    animate_over = backFrm)
staticFrms1 = overlay(staticFrms1, legFrm_chip_tops, top_left_point = tlPt)

staticFrms2, _ = vs.animate_static(Scene.make_empty(), static_cam = topCam, **prmDict, 
                                    background_scene = courtScn_subset,
                                    annotated_points = jntPts2, num_frames = 45,
                                    animate_over = backFrm)
staticFrms2 = overlay(staticFrms2, legFrm_chip_tops, top_left_point = tlPt)

staticFrms3, _ = vs.animate_static(Scene.make_empty(), static_cam = topCam, **prmDict, 
                                    background_scene = courtScn_subset,
                                    annotated_points = jntPts3, num_frames = 45,
                                    animate_over = backFrm)
staticFrms3 = overlay(staticFrms3, legFrm_fore_back, top_left_point = tlPt)

itplFrms1 = vs.interpolate_frames(staticFrms1[-1], staticFrms2[0], 20)
itplFrms2 = vs.interpolate_frames(staticFrms2[-1], staticFrms3[0], 20)

back_frames = [staticFrms1, itplFrms1, staticFrms2, itplFrms2, staticFrms3]
backFrms = BGRFrames.from_frames(back_frames)

vis_ut.show(back_frames)

# b) animate lines over it
itplScns1 = vs.interpolate_scenes(graphScn1, graphScn2, 20)
itplScns2 = vs.interpolate_scenes(graphScn2, graphScn3, 20)

animScns = [*[graphScn1 for _ in range(45)], 
            *itplScns1, 
            *[graphScn2 for _ in range(45)],
            *itplScns2,
            *[graphScn3 for _ in range(45)]]

animFrms, _ = vs.animate_sequence(animScns, start_cam = topCam, stop_cam = topCam, **prmDict,
                                  animate_over = backFrms)

DecapStream(animFrms, 0, 100).show()

### transition back to initial scene

motionFrms_back_virtual, _ = vs.animate_sequence([regLines3D for _ in range(25)], start_cam = topCam, stop_cam = heatCam.resize(2), **prmDict, 
                                        background_scene = courtScn, interpolate_func = vs.interpolate_camera_naive,
                                        annotated_points = annotated_points, 
                                        animate_over = backFrm)

motionFrms_back_graphed, _ = vs.animate_sequence([graphScn3 for _ in range(25)], start_cam = topCam, stop_cam = heatCam.resize(2), **prmDict, 
                                        background_scene = None, interpolate_func = vs.interpolate_camera_naive,
                                        annotated_points = jntPts3, 
                                        animate_over = backFrm)
motionFrms_back_graphed = overlay(motionFrms_back_graphed, legFrm_fore_back, top_left_point = tlPt)


motionFrms_back = motionFrms_back_virtual.dim_progressively(0, 1) + motionFrms_back_graphed.dim_progressively(1, 0)

DecapStream(motionFrms_back, 0, 100).show()

motionFrms_hold = BGRFrames.from_frame(motionFrms_back[-1], 25)

allFrms = BGRFrames.from_frames([motionFrms_into, animFrms, motionFrms_back, motionFrms_hold])
DecapStream(allFrms, 0, 100).show()

allFrms = allFrms.gaussian_noise(std = 10).gaussian_blur((3, 3))

allFrms = allFrms.rescale(1/3)

DecapStream(allFrms, 0, 100).show()

allFrms.save(fullPath, f'analyst', lossless = True, save_as = 'mp4')

testFrms = BGRFrames.from_frames([allFrms, allFrms, allFrms, allFrms])

DecapStream(testFrms, 0, 100).show()




