
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

regFrm = vis_ut.make_regressor_frame(retPnts, logReg, winSer, idxFrm_new)

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

columns_rename_dict = {'total': 'Returner Wins the Point', 'make': 'Return is Made', 'cond': 'Returner Wins the Point if Return is Made'}
columns_rename_dict_signs = {'total': 'Returner Wins =', 'make': 'Return is Made x', 'cond': 'Returner Wins if Return is Made'}

chips_naive_tot = {k: v for k, v in chips_naive.items() if k == 'total'}
tops_naive_tot = {k: v for k, v in tops_naive.items() if k == 'total'}
chips_smart_tot = {k: v for k, v in chips_smart.items() if k == 'total'}
tops_smart_tot = {k: v for k, v in tops_smart.items() if k == 'total'}

chips_smart_make = {k: v for k, v in chips_smart.items() if k != 'cond'}
tops_smart_make = {k: v for k, v in tops_smart.items() if k != 'cond'}

table_kwargs = {'number_format': '.1%', 'index_offset': 0.25, 'index_col_width': 1.25, 'header_row_height': 0.5,
                'skip_after_how_many_symbols': 13, 'dimensions': DIMENSIONS, 'columns_rename_dict': columns_rename_dict}

table_kwargs_nonull = {'number_format': '.1%', 'index_offset': 2.5, 'index_col_width': 1, 'header_row_height': 0.5,
                'skip_after_how_many_symbols': 13, 'dimensions': DIMENSIONS, 'columns_rename_dict': columns_rename_dict}

### First, simply compare total win percentage
table_data = {'Topspin Return': tops_naive_tot, 'Chipped Return': chips_naive_tot}
basic_naive_table_nonull = Table(table_data, **table_kwargs_nonull, null_columns = 0)
basic_naive_table_nonull.render().show()

table_data = {'Topspin Return': tops_naive_tot, 'Chipped Return': chips_naive_tot}
basic_naive_table = Table(table_data, **table_kwargs, null_columns = 2)
basic_naive_table.render().show()

table_data = {'Topspin Return': tops_smart_tot, 'Chipped Return (Choice)': chips_smart_tot}
basic_smart_table = Table(table_data, **table_kwargs, null_columns = 2)
basic_smart_table.render().show()

### Now, add in make data
table_data = {'Topspin Return': tops_smart_make, 'Chipped Return (Choice)': chips_smart_make}
make_smart_table = Table(table_data, **table_kwargs, null_columns = 1)
make_smart_table.render().show()

### finally, add in cond data
table_data = {'Topspin Return': tops_smart, 'Chipped Return (Choice)': chips_smart}
cond_smart_table =  Table(table_data, **table_kwargs, null_columns = 0)
cond_smart_table.render().show()

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

from ATMANpy.types.Overlays import FramesRaster, frame_cascade, reverse_cascade

### basic raster
baseRst = FramesRaster(3, 3)
baseRst.add(basic_naive_table_nonull.render(), 1, 1)
baseRst.render()[0].show()

finalRst, cascFrms = frame_cascade(high_pred_exFrms, baseRst)
DecapStream(cascFrms, 0, 100).show()
cascFrms[-1].show()
### all these have three things in common 
#   -> they're returned off the backhand...
#   -> ... on the stretch
#   -> ... as a chip!
# In fact, there is virtually no way to return these as topspin balls!
#   -> no choice! forced chip return.
# we are interested in the opposite: 'choice' chips! 
# given that the receiver can make a choice to chip, we want to assess whether that choice is worth it.
revRst, revFrms = reverse_cascade(finalRst, keep_which = [(1, 1)], acceleration = 1.15)
DecapStream(revFrms, 0, 100).show()
revFrms[-1].show()

#################################
##### PROPENSITY SCORE MODEL ####
#################################
 
RESOLUTION = 10
renameDict = {'serve_speed': 'Serve Speed', 'serve_spin': 'Serve Topspin', 'serve_width': 'Serve Width', 
              'serve_depth': 'Return Depth', 'serve_height': 'Bounce Height', 'ret_is_back': 'Returner uses Backhand', 'serve_dist_sq': 'Stretch Distance'}
fitted_model = logReg
max_line_thickness = 2.5
thickness_taper = 0.05
model_name = 'Propensity to Chip'

from ATMANpy.types.Overlays import Diagram
from ATMANpy.types.ShapesTennis3D import Arrow

def model_to_diagram(fitted_model, renameDict = None, model_name = 'Propensity to Chip',
                     max_line_thickness = 5, min_line_thickness = 0.1, resolution = RESOLUTION, render_height = 1500, 
                     shape_color = const.BGR['strong_blue'], text_color = const.BGR['smokey_grey']):
    ### collect relevant fitted parameters
    bSer, pSer = fitted_model.params, fitted_model.pvalues    
    pSer_sorted = pSer.sort_values()
    pSer_log = np.log10(1 / pSer_sorted)
    pSer_normed = (pSer_log / pSer_log.max())
    thickSer = pSer_normed * max_line_thickness
    thickSer = thickSer.clip(lower = min_line_thickness)
    positiveSer = bSer > 0

    ### initialize diagram
    regressor_keys = pSer_sorted.index.to_list()
    regressor_names = [renameDict[key] for key in regressor_keys if key in renameDict]
    myDiag = Diagram([regressor_names, [], [model_name]], resolution = resolution, text_skip_after_how_many_symbols = 3,
                     shape_color = shape_color, shape_offset = 0.05, text_color = text_color, arrow_hat_length = 10, 
                     dimensions = (640, 360), render_height = render_height)

    inverted_renameDict = {v: k for k, v in renameDict.items()}

    for idx, name in enumerate(regressor_names):
        key = inverted_renameDict[name]

        arrow_thickness, is_positive = thickSer[key], positiveSer[key]
        if is_positive:
            arrow_color = const.BGR['leafy_green']
        else:
            arrow_color = const.BGR['intense_red']

        shape, arrow = myDiag[0, idx]
        remade_arrow = arrow.recolor(arrow_color).resize(arrow_thickness)
        myDiag[0, idx] = shape, remade_arrow

    return myDiag

outDiag = model_to_diagram(fitted_model, renameDict = renameDict, model_name = 'Propensity to Chip', 
                           max_line_thickness = 3, resolution = RESOLUTION, shape_color = const.BGR['marine_blue'], text_color = const.BGR['bright_grey'])

for col in range(7):
    outDiag[0, col][0].text_vertical_offset = -8

outDiag[0, 0][0].text_horizontal_offset = 2.5
outDiag[0, 1][0].text_horizontal_offset = 0.5
outDiag[0, 2][0].text_horizontal_offset = 5
outDiag[0, 3][0].text_horizontal_offset = 6.5
outDiag[0, 4][0].text_horizontal_offset = 5.5
outDiag[0, 5][0].text_horizontal_offset = 4
outDiag[0, 6][0].text_horizontal_offset = 5

outDiag[1, 0][0].text_skip_after_how_many_symbols = None
outDiag[1, 0][0].text_horizontal_offset = 50
outDiag[1, 0][0].text_vertical_offset = 0
outDiag[1, 0][0].text_size = 2
outDiag[1, 0][0].text_thickness = 2

### staggered diagram
def remove_all_diagram_shapes_after_index(myDiag, idx):
    myDiag.shapes[0] = [shape for i, shape in enumerate(myDiag.shapes[0]) if i <= idx]
    myDiag.arrows[0] = [arrow for i, arrow in enumerate(myDiag.arrows[0]) if i <= idx]
    return myDiag

outDiags = [remove_all_diagram_shapes_after_index(outDiag.deepcopy(), idx) for idx in range(7)]

for myDiag in outDiags:
    diagFrm = myDiag.render()
    diagFrm.show()

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

def region_to_line_to_arrow(myReg, annPnt, relative_line_thickness = 0.1, relative_line_length = 1.1,
                            resolution = 10, num_frames = 25, ann_right_px = 100, ann_left_px = 60, ann_vert_px = 20):

    p1, p2, p3, p4 = myReg.p1, myReg.p2, myReg.p3, myReg.p4
    h = myReg['Left'].length()
    w = myReg['Bottom'].length()

    line_thickness = relative_line_thickness * h
    line_length = relative_line_length * w

    vert_px = (h - line_thickness) / 2
    hori_px = (line_length - w) / 2

    q1 = p1.back(vert_px).left(hori_px)
    q2 = p2.back(vert_px).right(hori_px)
    q3 = p3.front(vert_px).left(hori_px)
    q4 = p4.front(vert_px).right(hori_px)
    thinReg = Region3D.from_points3D(q1, q2, q3, q4, color = myReg.color)
    thinReg = thinReg.front(h / 2)

    startScn = Scene([myReg])
    endScn = Scene([thinReg])

    itpl_scenes = vs.interpolate_scenes(startScn, endScn, num_frames)

    myFrms, animDict = vs.animate_sequence(itpl_scenes, annotated_points = [annPnt], start_cam = outDiag.tabCam, stop_cam = outDiag.tabCam,
                                            argument_resolution = resolution)
    
    startFrms = BGRFrames.from_frame(myFrms[0], num_frames)
        
    ### turn thinReg into an arrow
    leftPt = thinReg['Left'].center_point()
    rightPt = thinReg['Right'].center_point()

    thinArr = Arrow.make(leftPt, rightPt, hat_length = 10, color = thinReg.color, alpha = thinReg.alpha, thickness = line_thickness)
    startPt, endPt = thinArr.start_point(), thinArr.end_point()
    
    ### annotation points
    annPnt_left_text = startPt.annotate('Choice Chip').front(ann_vert_px).left(ann_left_px).recolor(annPnt.color).rethicken(2)
    annPnt_left_number = startPt.annotate('0').left(10).back(5).recolor(annPnt.color).resize(2).rethicken(2)

    annPnt_right_text = endPt.annotate('Forced Chip').front(ann_vert_px).right(ann_right_px).recolor(annPnt.color).rethicken(2)
    annPnt_right_number = endPt.annotate('1').right(2).back(5).recolor(annPnt.color).resize(2).rethicken(2)

    ### make the scenes
    arrScn = Scene([thinArr])

    arrView = View.make(outDiag.tabCam, arrScn, annotatedPoints = [annPnt, annPnt_left_number, annPnt_right_number], resolution = resolution)
    arrFrm = arrView.snap()
    arrFrms = BGRFrames.from_frame(arrFrm, num_frames)

    arrView2 = View.make(outDiag.tabCam, arrScn, annotatedPoints = [annPnt, annPnt_left_number, annPnt_right_number, annPnt_right_text], resolution = resolution)
    arrFrm2 = arrView2.snap()
    arrFrms2 = BGRFrames.from_frame(arrFrm2, num_frames)

    ### annotation left -> low propensity
    arrView3 = View.make(outDiag.tabCam, arrScn, annotatedPoints = [annPnt, annPnt_left_number, annPnt_right_number, annPnt_right_text, annPnt_left_text], resolution = resolution)
    arrFrm3 = arrView3.snap()
    arrFrms3 = BGRFrames.from_frame(arrFrm3, num_frames)
    
    ### join all
    jntFrms = BGRFrames.from_frames([startFrms, myFrms, arrFrms, arrFrms2, arrFrms3])
    return jntFrms

transFrms = region_to_line_to_arrow(myReg, annPnt, relative_line_thickness = 0.02, relative_line_length = 0.9,
                                    resolution = 10, num_frames = 25, ann_right_px = -45, ann_left_px = 0, ann_vert_px = 15)

DecapStream(transFrms, 0, 100).show()
transFrms[-1].show()

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

transFrm = transFrms[-1]
propRst = baseRst.deepcopy()
propRst.add(transFrm, 1, 1)

finalRst, cascFrms = frame_cascade(low_pred_exFrms, propRst)
DecapStream(cascFrms, 0, 100).show() 

cascFrms[-1].show()

# on these returns, the PSM believes that the returner could very well have played a topspin return; instead, the returner opted for a chipped return.
# In this sense, the PSM expresses choice. 
# We can now use these propensities to focus on chipped returns where the returner actually had a choice, 
# and thereby attempt to assess the quality of this choice.

# (show table with choice returns)

revRst, revFrms = reverse_cascade(finalRst, keep_which = [(1, 1)], acceleration = 1.15)
DecapStream(revFrms, 0, 100).show()
revFrms[-1].show()

emptyFrm = revFrms[-1].zero()
emptyFrm.show()

baseRst = FramesRaster(3, 3)
baseRst.add(basic_naive_table.render(), 1, 1)
baseRst.render()[0].show()

baseRst = FramesRaster(3, 3)
baseRst.add(basic_smart_table.render(), 1, 1)
baseRst.render()[0].show()

### still pretty underwhelming for the chipped return: choice chipped returns fare only moderately better.
### we try to understand why.

# (split -> make% and cond%)

baseRst = FramesRaster(3, 3)
baseRst.add(make_smart_table.render(), 1, 1)
baseRst.render()[0].show()

baseRst = FramesRaster(3, 3)
baseRst.add(cond_smart_table.render(), 1, 1)
baseRst.render()[0].show()

### TODO: DecapStream(low_pred_exFrms[-1], 0, 100).show(ms_between_frames = 50) ### is this a let?
### TODO: Staggering model influences.
### TODO: fix dots at end of arrows
