"""
Short version of the return analysis
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


import ATMANpy.auxil.constants as const
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Shots, read_in_data_tables, read_in_placeholder_tables
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

DESCRIPTOR = 'short'
SAVE_FOLDER = 'return_chip_topspin'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

#######################################
####### PARANETERS ####################
#######################################

DIMENSIONS = (1920, 1080)
DIMENSIONS_HALF = (DIMENSIONS[0] // 2, DIMENSIONS[1] // 2)
DIMENSIONS_THIRD = (DIMENSIONS[0] // 3, DIMENSIONS[1] // 3)
TABLE_RESOLUTION = 50

### TITLE CARD ###
from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.Lines import Region

title_dims = (900, 200)
eps = 2
title_x, title_y = title_dims
a, b, c, d = Point2D((eps, eps)), Point2D((title_x - eps, eps)), Point2D((eps, title_y - eps)), Point2D((title_x - eps, title_y - eps))
titleReg = Region.from_points(a, b, c, d)

titleFrm = BGRFrame.make_black(*title_dims).put(titleReg, color = const.BGR['bright_grey'], thickness = 3)

titleStr = 'On the Effectiveness of the Chipped Return'
titleFrm = titleFrm.put(titleStr, size = 2, where = (70, title_y // 2))

titleFrm.show()

### THE DATA ###
x, y = DIMENSIONS_HALF
index_name_t = 'Topspin Returns (on first serve)'
index_name_c = 'Chipped Returns (on first serve)'
index_name_a = 'Total' ### weirdly, need two extra spaces...
title = 'Returner Point Win%'

table_kwargs_small = {  'number_format': '.1%', 'index_offset': 0.25, 'index_col_width': 1.25, 'header_row_height': 0.5,
                        'skip_after_how_many_symbols': 15, 'dimensions': (x, y),
                        'skeleton_shrinkage': 0.3, 'skeleton_vertical_offset': 1, 
                        'title_vertical_offset': 0, 'title_horizontal_offset': 3.5,
                        'data_text_size': 2, 'header_text_size': 2, 'index_text_size': 2,
                        'header_line_thickness': 0.1, 'resolution': TABLE_RESOLUTION,
                        'title': '2021 Brest Challenger', 'title_size': 2, 'title_thickness': 1, 
                        'title_color': const.BGR['bright_grey'], 
                        'title_horizontal_offset': 8.6, 'title_vertical_offset': 0.5
                        }

num_chip, num_tops = regFrm['is_chipped'].sum(), (~regFrm['is_chipped']).sum()  
num_tot = num_tops + num_chip

str_chips = f'{num_chip:>4} ({num_chip/num_tot:.0%})'
str_tops = f'{num_tops:>4} ({num_tops/num_tot:.0%})'

tdict = {'Sample Size': str_tops}
cdict = {'Sample Size': str_chips}
totDict = {'Sample Size': str(num_tot)}
table_data = {index_name_t: tdict, index_name_c: cdict, index_name_a: totDict}

sample_table = Table(table_data, **table_kwargs_small, null_columns = 0)
tabFrm = sample_table.render()

tabFrm = tabFrm.put('THE DATA', color = const.BGR['bright_grey'], size = 2, thickness = 2)
tabFrm.show()

#### THE MODEL ###
from ATMANpy.types.Overlays import Diagram, DiagramShape

regressor_names = ['Srv. Placement', 'Srv. Side', 'Ret. Placement', 'Ret. Spin', 'Ret. Speed', 'Ret. Angle', '...']
regressor_names = ['Player\nPositions', 'Return\nPlacement',  '...', 'Return\nSpin', 'Return\nAngle']

model_name = 'Logistic Regression'
out_name = 'Returner Win Probability'

text_color = const.BGR['bright_grey']
shape_color = const.BGR['marine_blue']
arrow_color = const.BGR['bright_blue']

modDiag = Diagram([[], regressor_names, [], [model_name], [], [out_name], []], dimensions = DIMENSIONS_HALF, 
                  arrow_hat_length = 0.5, shape_offset = 0.05, resolution = 1000, text_size = 2,
                  text_color = text_color, arrow_color = arrow_color, shape_color = shape_color)

modDiag.shapes[0][2].shape_color = (0, 0, 0)
modDiag.arrows[0].pop(2)

myScn, annPnts = modDiag.preview()

annPnts_shifted = [annPnts[0].right(0.2), 
                   annPnts[1], 
                   annPnts[2].right(2), 
                   annPnts[3].right(0.5), 
                   annPnts[4].right(0.5),
                   annPnts[5].right(7.4).back(0.2),
                   annPnts[6].right(6.55).back(0.2)]

horiLn = myScn[6]['Bottom']

from ATMANpy.types.ShapesTennis3D import Arrow

pctArrow = Arrow.make(horiLn.leftPt, horiLn.rightPt, rot_angle = np.pi / 2, color = const.BGR['bright_grey'])

zeroPt = horiLn.leftPt.back(1.5).annotate('0%').resize(2)
onePt = horiLn.rightPt.back(1.5).left(2).annotate('100%').resize(2) 

annPnts_shifted = [*annPnts_shifted, zeroPt, onePt]

myScn.shapes[6] = pctArrow

from ATMANpy.types.Visual3D import View
diagView = View.make(modDiag.tabCam, myScn, annotatedPoints = annPnts_shifted, resolution = modDiag.resolution)
diagFrm = diagView.snap()
diagFrm = diagFrm.put('THE MODEL', color = const.BGR['bright_grey'], size = 2, thickness = 2)
diagFrm.show()

### RESULT TEXT ###

resFrm = BGRFrame.make_black(*DIMENSIONS_HALF)
resFrm = resFrm.put('THE RESULTS', color = const.BGR['bright_grey'], size = 2, thickness = 2)

y = 120
i = 30

text1 = "Chipped Returns are a (mostly) viable strategy,\nas long as they're played ..."


text1a = "   ... low ..."
text1b = "   ... to the server's forehand ..."
text1c = "   ... with sufficient depth and width."

text2 = "There is one exception, where the topspin return is always preferable."
text2a = "    Ad-side serve out wide from righty server."
text2b = "    Deuce-side serve out wide from lefty server."

text3 = "In all other cases, chip returns are high-risk/high-reward."
text3a = "    - The typical chip return has a smaller target zone than the typical topspin return."
text3b = "    - The typical chip return has a higher peak returner win probability."

resFrm_simple = resFrm.put(text1, where = (50, y), size = 2)
resFrm_simple = resFrm_simple.put(text1a, where = (50, y+3*i), size = 2)
resFrm_simple = resFrm_simple.put(text1b, where = (50, y+5*i), size = 2)
resFrm_simple = resFrm_simple.put(text1c, where = (50, y+7*i), size = 2)
resFrm_simple.show()

resFrm_simple = resFrm_simple.put(text2, where = (50, y+9*i), size = 2)
resFrm_simple.show()

resFrm_simple = resFrm_simple.put(text3, where = (50, y+11*i), size = 2)
resFrm_simple.show()


resFrm = resFrm.put(text1, where = (50, y), size = 2)
resFrm = resFrm.put(text2, where = (50, y + 4*i), size = 2)
resFrm = resFrm.put(text2a, where = (50, y + 5*i), size = 2)
resFrm = resFrm.put(text2b, where = (50, y+6*i), size = 2)

resFrm.show()








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
prmDict_nolum = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_NORMAL, argument_luminosity = 1, 
                              adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
prmDict_heat = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_HEATMAP, argument_luminosity = 1, 
                             adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
RESOLUTION_KDE = prmDict_heat['dummy_resolution']

MIX_COLORS_KDE = True
BANDWIDTH_KDE = 35
SCENE_FADE = 0.9
CONTOUR_PROP = 0.5
EPSILON = 0.002
EMPHASIS_DIM_FACTOR = 1.5
BLINK_RATE = 1

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
prmDict_annot_heat['argument_luminosity'] = 1.5

backView_half = View.make(halfCam, courtScn, resolution = 1, annotatedPoints = [fhPt_annot, bhPt_annot])
backView_heat = View.make(heatCam, courtScn, resolution = 1, annotatedPoints = [fhPt_annot_heat, bhPt_annot_heat])

# backView_half.snap().show()
# backView_heat.snap().show()

### Points over which to render
aPnt = retPnts_in.get(26, 40)
bPnt = retPnts_in.get(3, 141)
vis_ut.is_serve_down_middle(aPnt)
vis_ut.is_serve_out_wide(bPnt)
### b -> is out wide! can use it as initial example

def get_spline_frames(myPnt):
    serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').values()
    splnScns = [Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in splns if spln is not None]) for splns in serve_splines]
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
    outScn = Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in serve_splines if spln is not None])
    
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
myFrms = get_spline_frames(relPnts[39])
preFrms = get_wide_frames(relPnts[39], pre_frames = pre_frames)
myScn = get_spline_and_player_scene(relPnts[39])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['deuce_middle'] = myFrms
wideDict['deuce_middle'] = preFrms
sceneDict['deuce_middle'] = myScn

# c) ad, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[40]) ### not nakashima again.
preFrms = get_wide_frames(relPnts[40], pre_frames = pre_frames) 
myScn = get_spline_and_player_scene(relPnts[40])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['ad_middle'] = myFrms
wideDict['ad_middle'] = preFrms
sceneDict['ad_middle'] = myScn

# d) ad, out wide
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[96])
preFrms = get_wide_frames(relPnts[96], pre_frames = pre_frames) 
myScn = get_spline_and_player_scene(relPnts[96])
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
