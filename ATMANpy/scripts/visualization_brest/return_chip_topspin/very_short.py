"""
Short version of the return analysis
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Logit

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Shots, read_in_data_tables, read_in_placeholder_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.Visual3D import View, Scene, DummyScene, Camera
from ATMANpy.types.Shapes3D import Point3D, Line3D, Circle3D, Circles3D, Region3D
from ATMANpy.types.ShapesTennis3D import Player
from ATMANpy.types.Overlays import FrameRaster, FramesRaster, Table, overlay, overlay_center, make_text_in

from ATMANpy.funcs.util_funcs import interpolate_tuples

SAVE = False

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

from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

current_idx = 0

#######################################
####### PARANETERS ####################
#######################################

DIMENSIONS = (1920, 1080)
DIMENSIONS_HALF = (DIMENSIONS[0] // 2, DIMENSIONS[1] // 2)
DIMENSIONS_THIRD = (DIMENSIONS[0] // 3, DIMENSIONS[1] // 3)
TABLE_RESOLUTION = 50

FRAME_THICKNESS = 2
FRAME_COLOR = const.BGR['bright_grey']

BALL_COLOR = const.BGR['ball_yellow']
TOPS_COLOR = const.BGR['light_brown']
CHIP_LO_COLOR = const.BGR['marine_blue']
CHIP_HI_COLOR = const.BGR['bright_blue']
CHIP_COLOR = interpolate_tuples(CHIP_LO_COLOR, CHIP_HI_COLOR)

STATIC_FRAMES = 100

###############################
####### AUXILIARY STUFF #######
###############################

### Legends
from ATMANpy.types.Overlays import Legend, LegendRow, overlay

from ATMANpy.funcs.util_funcs import interpolate_tuples
CHIP_COLOR = interpolate_tuples(CHIP_LO_COLOR, CHIP_HI_COLOR)

from ATMANpy.types.Overlays import Legend, LegendRow, overlay
tpRow = LegendRow('... Topspin', shape_color = TOPS_COLOR, text_size = 2)
loRow = LegendRow('... Low Chip', shape_color = CHIP_LO_COLOR, text_size = 2)
hiRow = LegendRow('... High Chip', shape_color = CHIP_HI_COLOR, text_size = 2)
cpRow = LegendRow('... (Choice) Chip', shape_color = CHIP_COLOR, text_size = 2)

leg_dims = (300, 300)
# leg_dims = (500, 500)
leg_params = {'borders': True, 'header_text': 'Return is ...', 'resolution': 200, 'render_height': 300,
              'legend_dimensions': leg_dims, 'header_offset': 2, 'header_thickness': 1}

leg1 = Legend([tpRow, LegendRow(None), LegendRow(None)], **leg_params)
leg1Frm = leg1.render()
# leg1Frm.show()

leg2 = Legend([tpRow, loRow, LegendRow(None)], **leg_params)
leg2Frm = leg2.render()
# leg2Frm.show()

leg3 = Legend([tpRow, loRow, hiRow], **leg_params)
leg3Frm = leg3.render()
# leg3Frm.show()

top_left_coords = (1920 - leg_dims[0]) // 2, (1080 - leg_dims[1]) // 2
tlPt = Point2D(top_left_coords)

### Little Flag
def make_flag(basePt, height_of_flag = 10, width_of_flag = 20, color = const.BGR['intense_red'], alpha = 1):
    from ATMANpy.types.Shapes3D import Region3D, Line3D
    leftPt = basePt.up(height_of_flag / 2)
    rightPt = leftPt.right(width_of_flag)

    flagLn = Line3D(leftPt, rightPt)
    return Region3D.from_line3D(flagLn, width = height_of_flag, dim = 'z', left_right = "+x", top_bottom = '+z',
                                color = color, alpha = alpha)


### TITLE CARD ###
from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.Lines import Region

title_dims = (850, 100)
eps = 2
title_x, title_y = title_dims

titleStr = 'On the Effectiveness of the Chipped Return'

titleFrm = BGRFrame.make_black(*title_dims).put(titleStr, size = 2, where = (50, round(0.575 * title_y)))
# titleFrm.show()

titleFrm_framed = titleFrm.framed(FRAME_THICKNESS, FRAME_COLOR)
# titleFrm_framed.show()

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
                        'title_color': const.BGR['bright_grey'], 'data_text_color': const.BGR['leafy_green'],
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

sampleTab = Table(table_data, **table_kwargs_small, null_columns = 0)
tabScn, annPnts = sampleTab.preview()
headerPt = annPnts[0]

headerFrm = View.make(sampleTab.tabCam, Scene([]), annotatedPoints = [headerPt]).snap()
headerFrm = headerFrm.put('THE DATA', color = const.BGR['bright_grey'], size = 2, thickness = 2)
# headerFrm.show()

tabFrm = sampleTab.render()
tabFrm = tabFrm.put('THE DATA', color = const.BGR['bright_grey'], size = 2, thickness = 2)
# tabFrm.show()

headerFrm_framed = headerFrm.framed(FRAME_THICKNESS, FRAME_COLOR)
# headerFrm_framed.show()

tabFrm_framed = tabFrm.framed(FRAME_THICKNESS, FRAME_COLOR)
# tabFrm_framed.show()

#### THE MODEL ###
from ATMANpy.types.Overlays import Diagram, DiagramShape

regressor_names = ['Srv. Placement', 'Srv. Side', 'Ret. Placement', 'Ret. Spin', 'Ret. Speed', 'Ret. Angle', '...']
regressor_names = ['Player\nPositions', 'Return\nPlacement',  '...', 'Return\nSpin', 'Return\nAngle']

model_name = 'Logistic Regression'
out_name = 'Returner Win Probability'

text_color = const.BGR['bright_grey']
shape_color = const.BGR['marine_blue']
arrow_color = const.BGR['bright_blue']

diag_kwargs = {'dimensions': DIMENSIONS_HALF, 'arrow_hat_length': 0.5, 'shape_offset': 0.05, 
                'resolution': 1000, 'text_size': 2,
                'text_color': text_color, 'arrow_color': arrow_color, 'shape_color': shape_color}

modDiag_0 = Diagram([[], [], [], [model_name], [], [], []], **diag_kwargs)
modDiag_1 = Diagram([[], regressor_names, [], [model_name], [], [], []], **diag_kwargs)
modDiag_2 = Diagram([[], regressor_names, [], [model_name], [], [out_name], []], **diag_kwargs)

myScn_0, annPnts_0 = modDiag_0.preview()

annPnts_shifted_0 = [annPnts_0[0].right(7.4).back(0.2)]

modDiag_1.shapes[0][2].shape_color = (0, 0, 0)
modDiag_1.arrows[0].pop(2)

myScn_1, annPnts_1 = modDiag_1.preview()

annPnts_shifted_1 = [annPnts_1[0].right(0.2), 
                    annPnts_1[1], 
                    annPnts_1[2].right(2), 
                    annPnts_1[3].right(0.5), 
                    annPnts_1[4].right(0.5),
                    annPnts_1[5].right(7.4).back(0.2)]

modDiag_2.shapes[0][2].shape_color = (0, 0, 0)
modDiag_2.arrows[0].pop(2)

myScn_2, annPnts_2 = modDiag_2.preview()

annPnts_shifted_2 = [annPnts_2[0].right(0.2), 
                   annPnts_2[1], 
                   annPnts_2[2].right(2), 
                   annPnts_2[3].right(0.5), 
                   annPnts_2[4].right(0.5),
                   annPnts_2[5].right(7.4).back(0.2),
                   annPnts_2[6].right(6.55).back(0.2)]

horiLn = myScn_2[6]['Bottom']

from ATMANpy.types.ShapesTennis3D import Arrow

pctArrow = Arrow.make(horiLn.leftPt, horiLn.rightPt, rot_angle = np.pi / 2, color = const.BGR['bright_grey'])

zeroPt = horiLn.leftPt.back(1.5).annotate('0%').resize(2)
onePt = horiLn.rightPt.back(1.5).left(2).annotate('100%').resize(2) 

annPnts_shifted_2 = [*annPnts_shifted_2, zeroPt, onePt]

myScn_2.shapes[6] = pctArrow

from ATMANpy.types.Visual3D import View
diagView = View.make(modDiag_0.tabCam, myScn_0, annotatedPoints = annPnts_shifted_0, resolution = modDiag_0.resolution)
diagFrm_0 = diagView.snap()
diagFrm_0 = diagFrm_0.put('THE MODEL', color = const.BGR['bright_grey'], size = 2, thickness = 2)
# diagFrm_0.show()

diagFrm_0_framed = diagFrm_0.framed(FRAME_THICKNESS, FRAME_COLOR)
# diagFrm_0_framed.show()

diagView = View.make(modDiag_1.tabCam, myScn_1, annotatedPoints = annPnts_shifted_1, resolution = modDiag_1.resolution)
diagFrm_1 = diagView.snap()
diagFrm_1 = diagFrm_1.put('THE MODEL', color = const.BGR['bright_grey'], size = 2, thickness = 2)
# diagFrm_1.show()

diagFrm_1_framed = diagFrm_1.framed(FRAME_THICKNESS, FRAME_COLOR)
# diagFrm_1_framed.show()

diagView = View.make(modDiag_2.tabCam, myScn_2, annotatedPoints = annPnts_shifted_2, resolution = modDiag_2.resolution)
diagFrm_2 = diagView.snap()
diagFrm_2 = diagFrm_2.put('THE MODEL', color = const.BGR['bright_grey'], size = 2, thickness = 2)
# diagFrm_2.show()

diagFrm_2_framed = diagFrm_2.framed(FRAME_THICKNESS, FRAME_COLOR)
# diagFrm_2_framed.show()

### RESULT TEXT ###
resFrm = BGRFrame.make_black(*DIMENSIONS_HALF)
resFrm = resFrm.put('THE RESULTS', color = const.BGR['bright_grey'], size = 2, thickness = 2)

y = 120
i = 30

text1 = "Chipped Returns are a (mostly) viable strategy,\nas long as they're played ..."

text1a = "   ... low ..."
text1b = "   ... with sufficient depth and width ..."
text1c = "   ... to the server's forehand."

text2 = ' --> High-Risk / High-Reward'

# text2 = "Chip returns are high-risk/high-reward, except ..."
# text2a = "    - The typical chip return has a smaller target zone than the typical topspin return."
# text2b = "    - The typical chip return has a higher peak returner win probability."

# text3 = "... in one situation, where topspin is preferable."
# text3a = "    Ad-side serve out wide from righty server."
# text3b = "    Deuce-side serve out wide from lefty server."

resFrm_simple1 = resFrm.put(text1, where = (50, y), size = 2, color = const.BGR['leafy_green'])
resFrm_simple1a = resFrm_simple1.put(text1a, where = (50, y+3*i), size = 2, color = const.BGR['leafy_green'])
resFrm_simple1b = resFrm_simple1a.put(text1b, where = (50, y+5*i), size = 2, color = const.BGR['leafy_green'])
resFrm_simple1c = resFrm_simple1b.put(text1c, where = (50, y+7*i), size = 2, color = const.BGR['leafy_green'])
# resFrm_simple1c.show() 

resFrm_simple2 = resFrm_simple1c.put(text2, where = (50, y+round(9.5*i)), size = 2, color = const.BGR['leafy_green'])
# resFrm_simple2.show()

# resFrm_simple3 = resFrm_simple2.put(text3, where = (50, y+12*i), size = 2)
# resFrm_simple3.show()

resFrm_simple1_framed = resFrm_simple1.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple1a_framed = resFrm_simple1a.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple1b_framed = resFrm_simple1b.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple1c_framed = resFrm_simple1c.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple2_framed = resFrm_simple2.framed(FRAME_THICKNESS, FRAME_COLOR)
# resFrm_simple3_framed = resFrm_simple3.framed(FRAME_THICKNESS, FRAME_COLOR)

# resFrm_simple3_framed.show()

footnoteStr = 'test footnote'
footnoteStr = 'test footnote\ntwo lines'
footnoteStr = 'test footnote\ntwo lines\nthree lines'
timeStr = '12:00-13:00:'

def overlay_footnote(someFrms, timeStr, footnoteStr):
    footnote_where = (50, 950)
    footnote_dims = (500, 100)
    
    blackFrm = BGRFrame.make_black(*footnote_dims)

    footnoteFrm = blackFrm.put(timeStr, size = 1, where = (30, 40), color = const.BGR['grass_green'])
    footnoteFrm = footnoteFrm.put(footnoteStr, size = 1, where = (150, 40), color = const.BGR['grass_green'])

    footnoteFrm_framed = footnoteFrm.framed(FRAME_THICKNESS, FRAME_COLOR)
    # footnoteFrm_framed.show()

    return overlay(someFrms, footnoteFrm_framed, top_left_point = Point2D(footnote_where))
    

### INITIAL RASTER ###
all_frames = []
script_lines = []

current_idx = 0

STILL_FRAMES = 10

script_line =   """
                    The primary goal of this analysis is to understand the effectiveness of the chip return as compared to the topspin return. 
                """  ### 22 sec until 'placed'

blackFrm = BGRFrame.make_black(*DIMENSIONS)
black_with_titleFrm = overlay_center(blackFrm, titleFrm)
# black_with_titleFrm.show()
black_with_titleFrms = BGRFrames.from_frame(black_with_titleFrm, STILL_FRAMES)

all_frames.append(black_with_titleFrms)
script_lines.append(script_line)
# del black_with_titleFrms

script_line =   """
                    The engine of our analysis is a statistical model...
                """ # 3 sec


startRst = FrameRaster(2, 2)

startRst.add(diagFrm_0_framed, 0, 0)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

itplFrms = vs.interpolate_frames(black_with_titleFrm, rstFrm, 25)

jointFrms = BGRFrames.from_frames([itplFrms, rstFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)
# del rstFrms
    
script_line =   """
                    ... that takes into account a great variety of relevant variables ...
                """ # 4 sec

startRst.add(diagFrm_1_framed, 0, 0)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)
# del rstFrms

script_line =   """
                    ... and spits out an estimated probability that the returner wins the point.
                """ # 5 sec

startRst.add(diagFrm_2_framed, 0, 0)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)
# del rstFrms

script_line =   """
                    The model is estimated on 1.451 first serve returns observed in 20 matches played during the 2021 Brest Challenger tournament.
                """ # 6 sec

startRst.add(tabFrm_framed, 1, 0)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)
# del rstFrms

script_line =   """
                    We find that a good chip return should be played low ...
                """ # 8 sec

startRst.add(resFrm_simple1_framed, 0, 1)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)
# del rstFrms

script_line =   """
                    ... and with sufficient depth and width ...
                """ # 3 sec

startRst.add(resFrm_simple1a_framed, 0, 1)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)         
# del rstFrms

script_line =   """
                    ... to the server's forehand.
                """ # 3 sec

startRst.add(resFrm_simple1b_framed, 0, 1)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)  
# del rstFrms

script_line =   """
                    If these three guidelines are followed, the chip return can be a high-risk/high-reward alternative to the topspin return.
                """ # 10 sec

startRst.add(resFrm_simple1c_framed, 0, 1)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms_a = BGRFrames.from_frame(rstFrm, 75)

startRst.add(resFrm_simple2_framed, 0, 1)
rstFrm = startRst.render()
# rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms_b = BGRFrames.from_frame(rstFrm, 50)

rstFrms = BGRFrames.from_frames([rstFrms_a, rstFrms_b])

all_frames.append(rstFrms)
script_lines.append(script_line)  
# del rstFrms

script_line =   """
                    Before commenting on these (partially surprising) results, let's look at the model's suggestions in greater depth.
                """ # 14 sec
    
all_frames.append(rstFrms_b)      
script_lines.append(script_line)  
# del rstFrms

vis_ut.show(all_frames)

if SAVE:
    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)

### -> total: 93 seconds of text.

all_frames = []
script_lines = []

### PARAMS FOR ANIMATION ###
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


RESOLUTION_MODIFIER_NORMAL = 1 #35
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
    myFrms, _ = vs.animate_sequence(splnScns, start_cam = halfCam, stop_cam = halfCam, cumulate_argument = False, 
                                    animate_over = realFrms.rescale(1.5), **prmDict)
    return myFrms

def get_wide_frames(myPnt, pre_frames = 150):
    preDtm = myPnt.stream(confs = confs).widen(pre_frames).decap()
    preFrms = preDtm.frames[:pre_frames]
    preArr_dimmed = np.stack([f.dim((idx / pre_frames) ** 2).array for idx, f in enumerate(preFrms)], axis = 0)
    preFrms_dimmed = BGRFrames(preArr_dimmed).rescale(1.5)
    return preFrms_dimmed

def get_spline_and_player_scene(myPnt, player_color = const.BGR['smokey_grey']):
    _, serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').last()
    outScn = Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in serve_splines if spln is not None])
    
    x0, y0 = myPnt[0].get_player_hit_position()
    x1, y1 = myPnt[1].get_player_hit_position()

    srvPly = Player.make((x0, y0), color = player_color)
    recPly = Player.make((x1, y1), color = player_color)

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

# View.make(halfCam, myScn.combine(courtScn)).snap().show()

# b) deuce, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_deuce').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[39])
preFrms = get_wide_frames(relPnts[39], pre_frames = pre_frames // 2)
myScn = get_spline_and_player_scene(relPnts[39])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['deuce_middle'] = myFrms
wideDict['deuce_middle'] = preFrms
sceneDict['deuce_middle'] = myScn

# c) ad, down middle
relPnts = retPnts_in.filter(vis_ut.is_serve_down_middle).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[40]) ### not nakashima again.
preFrms = get_wide_frames(relPnts[40], pre_frames = pre_frames // 2) 
myScn = get_spline_and_player_scene(relPnts[40])
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 100)
# myFrms[-1].show()
realDict['ad_middle'] = myFrms
wideDict['ad_middle'] = preFrms
sceneDict['ad_middle'] = myScn

# d) ad, out wide
relPnts =  retPnts_in.filter(vis_ut.is_serve_out_wide).filter('is_serve_ad').filter(is_return_topspin)
myFrms = get_spline_frames(relPnts[96])
preFrms = get_wide_frames(relPnts[96], pre_frames = pre_frames // 2) 
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
                                    'ret_dist_to_bh_serve_fore_sq_x_launch_angle', 'ret_dist_to_bh_serve_back_sq_x_launch_angle'],
        'nochip_launch_speed_rpm_inter3': ['launch_angle', 'ret_speed', 'ret_spin', 'ret_dist_to_bsl', 'ret_dist_to_opt', 'ret_dist_to_opt_sq',
                                    'ret_dist_to_bh', 'ret_dist_to_bh_sq', 'ret_dist_to_bsl_bh',
                                    'ret_dist_to_bsl_x_launch_angle', 
                                    'ret_dist_to_opt_x_launch_angle', 'ret_dist_to_opt_sq_x_launch_angle',
                                    'ret_dist_to_bh_x_launch_angle', 'ret_dist_to_bh_sq_x_launch_angle',
                                    'ret_dist_to_bsl_x_ret_speed', 
                                    'ret_dist_to_opt_x_ret_speed', 'ret_dist_to_opt_sq_x_ret_speed',
                                    'ret_dist_to_bh_x_ret_speed', 'ret_dist_to_bh_sq_x_ret_speed',
                                    'ret_dist_to_bh_x_is_chipped', 'ret_dist_to_bh_sq_x_is_chipped',
                                    'ret_dist_to_bsl_x_is_chipped',
                                    'ret_dist_to_opt_x_is_chipped', 'ret_dist_to_opt_sq_x_is_chipped']}

###### ESTIMATE MODEL ######
reg_name = 'nochip_launch_speed_rpm_inter3'
regressors = regDict[reg_name]

endog = 'ret_wins_weighted'
weight = 'weight_bal_pred'

glmReg = vis_ut.make_and_save_model(regFrm_in, regressors, save_path = None, save_name = None, 
                                                endog_col = endog, weight_col = weight)
    


########## SETUP #########
dimRst = FramesRaster(2, 2, borders = None)
movRst = FramesRaster(2, 2, borders = None)

prevRst_tops = FramesRaster(2, 2, borders = None)
prevRst_tops_wide = FramesRaster(2, 2, borders = None)
prevRst_tops_middle = FramesRaster(2, 2, borders = None)
prevRst_chip = FramesRaster(2, 2, borders = None)
prevRst_zone = FramesRaster(2, 2, borders = None)
prevRst_green = FramesRaster(2, 2, borders = None)
prevRst_red = FramesRaster(2, 2, borders = None)

# turn the starting raster into a moving raster
tlFrm, trFrm, blFrm = startRst[0, 0], startRst[0, 1], startRst[1, 0]
tlFrms, trFrms, blFrms = BGRFrames.from_frame(tlFrm, 150), BGRFrames.from_frame(trFrm, 150), BGRFrames.from_frame(blFrm, 150)

tlFrms_dim = tlFrms.dim_progressively(ls_dim_factor = 0.2)
trFrms_dim = trFrms.dim_progressively(ls_dim_factor = 0.2)
blFrms_dim = blFrms.dim_progressively(ls_dim_factor = 0.2)

tlFrms_dim_last = BGRFrames.from_frame(tlFrms_dim[-1], 25)
trFrms_dim_last = BGRFrames.from_frame(trFrms_dim[-1], 25)
blFrms_dim_last = BGRFrames.from_frame(blFrms_dim[-1], 25)

dimRst.add(tlFrms_dim, 0, 0)
dimRst.add(trFrms_dim, 0, 1)
dimRst.add(blFrms_dim, 1, 0)

movRst.add(tlFrms_dim_last, 0, 0)
movRst.add(trFrms_dim_last, 0, 1)
movRst.add(blFrms_dim_last, 1, 0)

# del tlFrms, trFrms, blFrms
# del tlFrms_dim, trFrms_dim, blFrms_dim
# del tlFrms_dim_last, trFrms_dim_last, blFrms_dim_last

####
dim_factor = 0.25
annot_color = const.BGR['bright_grey']
bandwidth_field = 50

postRst = FramesRaster(2, 2, borders = 2)
backFrm = View.make(halfCam, courtScn).snap()

comparison_prob_dict = {'tp': 0.55, ### this way, target areas will have roughly the same size:
                        'lo': 0.55,      ### we then annotate their quality with annotation point
                        'hi': 0.5}

color_dict = {'tp': TOPS_COLOR,
              'lo': CHIP_LO_COLOR,
              'hi': CHIP_HI_COLOR}

TRANS_FACTOR = 0.5
DIM_FACTOR_REDGREEN = 0.5
FRAME_COLOR_GREEN = const.BGR['leafy_green']
FRAME_COLOR_RED = const.BGR['intense_red']

###### DEUCE WIDE ######
side, where = 'deuce', 'wide'
i, j = 1, 1

# frames to render on
preFrms = wideDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms = realDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms_last = BGRFrames.from_frame(realFrms[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
pvalDict = {}
zoneDict = {}
zoneDict_trans = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg,
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1

    zoneDict_trans[angle] = rastScn1.transparify(TRANS_FACTOR)

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    The first situation we evaluate is a serve out wide from the server's forehand side, 
                    that is, the deuce side for a right-handed server or the ad side for a lefty server.
                """  # 9 sec

dimRst.add(preFrms, i, j)
rastFrms = dimRst.render()

# # fade out title frame
# titleFrms = BGRFrames.from_frame(titleFrm, pre_frames)
# titleFrms_framed = BGRFrames.from_frame(titleFrm_framed, pre_frames)

# # overlay the title with fade out
# rastFrms1 = overlay_center(rastFrms, titleFrms_framed, start_alpha = 1, stop_alpha = 0, alpha_exp = 4)
# # DecapStream(rastFrms, 0, 100).show()

rastFrms1 = rastFrms

movRst.add(realFrms, i, j)
rastFrms2 = movRst.render()
# rastFrms[-1].show()

movRst.add(realFrms_last, i, j)
rastFrms3 = movRst.render()

joint_rastFrms = BGRFrames.from_frames([rastFrms1, rastFrms2, rastFrms3])

all_frames.append(joint_rastFrms)
script_lines.append(script_line)
# del joint_rastFrms, rastFrms1, rastFrms2, rastFrms3

script_line =   """
                    The orange area shows the target zone for the returning player on a topspin shot.
                """

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES).framed(FRAME_THICKNESS, FRAME_COLOR)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
movRst.add(myFrms, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)
# del rastFrms, myFrms

script_line =   """
                    The target zone is where the model suggests to aim the return; if the ball
                    lands within the target zone, the returning player has a probability of winning the subsequent
                    point greater than 50%.
                """
rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)

rastFrms_static = overlay_footnote(rastFrms_static, "XX:XX-XX:XX:", "For more info on the construction\nand interpretation of target zones.")

all_frames.append(rastFrms_static)
script_lines.append(script_line)
# del rastFrms_static

script_line =   """
                    In this situation, our model believes it is best to hit the topspin return cross-court.
                """ 

script_line = "" ### a little pause before the next segment
rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)

all_frames.append(rastFrms_static)
script_lines.append(script_line)
# del rastFrms_static

script_line =   """
                    As for a low chip return, the model suggests going cross-court also, at a more acute angle.

                    Note that we differentiate between a low chip and a high chip. 
                    The low chip is closer to a traditional slice, whereas ...
                """ # 21 sec

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
movRst.add(myFrms, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg2Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)
# del rastFrms

script_line =   """
                    ... the high chip is a floating chip with a high arc and a high bounce.
                """

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
movRst.add(myFrms, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)
# del myFrms, rastFrms

script_line =   """
                    We differentiate between these two kinds of chip return 
                    because our data shows that the low chip and the high chip return are distinct shots 
                    that are employed with purpose.
                """

rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)

rastFrms_static = overlay_footnote(rastFrms_static, "XX:XX-XX:XX:", "For more info on the difference\nbetween low and high chip returns.")

all_frames.append(rastFrms_static)
script_lines.append(script_line)

script_line =   """
                    In any case, the target zone for the high chip is considerably smaller and deeper than the other zones.
                """ 

rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    Let's zoom in on the target zones a bit more closely.
                """ # 3 sec

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

postFrms = BGRFrames.from_frames([transFrms, moveFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)

movRst.add(postFrms, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)
# del transFrms, moveFrms, postFrms, rastFrms

script_line =   """
                    The flags indicate the highest returner win probability spot within the target zone, for each of our three return types. 
                """ # 15 sec

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 10), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = TRANSITION_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)

postFrms = BGRFrames.from_frames([lineFrms, statFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)
movRst.add(postFrms, i, j)

rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)
# del lineFrms, statFrms, postFrms, rastFrms

## for later
combScn_trans = combScn.transparify(TRANS_FACTOR)

# -> only topspin zones lit up
zoneScn_tops = zoneDict['tp'].combine(zoneDict_trans['lo']).combine(zoneDict_trans['hi']).consolidate()
statFrms_tops, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_tops, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_tops = statFrms_tops.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only chip zones lit up
zoneScn_chip = zoneDict_trans['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
statFrms_chip, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_chip, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_chip = statFrms_chip.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only flags lit up
zoneScn_zone = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
statFrms_zone, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_zone, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_zone = statFrms_zone.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> framed with color
statFrms_green = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR_GREEN)
statFrms_red = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR).dim(DIM_FACTOR_REDGREEN)

prevRst_tops.add(statFrms_tops, i, j)
prevRst_tops_wide.add(statFrms_tops, i, j)
prevRst_tops_middle.add(statFrms_tops.dim(DIM_FACTOR_REDGREEN), i, j)
prevRst_chip.add(statFrms_chip, i, j)
prevRst_zone.add(statFrms_zone, i, j)
prevRst_green.add(statFrms_green, i, j)
prevRst_red.add(statFrms_red, i, j)

prevRst = movRst.to_frame_raster().to_frames_raster(25)

FAST_FRAMES = 10

###### DEUCE MIDDLE ######
side, where = 'deuce', 'middle'
i, j = 0, 1

# frames to render on
preFrms = wideDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms = realDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms_last = BGRFrames.from_frame(realFrms[-1], 25) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
pvalDict = {}
zoneDict = {}
zoneDict_trans = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg,
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1

    zoneDict_trans[angle] = rastScn1.transparify(TRANS_FACTOR)

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Let's quickly cycle through the other cases. First, a serve DOWN THE MIDDLE from the server's forehand side.
                """ # 5 sec

prevRst.add(preFrms, i, j)
rastFrms1 = prevRst.render()

prevRst.add(realFrms, i, j)
rastFrms2 = prevRst.render()

rastFrms_a = BGRFrames.from_frames([rastFrms1, rastFrms2])

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = FAST_FRAMES).framed(FRAME_THICKNESS, FRAME_COLOR)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], FAST_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
prevRst.add(myFrms, i, j)
rastFrms_b = prevRst.render()

rastFrms_c = BGRFrames.from_frame(rastFrms_b[-1], FAST_FRAMES)

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = FAST_FRAMES)
prevRst.add(myFrms, i, j)

rastFrms_d = prevRst.render()

# fade in hi chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = FAST_FRAMES)
prevRst.add(myFrms, i, j)

rastFrms_e = prevRst.render()

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms[-1].dim(dim_factor), backFrm_with_shapes, num_frames = FAST_FRAMES)
zoneScns = [zoneScn for _ in range(FAST_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

postFrms = BGRFrames.from_frames([transFrms, moveFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)

prevRst.add(postFrms, i, j)

rastFrms_f = prevRst.render()

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 10), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = FAST_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = FAST_FRAMES)

postFrms = BGRFrames.from_frames([lineFrms, statFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)

prevRst.add(postFrms, i, j)

rastFrms_g = prevRst.render()

jointFrms = BGRFrames.from_frames([rastFrms_a, rastFrms_b, rastFrms_c, 
                                   rastFrms_d, rastFrms_e, rastFrms_f, 
                                   rastFrms_g])
jointFrms = overlay(jointFrms, leg3Frm, top_left_point = tlPt)

DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

## for later
combScn_trans = combScn.transparify(TRANS_FACTOR)

# -> only topspin zones lit up
zoneScn_tops = zoneDict['tp'].combine(zoneDict_trans['lo']).combine(zoneDict_trans['hi']).consolidate()
statFrms_tops, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_tops, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_tops = statFrms_tops.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only chip zones lit up
zoneScn_chip = zoneDict_trans['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
statFrms_chip, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_chip, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_chip = statFrms_chip.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only flags lit up
zoneScn_zone = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
statFrms_zone, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_zone, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_zone = statFrms_zone.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> framed with color
statFrms_green = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR_GREEN)
statFrms_red = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR).dim(DIM_FACTOR_REDGREEN)

prevRst_tops.add(statFrms_tops, i, j)
prevRst_tops_wide.add(statFrms_tops.dim(DIM_FACTOR_REDGREEN), i, j)
prevRst_tops_middle.add(statFrms_tops, i, j)
prevRst_chip.add(statFrms_chip, i, j)
prevRst_zone.add(statFrms_zone, i, j)
prevRst_green.add(statFrms_green, i, j)
prevRst_red.add(statFrms_red, i, j)

prevRst = prevRst.to_frame_raster().to_frames_raster(25)

###### AD MIDDLE ######
side, where = 'ad', 'middle'
i, j = 0, 0

# frames to render on
preFrms = wideDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms = realDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms_last = BGRFrames.from_frame(realFrms[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
pvalDict = {}
zoneDict = {}
zoneDict_trans = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg,
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1
    
    zoneDict_trans[angle] = rastScn1.transparify(TRANS_FACTOR)

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Consider now a serve down the middle from the server's BACKHAND side,
                    that is: the ad side for a right-handed server, or the deuce side for a lefty server.
                """ # 9 sec

prevRst.add(preFrms, i, j)
rastFrms1 = prevRst.render()

prevRst.add(realFrms, i, j)
rastFrms2 = prevRst.render()

rastFrms_a = BGRFrames.from_frames([rastFrms1, rastFrms2])

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = FAST_FRAMES).framed(FRAME_THICKNESS, FRAME_COLOR)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], FAST_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
prevRst.add(myFrms, i, j)
rastFrms_b = prevRst.render()

rastFrms_c = BGRFrames.from_frame(rastFrms_b[-1], FAST_FRAMES)

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = FAST_FRAMES)
prevRst.add(myFrms, i, j)

rastFrms_d = prevRst.render()

# fade in hi chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = FAST_FRAMES)
prevRst.add(myFrms, i, j)

rastFrms_e = prevRst.render()

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms[-1].dim(dim_factor), backFrm_with_shapes, num_frames = FAST_FRAMES)
zoneScns = [zoneScn for _ in range(FAST_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

postFrms = BGRFrames.from_frames([transFrms, moveFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)

prevRst.add(postFrms, i, j)

rastFrms_f = prevRst.render()

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 10), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = FAST_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = FAST_FRAMES)

postFrms = BGRFrames.from_frames([lineFrms, statFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)

prevRst.add(postFrms, i, j)

rastFrms_g = prevRst.render()

jointFrms = BGRFrames.from_frames([rastFrms_a, rastFrms_b, rastFrms_c, 
                                   rastFrms_d, rastFrms_e, rastFrms_f, 
                                   rastFrms_g])
jointFrms = overlay(jointFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(jointFrms)
script_lines.append(script_line)

DecapStream(jointFrms, 0, 100).show()

## for later
combScn_trans = combScn.transparify(TRANS_FACTOR)

# -> only topspin zones lit up
zoneScn_tops = zoneDict['tp'].combine(zoneDict_trans['lo']).combine(zoneDict_trans['hi']).consolidate()
statFrms_tops, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_tops, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_tops = statFrms_tops.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only chip zones lit up
zoneScn_chip = zoneDict_trans['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
statFrms_chip, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_chip, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_chip = statFrms_chip.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only flags lit up
zoneScn_zone = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
statFrms_zone, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_zone, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_zone = statFrms_zone.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> framed with color
statFrms_green = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR_GREEN)
statFrms_red = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR).dim(DIM_FACTOR_REDGREEN)

prevRst_tops.add(statFrms_tops, i, j)
prevRst_tops_wide.add(statFrms_tops.dim(DIM_FACTOR_REDGREEN), i, j)
prevRst_tops_middle.add(statFrms_tops, i, j)
prevRst_chip.add(statFrms_chip, i, j)
prevRst_zone.add(statFrms_zone, i, j)
prevRst_green.add(statFrms_green, i, j)
prevRst_red.add(statFrms_red, i, j)

prevRst = prevRst.to_frame_raster().to_frames_raster(25)

###### AD WIDE ######
side, where = 'ad', 'wide'
i, j = 1, 0

# frames to render on
preFrms = wideDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms = realDict[side + '_' + where].framed(FRAME_THICKNESS, FRAME_COLOR)
realFrms_last = BGRFrames.from_frame(realFrms[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
pvalDict = {}
zoneDict = {}
zoneDict_trans = {}
for angle in ['tp', 'lo', 'hi']:
    comparison_prob = comparison_prob_dict[angle]
    color = color_dict[angle]

    regressor_frame = get_reg_frame(angle, side, where)

    ### zone scene new
    rastArr_upscaled, colArr, alArr, max_coords, max_val = vis_ut.field_array_kernelized(glmReg, regressor_frame, relative_to = comparison_prob, showReg = showReg,
                                                                                bandwidth = bandwidth_field, factor_array = shareArr, color = color)
    
    ### make zone scene
    zArr = np.zeros_like(rastArr_upscaled[:, [0]])
    rastArr_upscaled_3D = np.concatenate([rastArr_upscaled, zArr], axis = 1)
    rastScn1 = DummyScene([rastArr_upscaled_3D], [colArr], [alArr])
    zoneDict[angle] = rastScn1

    zoneDict_trans[angle] = rastScn1.transparify(TRANS_FACTOR)

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Finally, we have a serve out wide from the server's backhand half of the court.
                """ # 6 sec

prevRst.add(preFrms, i, j)
rastFrms1 = prevRst.render()

prevRst.add(realFrms, i, j)
rastFrms2 = prevRst.render()

rastFrms_a = BGRFrames.from_frames([rastFrms1, rastFrms2])

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = FAST_FRAMES).framed(FRAME_THICKNESS, FRAME_COLOR)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], FAST_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
prevRst.add(myFrms, i, j)
rastFrms_b = prevRst.render()

rastFrms_c = BGRFrames.from_frame(rastFrms_b[-1], FAST_FRAMES)

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = FAST_FRAMES)
prevRst.add(myFrms, i, j)

rastFrms_d = prevRst.render()

# fade in hi chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['hi']).combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = FAST_FRAMES)
prevRst.add(myFrms, i, j)

rastFrms_e = prevRst.render()

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms[-1].dim(dim_factor), backFrm_with_shapes, num_frames = FAST_FRAMES)
zoneScns = [zoneScn for _ in range(FAST_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

postFrms = BGRFrames.from_frames([transFrms, moveFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)

prevRst.add(postFrms, i, j)

rastFrms_f = prevRst.render()

### Annotation frames 
tpPt, tp_max = pvalDict['tp']
loPt, lo_max = pvalDict['lo']
hiPt, hi_max = pvalDict['hi']

### Line Points -> up!
up_pts = 20
tpPt_up, loPt_up, hiPt_up = tpPt.up(up_pts + 10), loPt.up(up_pts + 10), hiPt.up(up_pts - 5)

### Little Flag
tpReg = make_flag(tpPt_up, color = TOPS_COLOR, height_of_flag = 9, width_of_flag = 20)
loReg = make_flag(loPt_up, color = CHIP_LO_COLOR, height_of_flag = 9, width_of_flag = 20)
hiReg = make_flag(hiPt_up, color = CHIP_HI_COLOR, height_of_flag = 9, width_of_flag = 20)

# ### Annotations
tpPt_annot = make_text_in(tpReg, f'{tp_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
loPt_annot = make_text_in(loReg, f'{lo_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
hiPt_annot = make_text_in(hiReg, f'{hi_max:.0%}', color = (0, 0, 0), number_format = '.0%').down(3).right(1)
annotPts = [tpPt_annot, loPt_annot, hiPt_annot]

nullScn = Scene([Line3D(tpPt, tpPt, color = annot_color), Line3D(loPt, loPt, color = annot_color), Line3D(hiPt, hiPt, color = annot_color)])
lineScn = Scene([Line3D(tpPt, tpPt_up, color = annot_color), Line3D(loPt, loPt_up, color = annot_color), Line3D(hiPt, hiPt_up, color = annot_color)])
itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = FAST_FRAMES)

lineFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, 
                                    background_scene = fullScn, previous_dummy_scene = zoneScn, **prmDict_annot_heat)

combScn = lineScn.put(tpReg).put(loReg).put(hiReg)
statFrms, _ = vs.animate_static(combScn, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = FAST_FRAMES)

postFrms = BGRFrames.from_frames([lineFrms, statFrms]).framed(FRAME_THICKNESS, FRAME_COLOR)

prevRst.add(postFrms, i, j)

rastFrms_g = prevRst.render()

jointFrms = BGRFrames.from_frames([rastFrms_a, rastFrms_b, rastFrms_c, 
                                   rastFrms_d, rastFrms_e, rastFrms_f, 
                                   rastFrms_g])
jointFrms = overlay(jointFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(jointFrms)
script_lines.append(script_line)

DecapStream(jointFrms, 0, 100).show()

## for later
combScn_trans = combScn.transparify(TRANS_FACTOR)

# -> only topspin zones lit up
zoneScn_tops = zoneDict['tp'].combine(zoneDict_trans['hi']).combine(zoneDict_trans['lo']).consolidate()
statFrms_tops, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_tops, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_tops = statFrms_tops.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only chip zones lit up
zoneScn_chip = zoneDict_trans['tp'].combine(zoneDict['hi']).combine(zoneDict['lo']).consolidate()
statFrms_chip, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_chip, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_chip = statFrms_chip.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> only flags lit up
zoneScn_zone = zoneDict['tp'].combine(zoneDict['hi']).combine(zoneDict['lo']).consolidate()
statFrms_zone, _ = vs.animate_static(combScn_trans, static_cam = heatCam, 
                                background_scene = fullScn, previous_dummy_scene = zoneScn_zone, 
                                annotated_points = annotPts, **prmDict_annot_heat, num_frames = TRANSITION_FRAMES)
statFrms_zone = statFrms_zone.framed(FRAME_THICKNESS, FRAME_COLOR)

# -> framed with color
statFrms_green = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR).dim(DIM_FACTOR_REDGREEN)
statFrms_red = statFrms.framed(FRAME_THICKNESS, FRAME_COLOR_RED)

prevRst_tops.add(statFrms_tops, i, j)
prevRst_tops_wide.add(statFrms_tops, i, j)
prevRst_tops_middle.add(statFrms_tops.dim(DIM_FACTOR_REDGREEN), i, j)
prevRst_chip.add(statFrms_chip, i, j)
prevRst_zone.add(statFrms_zone, i, j)
prevRst_green.add(statFrms_green, i, j)
prevRst_red.add(statFrms_red, i, j)

prevRst = prevRst.to_frame_raster().to_frames_raster(25)

### explanations

topsFrms = prevRst_tops.render()
topsFrms_wide = prevRst_tops_wide.render()
topsFrms_middle = prevRst_tops_middle.render()
chipFrms = prevRst_chip.render()
zoneFrms = prevRst_zone.render()
greenFrms = prevRst_green.render()
redFrms = prevRst_red.render()

topsFrms = overlay(topsFrms, leg3Frm, top_left_point = tlPt)
topsFrms_wide = overlay(topsFrms_wide, leg3Frm, top_left_point = tlPt)
topsFrms_middle = overlay(topsFrms_middle, leg3Frm, top_left_point = tlPt)
chipFrms = overlay(chipFrms, leg3Frm, top_left_point = tlPt)
zoneFrms = overlay(zoneFrms, leg3Frm, top_left_point = tlPt)
greenFrms = overlay(greenFrms, leg3Frm, top_left_point = tlPt)
redFrms = overlay(redFrms, leg3Frm, top_left_point = tlPt)

topsFrms[-1].show()
topsFrms_wide[-1].show()
topsFrms_middle[-1].show()
chipFrms[-1].show()
zoneFrms[-1].show()
greenFrms[-1].show()
redFrms[-1].show()

script_line =   """
                    As for placement, the topspin return should be played...
                """
## -> topspin zones lit

itplFrms = vs.interpolate_frames(jointFrms[-1], topsFrms[0], TRANSITION_FRAMES)
joinFrms = BGRFrames.from_frames([itplFrms, topsFrms])
# DecapStream(joinFrms, 0, 100).show()

all_frames.append(joinFrms)
script_lines.append(script_line)

script_line =   """
                    ... cross-court if pulled out wide ...
                """

itplFrms = vs.interpolate_frames(topsFrms[-1], topsFrms_wide[0], TRANSITION_FRAMES)
joinFrms = BGRFrames.from_frames([itplFrms, topsFrms_wide])
# DecapStream(joinFrms, 0, 100).show()

all_frames.append(joinFrms)
script_lines.append(script_line)

script_line =   """
                    ... and anywhere between the server's feet and away from the server if returning from the middle.
                """

itplFrms = vs.interpolate_frames(topsFrms_wide[-1], topsFrms_middle[0], TRANSITION_FRAMES)
joinFrms = BGRFrames.from_frames([itplFrms, topsFrms_middle])
# DecapStream(joinFrms, 0, 100).show()

all_frames.append(joinFrms)
script_lines.append(script_line)

script_line =   """
                    - > the chip return should be aimed at the server's forehand, according to the model. 
                """
## -> chip zones lit
                
itplFrms = vs.interpolate_frames(topsFrms_middle[-1], chipFrms[0], TRANSITION_FRAMES)
joinFrms = BGRFrames.from_frames([itplFrms, chipFrms])
# DecapStream(joinFrms, 0, 100).show()

all_frames.append(joinFrms)
script_lines.append(script_line)

script_line =   """
                    A chip of any kind typically has a smaller target zone than the topspin return.
                    in this sense, it is high risk: there is less room for error.
                """
## -> all zones lit, flags not

itplFrms = vs.interpolate_frames(chipFrms[-1], zoneFrms[0], TRANSITION_FRAMES)
joinFrms = BGRFrames.from_frames([itplFrms, zoneFrms])
# DecapStream(joinFrms, 0, 100).show()

all_frames.append(joinFrms)
script_lines.append(script_line)

script_line =   """
                    On the other hand, in three out of the four cases, 
                    the peak returner win probability is higher for the low chip than the topspin return:
                    in this sense, it is high reward.
                """
## -> flags lit, zones lit
## green borders around three cases
itplFrms = vs.interpolate_frames(zoneFrms[-1], greenFrms[0], TRANSITION_FRAMES)
joinFrms = BGRFrames.from_frames([itplFrms, greenFrms])
# DecapStream(joinFrms, 0, 100).show()

all_frames.append(joinFrms)
script_lines.append(script_line)

script_line =   """
                    The high chip return, having the smallest target area and the lowest peak probability, is cautioned against by our model.            
                """
## maybe cut

script_line =   """
	                The exception is the serve out wide from the server's backhand side. 
                                    
                    (The high chip zone is larger than the low chip zone, which is probably due to the fact that the returner plays himself out of position 
                    going down the line, and therefore needs as much time to recover as possible: the high chip, with its long travel time, affords that. 
                    Nevertheless, )

                    The topspin has both the highest peak and the largest zone, making it clearly the best option.
                """
## -> flags lit, zones lit
## red border around exception.

itplFrms = vs.interpolate_frames(greenFrms[-1], redFrms[0], TRANSITION_FRAMES)
joinFrms = BGRFrames.from_frames([itplFrms, redFrms])
# DecapStream(joinFrms, 0, 100).show()

all_frames.append(joinFrms)
script_lines.append(script_line)

script_line =    """
                    That the chip return is a viable alternative to the topspin return 
                    if played to the server's forehand is both an unintuitive and uncommon strategy.
                    Unintuitive, because, in the aggregate, it appears that the chip return is a bad choice.
                """ # 15 sec

statFrms = BGRFrames.from_frame(redFrms[-1], 300)
statFrms = statFrms.dim_progressively()

all_frames.append(statFrms)
script_lines.append(script_line)
# del statFrms

vis_ut.show(all_frames)

### Saving the separate MultiFrame objects and the Joint Script
if SAVE:
    assert len(script_lines) == len(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)

### INTERPRETATION ###

all_frames = []
script_lines = []

# 1) Results are surprising and not common practice
    # a) surprising: on aggregate, the chip return has a worse win rate
    # b) not common practice: if a chip return is done, it goes mostly to the backhand (like topspin returns).

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

###########################################
### Tables with incremental adjustments ###
###########################################

x, y = DIMENSIONS_THIRD
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

tdict = {'': t0}
cdict = {'': c0}
table_data = {index_name_t: tdict, index_name_c: cdict}
table_data_half = {index_name_t: tdict}

initial_naive_table = Table(table_data, **table_kwargs_small, title = title, header_text_size = 0)
# initial_naive_table.render().show()

naive_table_half = Table(table_data_half, **table_kwargs_large, title = title, null_columns = 3, null_rows = 1)
naive_frm_half = naive_table_half.render()
# naive_frm_half.show()

naive_table = Table(table_data, **table_kwargs_large, title = title, null_columns = 3)
naive_frm = naive_table.render()
# naive_frm.show()

tdict = {'Unadjusted': t0, '+Choice Adj.': t1}
cdict = {'Unadjusted': c0, '+Choice Adj.': c1}
table_data = {index_name_t: tdict, index_name_c: cdict}

smart_table = Table(table_data, **table_kwargs_large, title = title, null_columns = 2)
smart_frm = smart_table.render()
# smart_frm.show()

tdict = {'Unadjusted': t0, '+Choice Adj.': t1, '+Balance Adj.': t2}
cdict = {'Unadjusted': c0, '+Choice Adj.': c1, '+Balance Adj.': c2}
table_data = {index_name_t: tdict, index_name_c: cdict}

smart_bal_table = Table(table_data, **table_kwargs_large, title = title, null_columns = 1)
smart_bal_frm = smart_bal_table.render()
# smart_bal_frm.show()

tdict = {'Unadjusted': t0, '+Choice Adj.': t1, '+Balance Adj.': t2, '+Attrib. Adj.': t3}
cdict = {'Unadjusted': c0, '+Choice Adj.': c1, '+Balance Adj.': c2, '+Attrib. Adj.': c3}
table_data = {index_name_t: tdict, index_name_c: cdict}

smart_bal_len_table = Table(table_data, **table_kwargs_large, title = title, null_columns = 0)
smart_bal_len_frm = smart_bal_len_table.render()
# smart_bal_len_frm.show()

r_f, c_f = 1080, 1920
r_t, c_t, _ = naive_frm.shape
r_d, c_d = (r_f - r_t) // 2, (c_f - c_t) // 2
table_offsetPt = Point2D((c_d, r_d))

script_line = "Over our entire sample, the returner's win percentage is ..."

tabScn, annPnts = naive_table.preview()
headerPt = annPnts[0]

headerFrm = View.make(naive_table.tabCam, Scene([]), annotatedPoints = [headerPt]).snap()
headerFrm_large = overlay(BGRFrame.make_black(), headerFrm, top_left_point = table_offsetPt)
headerFrms_large = BGRFrames.from_frame(headerFrm_large, 50)

all_frames.append(headerFrms_large)
script_lines.append(script_line)
# del headerFrms_large

script_line = "... 39.6 % when returning with topspin ..."
naive_frm_half_large = overlay(BGRFrame.make_black(), naive_frm_half, top_left_point = table_offsetPt)
naiveFrms_half_large = BGRFrames.from_frame(naive_frm_half_large, 25)

all_frames.append(naiveFrms_half_large)
script_lines.append(script_line)
# del naiveFrms_half_large

script_line = "... but only 19.9% when playing a chip."
naive_frm_large = overlay(BGRFrame.make_black(), naive_frm, top_left_point = table_offsetPt)
naiveFrms_large = BGRFrames.from_frame(naive_frm_large, 25)

all_frames.append(naiveFrms_large)
script_lines.append(script_line)
# del naiveFrms_large

### Tables with Arrows

# a) naive table
script_line = "A seemingly insurmountable deficit of almost 20%."

tabScn, annPnts = naive_table.preview()

topsPt, chipPt = annPnts[-2].right(1.25).back(0.5), annPnts[-1].right(1.25).front(1)
deltArr_1 = Arrow.make(topsPt, chipPt, color = const.BGR['intense_red'])
tabScn = tabScn.put(deltArr_1)

delStr_1 = f'{c0 - t0:.1%}'
deltPt_1 = (topsPt + chipPt) / 2
deltPt_1 = deltPt_1.right(0.1)
deltPt_1 = deltPt_1.annotate(delStr_1).resize(2).recolor(const.BGR['intense_red'])

annPnts = [*annPnts, deltPt_1]

naive_frm_with_arrow = View.make(naive_table.tabCam, tabScn, annPnts, resolution = TABLE_RESOLUTION).snap()
# naive_frm_with_arrow.show()

naive_frm_with_arrow_large = overlay(BGRFrame.make_black(), naive_frm_with_arrow, top_left_point = table_offsetPt)
naiveFrms_with_arrow_large = BGRFrames.from_frame(naive_frm_with_arrow_large, 25)

all_frames.append(naiveFrms_with_arrow_large)
script_lines.append(script_line)
# del naiveFrms_with_arrow_large

# b) final table
script_line =   """
                    Even when applying adjustments that serve to negate some statistical biases in the naive average...
                """

TRANS_ARROW = 0.6

tabScn, annPnts = smart_table.preview()
tabScn = tabScn.put(deltArr_1.transparify(TRANS_ARROW))
annPnts = [*annPnts, deltPt_1.dim(TRANS_ARROW)]
smart_frm_with_arrow = View.make(naive_table.tabCam, tabScn, annPnts, resolution = TABLE_RESOLUTION).snap()

smart_frm_large = overlay(BGRFrame.make_black(), smart_frm_with_arrow, top_left_point = table_offsetPt)
smartFrms_large = BGRFrames.from_frame(smart_frm_large, 25)

tabScn, annPnts = smart_bal_table.preview()
tabScn = tabScn.put(deltArr_1.transparify(TRANS_ARROW ** 2))
annPnts = [*annPnts, deltPt_1.dim(TRANS_ARROW ** 2)]
smart_bal_frm_with_arrow = View.make(naive_table.tabCam, tabScn, annPnts, resolution = TABLE_RESOLUTION).snap()

smart_bal_frm_large = overlay(BGRFrame.make_black(), smart_bal_frm_with_arrow, top_left_point = table_offsetPt)
smart_balFrms_large = BGRFrames.from_frame(smart_bal_frm_large, 25)

tabScn, annPnts = smart_bal_len_table.preview()
tabScn = tabScn.put(deltArr_1.transparify(TRANS_ARROW ** 3))
annPnts = [*annPnts, deltPt_1.dim(TRANS_ARROW ** 3)]
smart_bal_frm_with_arrow = View.make(naive_table.tabCam, tabScn, annPnts, resolution = TABLE_RESOLUTION).snap()

smart_bal_len_frm_large = overlay(BGRFrame.make_black(), smart_bal_frm_with_arrow, top_left_point = table_offsetPt)
smart_bal_lenFrms_large = BGRFrames.from_frame(smart_bal_len_frm_large, 25)

jointFrms = BGRFrames.from_frames([naiveFrms_with_arrow_large, smartFrms_large, smart_balFrms_large, smart_bal_lenFrms_large])

jointFrms = overlay_footnote(jointFrms, "XX:XX-XX:XX:", "For more info on these statistical\nadjustments.")

all_frames.append(jointFrms)
script_lines.append(script_line)

### 
tabScn, annPnts = smart_bal_len_table.preview()

topsPt, chipPt = annPnts[-5].right(1.25).back(0.5), annPnts[-1].right(1.25).front(1)

tabScn = tabScn.put(deltArr_1.transparify(TRANS_ARROW ** 4))
annPnts = [*annPnts, deltPt_1.dim(TRANS_ARROW ** 4)]

deltArr = Arrow.make(topsPt, chipPt, color = const.BGR['intense_red'])
tabScn = tabScn.put(deltArr)

delStr = f'{c3 - t3:.1%}'
deltPt = (topsPt + chipPt) / 2
deltPt = deltPt.right(0.1)
deltPt = deltPt.annotate(delStr).resize(2).recolor(const.BGR['intense_red'])

annPnts = [*annPnts, deltPt]

smart_bal_len_frm_with_arrow = View.make(smart_bal_len_table.tabCam, tabScn, annPnts, resolution = TABLE_RESOLUTION).snap()
# smart_bal_len_frm_with_arrow.show()

script_line = "... the difference in win rate is still around 12%."
smart_bal_len_frm_large_with_arrow = overlay(BGRFrame.make_black(), smart_bal_len_frm_with_arrow, top_left_point = table_offsetPt)
smart_bal_lenFrms_large_with_arrow = BGRFrames.from_frame(smart_bal_len_frm_large_with_arrow, 50)

smart_bal_lenFrms_large_with_arrow = overlay_footnote(smart_bal_lenFrms_large_with_arrow, "XX:XX-XX:XX:", "For more info on these statistical\nadjustments.")

all_frames.append(smart_bal_lenFrms_large_with_arrow)
script_lines.append(script_line)

# del jointFrms, naiveFrms_large, smartFrms_large, smart_balFrms_large, smart_bal_lenFrms_large, smart_bal_lenFrms_large_with_arrow

# DecapStream(jointFrms, 0, 100).show()

### ZONE MAPS 
retShots = retPnts_in.get_returns()
retShots_rectified = vis_ut.rectify_shots(retShots)
chip_retShots_rectified = retShots_rectified.filter('is_slice_visual')
tops_retShots_rectified = retShots_rectified.filter('is_slice_visual', negate = True)

script_line =   """
                    Beyond being unintuitive, targeting the forehand on a chip return is also slightly uncommon.
                """

fadeFrms = BGRFrames.from_frame(smart_bal_len_frm_large_with_arrow, 100).dim_progressively()

all_frames.append(fadeFrms)
script_lines.append(script_line)
# del fadeFrms

script_line =   """
                    Taking a look where the chip return typically lands...
                """

headerFrm = BGRFrame.make_black().put('CHIP RETURN LOCATION', color = CHIP_COLOR, size = 2, thickness = 2)
headerFrms = BGRFrames.from_frame(headerFrm, 50)

all_frames.append(headerFrms)
script_lines.append(script_line)
# del headerFrms

### a) Chip Shares
script_line =   """
                    ... we see a clear preference for the backhand side on aggregate:
                """
a, b, c, d, dc, ad = vis_ut.get_court_sectors(side = 'Top')
shrSer = vis_ut.shares_in_sector(chip_retShots_rectified, side = 'Top')
shrStrs = [f'{share:>3.0%}' for share in shrSer.values]

myRegs = [  Region3D.from_region2D(a, z = 0), Region3D.from_region2D(b, z = 0), 
            Region3D.from_region2D(c, z = 0), Region3D.from_region2D(d, z = 0), 
            Region3D.from_region2D(dc, z = 0), Region3D.from_region2D(ad, z = 0)]
myLns = [regLn.recolor(CHIP_COLOR).resize(1) for reg in myRegs for regLn in reg.lineList]

centerPts = [reg.center_point().annotate(shrStr).resize(2).recolor(CHIP_COLOR).left(10) for reg, shrStr in zip(myRegs, shrStrs)]
annotPts = [*centerPts, fhPt_annot_heat.resize(2), bhPt_annot_heat.resize(2)]

zoneFrms, animDict = vs.animate_static(myLns, background_scene = courtScn, static_cam = heatCam.resize(2), **prmDict_nolum,
                                    annotated_points = annotPts, num_frames = 50)

zoneFrms = zoneFrms.put('CHIP RETURN LOCATION', color = CHIP_COLOR, size = 2, thickness = 2)

all_frames.append(zoneFrms)
script_lines.append(script_line)
# del zoneFrms

### lines converging to show 40/60 split
script_line =   """
                    only 40% of returns land in the forehand half of the server's court.
                """
x, y, z = 1, 1, 3
s = x + y + z
shorten_factor = 0.125
line_color = const.BGR['bright_grey']

c0, c1, c2, c3, c4, c5 = centerPts

c0 = c0.right(10).back(5)
c1 = c1.right(7).back(5)
c4 = c4.right(5).front(5)

c2 = c2.right(10).back(5)
c3 = c3.right(7).back(5)
c5 = c5.right(5).front(5)

midlfPt = (c0 * x + c1 * y + c4 * z) // s
midrtPt = (c2 * x + c3 * y + c5 * z) // s

l0 = Line3D(c0, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l1 = Line3D(c1, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l2 = Line3D(c2, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color) 
l3 = Line3D(c3, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l4 = Line3D(c4, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l5 = Line3D(c5, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)

midlfPt_annot = midlfPt.annotate('40%').recolor(CHIP_COLOR).resize(2).rethicken(2).left(5).back(2)
midrtPt_annot = midrtPt.annotate('60%').recolor(CHIP_COLOR).resize(2).rethicken(2).left(5).back(2)

annotPts_mid = [*annotPts, midlfPt_annot, midrtPt_annot]

l0_null = Line3D(c0, c0).recolor(line_color)
l1_null = Line3D(c1, c1).recolor(line_color)
l2_null = Line3D(c2, c2).recolor(line_color)
l3_null = Line3D(c3, c3).recolor(line_color)
l4_null = Line3D(c4, c4).recolor(line_color)
l5_null = Line3D(c5, c5).recolor(line_color)

nullScn = Scene([l0_null, l1_null, l2_null, l3_null, l4_null, l5_null])
lineScn = Scene([l0, l1, l2, l3, l4, l5])

itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = 50)

itplFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam.resize(2), stop_cam = heatCam.resize(2),
                                annotated_points = annotPts, background_scene = courtScn,
                                **prmDict_nolum, **animDict)
itplFrms = itplFrms.put('CHIP RETURN LOCATION', color = CHIP_COLOR, size = 2, thickness = 2)


statFrms, _ = vs.animate_static(Scene([l0, l1, l2, l3, l4, l5]), static_cam = heatCam.resize(2),
                              annotated_points = annotPts_mid, background_scene = courtScn,
                              **prmDict_nolum, **animDict, num_frames = 50)
statFrms = statFrms.put('CHIP RETURN LOCATION', color = CHIP_COLOR, size = 2, thickness = 2)

jointFrms = BGRFrames.from_frames([itplFrms, statFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)
# del jointFrms, itplFrms, statFrms

script_line = ""
statFrms_dim = BGRFrames.from_frame(statFrms[-1], 25).dim_progressively()

all_frames.append(statFrms_dim)
script_lines.append(script_line)

### b) Topspin Shares
script_line = "For topspin returns ..."

headerFrm = BGRFrame.make_black().put('TOPSPIN RETURN LOCATION', color = TOPS_COLOR, size = 2, thickness = 2)
headerFrms = BGRFrames.from_frame(headerFrm, 25)

all_frames.append(headerFrms)
script_lines.append(script_line)

script_line = "... we see virtually the same overall distribution."

a, b, c, d, dc, ad = vis_ut.get_court_sectors(side = 'Top')
shrSer = vis_ut.shares_in_sector(tops_retShots_rectified, side = 'Top')
shrStrs = [f'{share:>3.0%}' for share in shrSer.values]

myRegs = [  Region3D.from_region2D(a, z = 0), Region3D.from_region2D(b, z = 0), 
            Region3D.from_region2D(c, z = 0), Region3D.from_region2D(d, z = 0), 
            Region3D.from_region2D(dc, z = 0), Region3D.from_region2D(ad, z = 0)]
myLns = [regLn.recolor(TOPS_COLOR).resize(1) for reg in myRegs for regLn in reg.lineList]

centerPts = [reg.center_point().annotate(shrStr).resize(2).recolor(TOPS_COLOR).left(10) for reg, shrStr in zip(myRegs, shrStrs)]
annotPts = [*centerPts, fhPt_annot_heat.resize(2), bhPt_annot_heat.resize(2)]

zoneFrms, animDict = vs.animate_static(myLns, background_scene = courtScn, static_cam = heatCam.resize(2), **prmDict_nolum,
                                    annotated_points = annotPts, num_frames = 50)

zoneFrms = zoneFrms.put('TOPSPIN RETURN LOCATION', color = TOPS_COLOR, size = 2, thickness = 2)

all_frames.append(zoneFrms)
script_lines.append(script_line)
# del zoneFrms

### lines converging to show 40/60 split
script_line =   """
                    Again, the split is 60/40 in favor of the backhand side.
                    However, only for chip returns, this split is inconsistent with our model results.
                """
                    
x, y, z = 1, 1, 3
s = x + y + z
shorten_factor = 0.125
line_color = const.BGR['bright_grey']

c0, c1, c2, c3, c4, c5 = centerPts

c0 = c0.right(10).back(5)
c1 = c1.right(7).back(5)
c4 = c4.right(5).front(5)

c2 = c2.right(10).back(5)
c3 = c3.right(7).back(5)
c5 = c5.right(5).front(5)

midlfPt = (c0 * x + c1 * y + c4 * z) // s
midrtPt = (c2 * x + c3 * y + c5 * z) // s

l0 = Line3D(c0, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l1 = Line3D(c1, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l2 = Line3D(c2, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color) 
l3 = Line3D(c3, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l4 = Line3D(c4, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
l5 = Line3D(c5, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)

midlfPt_annot = midlfPt.annotate('40%').recolor(TOPS_COLOR).resize(2).rethicken(2).left(5).back(2)
midrtPt_annot = midrtPt.annotate('60%').recolor(TOPS_COLOR).resize(2).rethicken(2).left(5).back(2)

annotPts_mid = [*annotPts, midlfPt_annot, midrtPt_annot]

l0_null = Line3D(c0, c0).recolor(line_color)
l1_null = Line3D(c1, c1).recolor(line_color)
l2_null = Line3D(c2, c2).recolor(line_color)
l3_null = Line3D(c3, c3).recolor(line_color)
l4_null = Line3D(c4, c4).recolor(line_color)
l5_null = Line3D(c5, c5).recolor(line_color)

nullScn = Scene([l0_null, l1_null, l2_null, l3_null, l4_null, l5_null])
lineScn = Scene([l0, l1, l2, l3, l4, l5])

itplScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = 50)

itplFrms, _ = vs.animate_sequence(itplScns, start_cam = heatCam.resize(2), stop_cam = heatCam.resize(2),
                                annotated_points = annotPts, background_scene = courtScn,
                                **prmDict_nolum, **animDict)
itplFrms = itplFrms.put('TOPSPIN RETURN LOCATION', color = TOPS_COLOR, size = 2, thickness = 2)


statFrms, _ = vs.animate_static(Scene([l0, l1, l2, l3, l4, l5]), static_cam = heatCam.resize(2),
                              annotated_points = annotPts_mid, background_scene = courtScn, num_frames = 75,
                              **prmDict_nolum, **animDict)
statFrms = statFrms.put('TOPSPIN RETURN LOCATION', color = TOPS_COLOR, size = 2, thickness = 2)

jointFrms = BGRFrames.from_frames([itplFrms, statFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)
# del jointFrms, itplFrms, statFrms

script_line =   """
                    So chipping returns to the forehand is unintuitive and relatively uncommon. 
                """
statFrms_dim = BGRFrames.from_frame(statFrms[-1], 25).dim_progressively()

all_frames.append(statFrms_dim)
script_lines.append(script_line)

# DecapStream(jointFrms, 0, 100).show()

### idea for a "agreement" metric: to what extent does empirical behaviour agree with model results
### -> = loc % / peak % ----> how much of the win% is not realized.

### idea for interpretation "teaser": 
# -> show three factors (repositioning, impatience, run-around) in 3x3 raster
# top left: bullet points for intepretation
# as each of the three factor bullets lights up, show 8 examples of actual play.

from ATMANpy.funcs.util_funcs import interpolate_tuples

text_color = const.BGR['bright_grey']
bright_color = const.BGR['leafy_green']
dark_color = interpolate_tuples(bright_color, (0, 0, 0), itpl_lambda = 0.4)

itpRst = FramesRaster(3, 3)

bulletFrm = BGRFrame.make_black(*DIMENSIONS_THIRD)

bulletFrm = bulletFrm.put('', color = const.BGR['bright_grey'], size = 2, thickness = 2)

y = 60
i = 30

text1 = "Chip returns to the forehand are\nunintuitive and uncommon.\n\nBut they can be explained:"

text1a = "   1) Run-around threat"
text1b = "   2) Time to reposition"
text1c = "   3) Impatience on chip return"

bulletFrm1 = bulletFrm.put(text1, where = (50, y), size = 2, color = text_color)

bulletFrm1a = bulletFrm1.put(text1a, where = (50, round(y+4.5*i)), size = 2, color = bright_color)

bulletFrm1b = bulletFrm1.put(text1a, where = (50, round(y+4.5*i)), size = 2, color = dark_color)
bulletFrm1b = bulletFrm1b.put(text1b, where = (50, round(y+6.5*i)), size = 2, color = bright_color)

bulletFrm1c = bulletFrm1.put(text1a, where = (50, round(y+4.5*i)), size = 2, color = dark_color)
bulletFrm1c = bulletFrm1c.put(text1b, where = (50, round(y+6.5*i)), size = 2, color = dark_color)
bulletFrm1c = bulletFrm1c.put(text1c, where = (50, round(y+8.5*i)), size = 2, color = bright_color)

# bulletFrm1.show()
# bulletFrm1a.show()
# bulletFrm1b.show()
# bulletFrm1c.show()

bulletFrms1 = BGRFrames.from_frame(bulletFrm1, 50).framed(FRAME_THICKNESS, FRAME_COLOR)
bulletFrms1a = BGRFrames.from_frame(bulletFrm1a, 50).framed(FRAME_THICKNESS, FRAME_COLOR)
bulletFrms1b = BGRFrames.from_frame(bulletFrm1b, 50).framed(FRAME_THICKNESS, FRAME_COLOR)
bulletFrms1c = BGRFrames.from_frame(bulletFrm1c, 50).framed(FRAME_THICKNESS, FRAME_COLOR)

script_line =   """
                    Nevertheless, three factors lead our model to consider it a viable alternative to the topspin return.

                    A chip return, especially when played with a higher launch angle, 
                    travels for longer than a typical topspin return. 
                """

exRst = FramesRaster(3, 3)
exRst.add(bulletFrms1, 0, 0)
exFrms = exRst.render()

all_frames.append(exFrms)
script_lines.append(script_line)
# del exFrms, bulletFrms1

### find 8 examples for each bullet point.
def get_example_spline_scenes(myPnt, third_color = BALL_COLOR):
    ### splines
    serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').values()
    return_splines = myPnt[1].feat_spline_cumul(which_endpoint = 'penultimate').values()
    third_splines = myPnt[2].feat_spline_cumul(which_endpoint = 'penultimate').values()

    serveScns = [Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in splns if spln is not None]) for splns in serve_splines]
    returnScns = [Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in splns if spln is not None]) for splns in return_splines]
    thirdScns = [Scene([spln.recolor(third_color).resize(1.5) for spln in splns if spln is not None]) for splns in third_splines]
    splnScns = serveScns + returnScns + thirdScns

    ### pad splines to length
    to_length = myPnt[:3].get_frames()[2][-1] - myPnt[:3].get_frames()[0][0] - 2 ### adjust for overlaps
    to_pad = to_length - len(splnScns)
    splnScns_pad = [Scene([]) for _ in range(to_pad)]
    splnScns = [*splnScns, *splnScns_pad]

    return splnScns

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

def shot_is_where(myPnt, shot_idx, areas):
    return shot_lands_in_or_near_area(myPnt, shot_idx = shot_idx, num_subdivisions = 4) in areas

def at_least_three_shots(myPnt):
    return len(myPnt) >= 3

def return_is_chipped(myPnt):
    return myPnt.get_return().is_slice_visual()

def play(myPnts):
    import time
    for idx, pt in enumerate(myPnts):
        print(idx)
        try:            
            ### real frames
            realFrms = pt[:3].stream(confs = confs).decap().frames
            ### splines
            splnScns = get_example_spline_scenes(pt)
            ### render    
            myCam = Camera.calibrated_brest_behind()
            exFrms, _ = vs.animate_sequence(splnScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False,
                                            animate_over = realFrms, 
                                            background_resolution = 1, argument_resolution = 0.5, 
                                            foreground_resolution = 0.5, argument_luminosity = 1)
        except:
            continue
        DecapStream(exFrms.rescale(2), 0, 100).show()
        time.sleep(2)
        print('___')

t2bPnts = dataPnts.filter('is_top_to_bot').filter(vis_ut.server_is_right_handed)
lenPnts = t2bPnts.filter(at_least_three_shots)
chipPnts = lenPnts.filter(return_is_chipped)

# a) run around threat
checkPnts = chipPnts.filter(shot_is_where, shot_idx = 1, areas = [3, 4])

def third_is_fore(myPnt):
    return myPnt[2].is_fore()

checkPnts = checkPnts.filter(third_is_fore)

checkPnts = checkPnts.filter(shot_is_where, shot_idx = 2, areas = [1, 4])

# play(checkPnts)

# candidates: found from play:
cands_a = [3, 4, 6, 9, 13, 16, 18, 20, 22, 32, 34, 37, 39, 41]
candPnts = [checkPnts[idx] for idx in cands_a]
# play(candPnts)

cands_a = [4, 9, 13, 16, 18, 20, 32, 39]
cands_a = [4, 9, 13, 18, 20, 32, 39]
candPnts = [checkPnts[idx] for idx in cands_a]

rast_frames = []
row, col = 0, 0

exRst = FramesRaster(3, 3)
exRst.add(bulletFrms1a, col, row)
exFrms = exRst.render()
rast_frames.append(exFrms)

script_line =   """
                    This allows chip returns to the backhand to be run around,
                    allowing the server to go for a forehand inside-out or inside-in. 
                """

exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on the impact of\nthe run-around serve+1")
rast_frames.append(exFrms)

def get_return_motion(myPnt, color = const.BGR['intense_red'], cumulative_transparence = 0.95, max_hat_len = 20, which = 'rec', next_frames = 0):
    if which == 'rec':
        currFrm = myPnt[1].recPlayer.playerFrm[['x', 'y']]
    else:
        currFrm = myPnt[1].hitPlayer.playerFrm[['x', 'y']]

    if which == 'rec':
        nextFrm = myPnt[2].hitPlayer.playerFrm[['x', 'y']]
    else:
        nextFrm = myPnt[2].recPlayer.playerFrm[['x', 'y']]

    playerFrm = pd.concat([currFrm[:-1], nextFrm[:next_frames]], axis = 0)
    playerFrm = playerFrm.rolling(window = 15, min_periods = 1).mean()
    playerFrm = playerFrm.reset_index(drop = True)
    max_idx = playerFrm.index[-1]

    pt_scenes = []
    prevScn = Scene.make_empty()
    prevPt = None
    for idx, coordSer in playerFrm.iterrows():
        hat_len = idx / max_idx * max_hat_len

        x, y = coordSer
        plPt = Point3D((x, y, 0))
        
        # plCrc = Circle3D.from_point3D(plPt, radius = 1).recolor(const.BGR['intense_red'])
        # prevScn = prevScn.dim(dim_factor).put(plCrc)
        # pt_scenes.append(prevScn)

        if prevPt is not None:
            myLn = Line3D(prevPt, plPt).recolor(color)

            if myLn.length() > 0:
                myArr = Arrow.make(myLn.leftPt, myLn.rightPt, hat_length = hat_len, rot_angle = 0, color = color)
            else:
                myArr = myLn ### ugly but doesn't matter

            if not prevScn.is_empty():
                prevScn = prevScn.drop(-1) ### drop the arrow
            
            prevScn = prevScn.transparify(cumulative_transparence).put(myLn).put(myArr)

        pt_scenes.append(prevScn)
        prevPt = plPt

    # View.make(myCam, pt_scenes[-1]).snap().show()

    pt_scenes = [*[Scene([]) for _ in myPnt[0].get_frames()[:-1]], 
                 *pt_scenes,
                 *[pt_scenes[-1] for _ in myPnt[2].get_frames()[:-1-next_frames]]]

    return pt_scenes

i = 1
for myPnt in candPnts:
    if i == 6:
        i = 7 ### skip lower left

    ### splines
    splnScns = get_example_spline_scenes(myPnt)
    ### other shapes
    othrScns = get_return_motion(myPnt, color = const.BGR['intense_red'], cumulative_transparence = 0.95, max_hat_len = 20, which = 'rec')
    ### combine
    jointScns = [splnScn.combine(othrScn) for splnScn, othrScn in zip(splnScns, othrScns)]
    ### render    
    realFrms = myPnt[:3].stream(confs = confs).decap().frames
    myCam = Camera.calibrated_brest_behind()
    exFrms, _ = vs.animate_sequence(jointScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False,
                                    animate_over = realFrms, 
                                    background_resolution = 1, argument_resolution = 1, 
                                    foreground_resolution = 0.5, argument_luminosity = 1)
    
    exFrms = exFrms.framed(FRAME_THICKNESS, FRAME_COLOR)

    ### freeze raster
    exRst = exRst.to_frame_raster().to_frames_raster(num_frames = 1)
    
    ### add new element
    row, col = i // 3, i % 3 

    ### speed up
    if row == 1:
        exFrms = exFrms[::2]
    if row == 2:
        exFrms = exFrms[::3]

    exRst.add(exFrms, row, col)

    exFrms = exRst.render()

    # if i < 6:
    #     exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on the impact of\nthe run-around serve+1")
    exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on the impact of\nthe run-around serve+1")

    rast_frames.append(exFrms)

    i += 1

exFrms_last = BGRFrames.from_frame(exFrms[-1], 25)
rast_frames.append(exFrms_last)

exRst = exRst.to_frame_raster().to_frames_raster(num_frames = 25)
exFrms_last_no_fn = exRst.render()
rast_frames.append(exFrms_last_no_fn)

rstFrms = BGRFrames.from_frames(rast_frames)
# DecapStream(rstFrms, 0, 100).show()

all_frames.append(rstFrms)
script_lines.append(script_line)
# del exFrms, bulletFrms1a, rstFrms

from ATMANpy.types.Overlays import reverse_cascade
acc, frms = 1.25, 10

script_line = ""
revRst, revFrms = reverse_cascade(exRst, keep_which = [(0, 0), (2, 0)], acceleration = acc, frames_per_element = frms)

all_frames.append(revFrms)
script_lines.append(script_line)

# b) time to reposition
def hi_launch(myPnt, threshold = 14):
    return myPnt.get_return().get_launch_angle() > threshold

def serve_ad(myPnt):
    return myPnt.is_serve_ad()

ad_dtlPnts1 = chipPnts.filter('is_serve_ad').filter(shot_is_where, shot_idx = 1, areas = [1])
dc_dtlPnts1 = chipPnts.filter('is_serve_deuce').filter(shot_is_where, shot_idx = 1, areas = [4])

# play(ad_dtlPnts1)
# play(dc_dtlPnts1)

ad_cands_b = [0, 4]
dc_cands_b = [5, 6, 8]

candPnts = [ad_dtlPnts1[idx] for idx in ad_cands_b]
# play(candPnts)

candPnts = [dc_dtlPnts1[idx] for idx in dc_cands_b]
# play(candPnts)

ad_dtlPnts2 = chipPnts.filter('is_serve_ad').filter(shot_is_where, shot_idx = 1, areas = [2])
dc_dtlPnts2 = chipPnts.filter('is_serve_deuce').filter(shot_is_where, shot_idx = 1, areas = [3])

ad_cands_b = [4, 5, 11, 13, 15]
dc_cands_b = [6, 12, 24, 25, 32]

candPnts = [ad_dtlPnts2[idx] for idx in ad_cands_b]
# play(candPnts)

candPnts = [dc_dtlPnts2[idx] for idx in dc_cands_b]
# play(candPnts)

# play(ad_dtlPnts2)
# play(dc_dtlPnts2)

dc_cands_b1 = [5, 8]
ad_cands_b1 = [0]

dc_cands_b2 = [24, 32]
ad_cands_b2 = [5, 13]

candPnts = [*[dc_dtlPnts1[idx] for idx in dc_cands_b1], *[ad_dtlPnts1[idx] for idx in ad_cands_b1], 
            *[dc_dtlPnts2[idx] for idx in dc_cands_b2], *[ad_dtlPnts2[idx] for idx in ad_cands_b2]]

script_line =   """
                    On the plus side, longer travel time means more time for the returner to reposition,
                    making the chip return down the line more viable especially on serves out wide.
                    By contrast, the topspin return has to go cross-court on serves out wide, according to our model.
                """

rast_frames = []
row, col = 0, 0

exRst = FramesRaster(3, 3)
exRst.add(bulletFrms1b, col, row)
exFrms = exRst.render()
rast_frames.append(exFrms)

exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on the impact of\nreturn shot selection\non time to reposition.")
rast_frames.append(exFrms)

i = 1
for myPnt in candPnts:
    if i == 6:
        i = 7 ### skip lower left

    ### splines
    splnScns = get_example_spline_scenes(myPnt)
    ### other shapes
    othrScns = get_return_motion(myPnt, color = const.BGR['intense_red'], cumulative_transparence = 0.95, max_hat_len = 20, which = 'hit', next_frames = 7)
    ### combine
    jointScns = [splnScn.combine(othrScn) for splnScn, othrScn in zip(splnScns, othrScns)]
    ### render    
    realFrms = myPnt[:3].stream(confs = confs).decap().frames
    myCam = Camera.calibrated_brest_behind()
    exFrms, _ = vs.animate_sequence(jointScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False,
                                    animate_over = realFrms, 
                                    background_resolution = 1, argument_resolution = 2, 
                                    foreground_resolution = 0.5, argument_luminosity = 1)
    
    exFrms = exFrms.framed(FRAME_THICKNESS, FRAME_COLOR)

    ### freeze raster
    exRst = exRst.to_frame_raster().to_frames_raster(num_frames = 1)
    
    ### add new element
    row, col = i // 3, i % 3
    
    ### speed up    
    if row == 1:
        exFrms = exFrms[::2]
    if row == 2:
        exFrms = exFrms[::3]

    exRst.add(exFrms, row, col)

    exFrms = exRst.render()
    
    # if i < 6:
    #     exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on the impact of\nreturn shot selection\non time to reposition.")
    exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on the impact of\nreturn shot selection\non time to reposition.")

    rast_frames.append(exFrms)

    i += 1

exFrms_last = BGRFrames.from_frame(exFrms[-1], 25)
rast_frames.append(exFrms_last)

exRst = exRst.to_frame_raster().to_frames_raster(num_frames = 25)
exFrms_last_no_fn = exRst.render()
rast_frames.append(exFrms_last_no_fn)

rstFrms = BGRFrames.from_frames(rast_frames)
# DecapStream(rstFrms, 0, 100).show()

all_frames.append(rstFrms)
script_lines.append(script_line)
# del exFrms, rstFrms, bulletFrms1b

script_line = ""
revRst, revFrms = reverse_cascade(exRst, keep_which = [(0, 0), (2, 0)], acceleration = acc, frames_per_element = frms)

all_frames.append(revFrms)
script_lines.append(script_line)

# c) impatience; premature aggression

def third_is_fault(myPnt):
    try:
        nextShot = myPnt[3]
    except:
        nextShot = None
    return myPnt[2].is_fault(nextShot)

def third_is_in_net(myPnt):
    return myPnt[2].is_in_net()

ftPnts1 = chipPnts.filter(shot_is_where, shot_idx = 1, areas = [1, 2]).filter(third_is_fault)
ntPnts1 = chipPnts.filter(shot_is_where, shot_idx = 1, areas = [1, 2]).filter(third_is_in_net)

# play(ftPnts1)
# play(ntPnts1)

ftPnts2 = chipPnts.filter(shot_is_where, shot_idx = 1, areas = [3]).filter(third_is_fault)
ntPnts2 = chipPnts.filter(shot_is_where, shot_idx = 1, areas = [3]).filter(third_is_in_net)

# play(ftPnts2)
# play(ntPnts2)

ftPnts3 = chipPnts.filter(shot_is_where, shot_idx = 1, areas = [4]).filter(third_is_fault)
ntPnts3 = chipPnts.filter(shot_is_where, shot_idx = 1, areas = [4]).filter(third_is_in_net)

# play(ftPnts3)
# play(ntPnts3)

ft_cands_b1 = [5, 6]
nt_cands_b1 = [1]

ft_cands_b2 = [2, 3]
nt_cands_b2 = [0, 1]

ft_cands_b3 = []
nt_cands_b3 = []

candPnts = [*[ftPnts1[idx] for idx in ft_cands_b1], *[ntPnts1[idx] for idx in nt_cands_b1],
            *[ftPnts2[idx] for idx in ft_cands_b2], *[ntPnts2[idx] for idx in nt_cands_b2],
            *[ftPnts3[idx] for idx in ft_cands_b3], *[ntPnts3[idx] for idx in nt_cands_b3],]


script_line =   """
                    Finally, there is evidence for the server typically being somewhat too impatient on
                    chip return serves + 1. 
                """

rast_frames = []
row, col = 0, 0

exRst = FramesRaster(3, 3)
exRst.add(bulletFrms1c, col, row)
exFrms = exRst.render()
rast_frames.append(exFrms)

script_line =   """
                    The model believes that a chip return to the forehand can goad the
                    server into premature aggression, and entice an error by the server.
                """

exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on premature aggression\nagainst chip returns.")
rast_frames.append(exFrms)

i = 1
for myPnt in candPnts:
    if i == 6:
        i = 7

    ### splines
    splnScns = get_example_spline_scenes(myPnt, third_color = const.BGR['intense_red'])
    ### render    
    realFrms = myPnt[:3].stream(confs = confs).decap().frames
    myCam = Camera.calibrated_brest_behind()
    exFrms, _ = vs.animate_sequence(splnScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False,
                                    animate_over = realFrms, 
                                    background_resolution = 1, argument_resolution = 2, 
                                    foreground_resolution = 0.5, argument_luminosity = 1)
    
    exFrms = exFrms.framed(FRAME_THICKNESS, FRAME_COLOR)

    ### freeze raster
    exRst = exRst.to_frame_raster().to_frames_raster(num_frames = 1)
    
    ### add new element
    row, col = i // 3, i % 3

    ### speed up
    if row == 1:
        exFrms = exFrms[::2]
    if row == 2:
        exFrms = exFrms[::3]

    exRst.add(exFrms, row, col)

    exFrms = exRst.render()
    
    if i < 6:
        exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on premature aggression\nagainst chip returns.")
    exFrms = overlay_footnote(exFrms, "XX:XX-XX:XX:", "For more info on premature aggression\nagainst chip returns.")

    rast_frames.append(exFrms)

    i += 1

exFrms_last = BGRFrames.from_frame(exFrms[-1], 25)
rast_frames.append(exFrms_last)

exRst = exRst.to_frame_raster().to_frames_raster(num_frames = 25)
exFrms_last_no_fn = exRst.render()
rast_frames.append(exFrms_last_no_fn)

rstFrms = BGRFrames.from_frames(rast_frames)
# DecapStream(rstFrms, 0, 100).show()

all_frames.append(rstFrms)
script_lines.append(script_line)
# del exFrms, rstFrms, bulletFrms1c

script_line =   """
                    All these factors lead our model to like the forehand side chip return as a viable alternative to the topspin return. 
                    Naturally, against any specific player, the optimal return strategy may be very different, 
                    however, this analysis provides evidence that chipping the ball into play low, to the forehand, 
                    may be an underutilized strategy on aggregate. 
                """
revRst, revFrms = reverse_cascade(exRst, keep_which = [(2, 0)], acceleration = acc, frames_per_element = frms * 5)

all_frames.append(revFrms)
script_lines.append(script_line)

vis_ut.show(all_frames)

### Saving the separate MultiFrame objects and the Joint Script
if SAVE:
    assert len(script_lines) == len(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)
