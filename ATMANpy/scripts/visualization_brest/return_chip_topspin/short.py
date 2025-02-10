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
from ATMANpy.types.Shapes3D import Point3D, Line3D, Circle3D, Circles3D
from ATMANpy.types.ShapesTennis3D import Player
from ATMANpy.types.Overlays import FrameRaster, FramesRaster, Table, overlay, overlay_center, make_text_in

from ATMANpy.funcs.util_funcs import interpolate_tuples


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
a, b, c, d = Point2D((eps, eps)), Point2D((title_x - eps, eps)), Point2D((eps, title_y - eps)), Point2D((title_x - eps, title_y - eps))
titleReg = Region.from_points(a, b, c, d)

titleStr = 'On the Effectiveness of the Chipped Return'

titleFrm = BGRFrame.make_black(*title_dims).put(titleStr, size = 2, where = (50, round(0.575 * title_y)))
titleFrm.show()

titleFrm_framed = titleFrm.framed(FRAME_THICKNESS, FRAME_COLOR)
titleFrm_framed.show()

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

sampleTab = Table(table_data, **table_kwargs_small, null_columns = 0)
tabScn, annPnts = sampleTab.preview()
headerPt = annPnts[0]

headerFrm = View.make(sampleTab.tabCam, Scene([]), annotatedPoints = [headerPt]).snap()
headerFrm = headerFrm.put('THE DATA', color = const.BGR['bright_grey'], size = 2, thickness = 2)
headerFrm.show()

tabFrm = sampleTab.render()
tabFrm = tabFrm.put('THE DATA', color = const.BGR['bright_grey'], size = 2, thickness = 2)
tabFrm.show()

headerFrm_framed = headerFrm.framed(FRAME_THICKNESS, FRAME_COLOR)
headerFrm_framed.show()

tabFrm_framed = tabFrm.framed(FRAME_THICKNESS, FRAME_COLOR)
tabFrm_framed.show()

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
diagFrm_0.show()

diagFrm_0_framed = diagFrm_0.framed(FRAME_THICKNESS, FRAME_COLOR)
diagFrm_0_framed.show()

diagView = View.make(modDiag_1.tabCam, myScn_1, annotatedPoints = annPnts_shifted_1, resolution = modDiag_1.resolution)
diagFrm_1 = diagView.snap()
diagFrm_1 = diagFrm_1.put('THE MODEL', color = const.BGR['bright_grey'], size = 2, thickness = 2)
diagFrm_1.show()

diagFrm_1_framed = diagFrm_1.framed(FRAME_THICKNESS, FRAME_COLOR)
diagFrm_1_framed.show()

diagView = View.make(modDiag_2.tabCam, myScn_2, annotatedPoints = annPnts_shifted_2, resolution = modDiag_2.resolution)
diagFrm_2 = diagView.snap()
diagFrm_2 = diagFrm_2.put('THE MODEL', color = const.BGR['bright_grey'], size = 2, thickness = 2)
diagFrm_2.show()

diagFrm_2_framed = diagFrm_2.framed(FRAME_THICKNESS, FRAME_COLOR)
diagFrm_2_framed.show()

### RESULT TEXT ###
resFrm = BGRFrame.make_black(*DIMENSIONS_HALF)
resFrm = resFrm.put('THE RESULTS', color = const.BGR['bright_grey'], size = 2, thickness = 2)

y = 120
i = 30

text1 = "Chipped Returns are a (mostly) viable strategy,\nas long as they're played ..."

text1a = "   ... low ..."
text1b = "   ... to the server's forehand ..."
text1c = "   ... with sufficient depth and width."

text2 = "Chip returns are high-risk/high-reward, except ..."
text2a = "    - The typical chip return has a smaller target zone than the typical topspin return."
text2b = "    - The typical chip return has a higher peak returner win probability."

text3 = "... in one situation, where topspin is preferable."
text3a = "    Ad-side serve out wide from righty server."
text3b = "    Deuce-side serve out wide from lefty server."

resFrm_simple1 = resFrm.put(text1, where = (50, y), size = 2)
resFrm_simple1a = resFrm_simple1.put(text1a, where = (50, y+3*i), size = 2)
resFrm_simple1b = resFrm_simple1a.put(text1b, where = (50, y+5*i), size = 2)
resFrm_simple1c = resFrm_simple1b.put(text1c, where = (50, y+7*i), size = 2)
# resFrm_simple1c.show()

resFrm_simple2 = resFrm_simple1c.put(text2, where = (50, y+round(9.5*i)), size = 2)
# resFrm_simple2.show()

resFrm_simple3 = resFrm_simple2.put(text3, where = (50, y+12*i), size = 2)
# resFrm_simple3.show()

resFrm_simple1_framed = resFrm_simple1.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple1a_framed = resFrm_simple1a.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple1b_framed = resFrm_simple1b.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple1c_framed = resFrm_simple1c.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple2_framed = resFrm_simple2.framed(FRAME_THICKNESS, FRAME_COLOR)
resFrm_simple3_framed = resFrm_simple3.framed(FRAME_THICKNESS, FRAME_COLOR)

resFrm_simple3_framed.show()

### INITIAL RASTER ###

all_frames = []
script_lines = []
STILL_FRAMES = 10

script_line =   """
                    Though employed less frequently than the topspin return, the chipped return is
                    consistently used by highly successful players to neutralize the serve, gain time to reposition,
                    or draw the server into the court.
                    The primary goal of this analysis is 
                    to understand the effectiveness of the chip return as compared to the topspin return; 
                    to understand when a chip return could be attempted, 
                    and where it should be placed.
                """  ### 22 sec

blackFrm = BGRFrame.make_black(*DIMENSIONS)
black_with_titleFrm = overlay_center(blackFrm, titleFrm)
# black_with_titleFrm.show()
black_with_titleFrms = BGRFrames.from_frame(black_with_titleFrm, STILL_FRAMES)

all_frames.append(black_with_titleFrms)
script_lines.append(script_line)

script_line =   """
                    To this end, we estimate a statistical model...
                """ # 3 sec

startRst = FrameRaster(2, 2)

startRst.add(diagFrm_0_framed, 0, 0)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)
    
script_line =   """
                    ... that takes into account a great variety of relevant variables ...
                """ # 4 sec

startRst.add(diagFrm_1_framed, 0, 0)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   """
                    ... and spits out an estimated probability that the returner wins the point.
                """ # 5 sec

startRst.add(diagFrm_2_framed, 0, 0)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   """
                    The model is estimated based on 20 matches played in the course of the 2021 Brest Challenger tournament.
                """ # 6 sec

startRst.add(headerFrm_framed, 1, 0)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   """
                    In the course of this tournament, ATMAN has recognized 1.451 first serve returns, of which 1.044 were played with topspin, and 407 were chipped.
                """ # 12 sec

startRst.add(tabFrm_framed, 1, 0)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)

script_line =   """
                    We find that, according to the model, chip returns are a viable alternative to the topspin return, 
                    as long as they're played low ...
                """ # 8 sec

startRst.add(resFrm_simple1, 0, 1)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)


script_line =   """
                    ... and with sufficient depth and width ...
                """ # 3 sec

startRst.add(resFrm_simple1a, 0, 1)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)         

script_line =   """
                    ... to the server's forehand.
                """ # 3 sec

startRst.add(resFrm_simple1b, 0, 1)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)  

script_line =   """
                    The topspin return is always liked by our model; however, if these three guidelines are followed,
                    the chip return represents a high-risk/high-reward alternative to the topspin return...
                """ # 10 sec

startRst.add(resFrm_simple1c, 0, 1)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)  

script_line =   """
                    ... with one exception.       
                """ # 3 sec

startRst.add(resFrm_simple2, 0, 1)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)  

script_line =   """
                    In case of a serve out wide from the server's backhand side, that is, 
                    from the ad side for a right-handed server, and the deuce side for a left-handed server,
                    the model prefers the topspin return strictly over the chip return.          
                """ # 14 sec
    
startRst.add(resFrm_simple3, 0, 1)
rstFrm = startRst.render()
rstFrm = overlay_center(rstFrm, titleFrm_framed)
rstFrms = BGRFrames.from_frame(rstFrm, STILL_FRAMES)

all_frames.append(rstFrms)
script_lines.append(script_line)  

### -> total: 93 seconds of text.


##### UNUSED YET
script_line =   """
                    That a chip return should be played to the server's forehand is somewhat unintuitive.
                    However, this result is fairly consistent across different model specifications.
                    We will attempt to explain why the model likes the chip return to the forehand later;
                    for the moment
                """

script_line =   """
                    for the moment, we will issue the following caveat: 
                    these results hold across many different players; it might be that any specific player struggles
                    with the chip to the backhand side, or another player's serve should not be chipped back altogether.
                    Though such an individualized analysis is much more actionable and interesting, 
                    it would require much more data. 
                    However, if we had that data, the statistical tools used here would apply exactly the same.
                """

script_line =   """
                    Explanations for chip to forehand
                        - with chip it is not so important to play yourself into position -> longer airtime
                            (most extreme with high chip)
                        - impatience on chip return serve + 1
                        - run around backhands.
                """

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

###### DEUCE WIDE ######
side, where = 'deuce', 'wide'
i, j = 1, 1

# frames to render on
preFrms = wideDict[side + '_' + where]
realFrms = realDict[side + '_' + where]
realFrms_last = BGRFrames.from_frame(realFrms[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
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

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Before commenting on the results, let's look at the model's suggestions in greater depth.
                    We take a look at serves from the deuce and ad side; and serves down the middle and out wide separately.
                    In this first example, Alexandre Mueller is pulled out wide by Luca van Assche's serve from the deuce side.
                """  # 19 sec

dimRst.add(preFrms, i, j)
rastFrms = dimRst.render()

# fade out title frame
titleFrms = BGRFrames.from_frame(titleFrm, pre_frames)
titleFrms_framed = BGRFrames.from_frame(titleFrm_framed, pre_frames)

# overlay the title with fade out
rastFrms1 = overlay_center(rastFrms, titleFrms_framed, start_alpha = 1, stop_alpha = 0, alpha_exp = 4)
# DecapStream(rastFrms, 0, 100).show()

movRst.add(realFrms, i, j)
rastFrms2 = movRst.render()
# rastFrms[-1].show()

movRst.add(realFrms_last, i, j)
rastFrms3 = movRst.render()

joint_rastFrms = BGRFrames.from_frames([rastFrms1, rastFrms2, rastFrms3])

all_frames.append(joint_rastFrms)
script_lines.append(script_line)

script_line =   """
                    The orange area shows the target zone for the returning player on a topspin shot.
                    The target zone is where the model suggests to aim the return; if the ball
                    lands within the target zone, the returning player has a probability of winning the subsequent
                    point greater than 50%.

                    In this situation, our model believes it is best to hit the topspin return cross-court.
                """ # 22 sec

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
movRst.add(myFrms, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg1Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "" ### a little pause before the next segment
rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)
all_frames.append(rastFrms_static)
script_lines.append(script_line)


script_line =   """
                    As for a low chip return, the model likes the cross-court return again. 
                    This time, it may land more shallow in the court, but needs a more acute angle to remain above 50%.

                    Note that we differentiate between a low chip and a high chip. 
                    The low chip is close to a traditional slice, with low net clearance and low bounce.
                    On the other hand, ...
                """ # 21 sec

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
movRst.add(myFrms, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg2Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    ... the high chip is a floating return hit with underspin; it has a high arc and a high bounce.
                    We differentiate between these two kinds of chip return 
                    because our data shows that the low chip and the high chip return are distinct shots 
                    that are employed with purpose; 
                    for a closer look at their differences, and some examples, consult the longer version of this video.

                    In any case, the target zone for the high chip is considerably smaller than either the topspin return or the low chip.
                    The high chip needs much more depth than the low chip; it may not land as shallow in the court.
                """ # 34 sec

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
movRst.add(myFrms, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
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

postFrms1 = BGRFrames.from_frames([transFrms, moveFrms])

movRst.add(postFrms1, i, j)
rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    The flags indicate the highest returner win probability spot within the target zone,
                    for each of our three return types. 
                    Note that the returner has a greater predicted probability of winning the point when
                    hitting a well-placed chip, versus a well-placed topspin return.
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

postFrms2 = BGRFrames.from_frames([lineFrms, statFrms])
movRst.add(postFrms2, i, j)

rastFrms = movRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

prevRst = movRst.to_frame_raster().to_frames_raster(25)

###### DEUCE MIDDLE ######
side, where = 'deuce', 'middle'
i, j = 0, 1

# frames to render on
preFrms = wideDict[side + '_' + where]
realFrms = realDict[side + '_' + where]
realFrms_last = BGRFrames.from_frame(realFrms[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
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

    ### make zone max point and value
    x, y = max_coords
    maxPt = Point3D((x, y, 0))
    pvalDict[angle] = (maxPt, max_val)

script_line =   """
                    Consider now a serve DOWN THE MIDDLE from the server's forehand side.
                """ # 4 sec

prevRst.add(preFrms, i, j)
rastFrms1 = prevRst.render()

prevRst.add(realFrms, i, j)
rastFrms2 = prevRst.render()

prevRst.add(realFrms_last, i, j)
rastFrms3 = prevRst.render()

rastFrms = BGRFrames.from_frames([rastFrms1, rastFrms2, rastFrms3])
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    For the topspin return, the model indicates a very generous target zone, 
                    ranging from the server's feet to his backhand side.
                """ # 8 sec

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "" ### a little pause before the next segment

rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)
all_frames.append(rastFrms_static)
script_lines.append(script_line)

script_line = "On the other hand, the model still recommends to play the low chip cross-court to the forehand." # 5 sec

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "Again, the high chip has the smallest target zone, requiring depth and width on the server's forehand side." # 7 sec

# fade in hi chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "Let's zoom in." # 2 sec

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

postFrms = BGRFrames.from_frames([transFrms, moveFrms])

prevRst.add(postFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    Again, a well-placed low chip has a higher probability of winning the point than a topspin return;
                    However, the target zone for the low chip is smaller than the topspin return. 
                    In this sense, the chip return is a high-risk/high-reward strategy: 
                    As long as the chip has required depth and width, it promises high reward, 
                    but the probability of the returner winning the point quickly falls off around the peak.
                    On the other hand, the topspin return is more forgiving: 
                    The ball can land in a larger area and still at least equalize the rally.  

                    The high chip, on the other hand, having both the lowest peak and the smallest zone,
                    is not recommended by the model in this situation.
                """ # 38 sec

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

postFrms = BGRFrames.from_frames([lineFrms, statFrms])

prevRst.add(postFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

prevRst = prevRst.to_frame_raster().to_frames_raster(25)

###### AD MIDDLE ######
side, where = 'ad', 'middle'
i, j = 0, 0

# frames to render on
preFrms = wideDict[side + '_' + where]
realFrms = realDict[side + '_' + where]
realFrms_last = BGRFrames.from_frame(realFrms[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
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

prevRst.add(realFrms_last, i, j)
rastFrms3 = prevRst.render()

rastFrms = BGRFrames.from_frames([rastFrms1, rastFrms2, rastFrms3])
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    In this situation, the model likes a topspin return played anywhere from down the line to the server's feet. 
                    However, presumably because now the return lands on the server's FOREHAND side, the model requires greater depth
                    to bring the returner's chances to above 50%.
                """ # 17 sec

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "" ### a little pause before the next segment

rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)
all_frames.append(rastFrms_static)
script_lines.append(script_line)

script_line = "The recipe for a good low chip is bizarrely consistent: target the server's forehand with good width." # 7 sec

# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "No change also in the high-chip strategy; the target zone remains woefully small, deep on the forehand side." # 7 sec

# fade in hi chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "Let's take a look at peak returner win probabilities." # 4 sec

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

postFrms = BGRFrames.from_frames([transFrms, moveFrms])

prevRst.add(postFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    Again, the risk-reward tradeoff between low chip and topspin returns is apparent.
                    The topspin target zone is nice and large, but with a lower peak than the low chip zone.
                """ # 10 sec

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

postFrms = BGRFrames.from_frames([lineFrms, statFrms])

prevRst.add(postFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

prevRst = prevRst.to_frame_raster().to_frames_raster(25)

###### AD WIDE ######
side, where = 'ad', 'wide'
i, j = 1, 0

# frames to render on
preFrms = wideDict[side + '_' + where]
realFrms = realDict[side + '_' + where]
realFrms_last = BGRFrames.from_frame(realFrms[-1], STATIC_FRAMES) ### for the first case, we only need the static case.

# heatmap and zone dictionary
heatDict = {}
zoneDict = {}
pvalDict = {}
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

prevRst.add(realFrms_last, i, j)
rastFrms3 = prevRst.render()

rastFrms = BGRFrames.from_frames([rastFrms1, rastFrms2, rastFrms3])
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    Here, the model recommends playing the topspin return strictly cross-court. 
                    On the topspin return, the returner is afforded very little time to reposition. 
                    A wide serve pulls the returning player off the court, 
                    the model therefore sees it as imperative to send the ball cross-court 
                    to reduce the distance the returning player 
                    needs to cover to get back in position.
                """ # 22 sec

# dim and fade in target area
itplFrms = vs.interpolate_frames(realFrms[-1], realFrms[-1].dim(dim_factor), num_frames = TRANSITION_FRAMES)
itplScns = vs.interpolate_dummy_scenes(zoneDict['tp'].transparify(0), zoneDict['tp'], TRANSITION_FRAMES)

myFrms, _ = vs.animate_sequence(itplScns, start_cam = halfCam, stop_cam = halfCam, **prmDict_annot, animate_over = itplFrms)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)

all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "" ### a little pause before the next segment

rastFrms_static = BGRFrames.from_frame(rastFrms[-1], 50)
all_frames.append(rastFrms_static)
script_lines.append(script_line)
script_line =   """
                    As before, the model thinks it best to send the low chip return down the line, to the server's forehand.
                    However, the target zone is decidedly smaller than usual.
                """ # 10 sec
# fade in lo chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    In this isolated case, the high-chip target zone is actually larger than the low-chip zone:
                    though the returner would play himself out of position going down-the-line, the model takes into account
                    that the high-chip affords the returner with plenty of time to recover to almost anywhere along the baseline; 
                    As we show in the longer version of this video, this is especially useful on the down-the-line return on a serve out wide.
                """ # 25 sec

# fade in hi chip target area
zoneScn = zoneDict['tp'].combine(zoneDict['lo']).combine(zoneDict['hi']).consolidate()
myFrms, _ = vs.animate_static(zoneScn, static_cam = halfCam, **prmDict_annot, animate_over = itplFrms[-1], num_frames = 50)
prevRst.add(myFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line = "Let's zoom in again." # 2 sec

fullScn = courtScn.combine(sceneDict[side + '_' + where])
backFrm_with_shapes = View.make(halfCam, fullScn).snap()

### storing the movement frames in a separate raster.
itplFrms = vs.interpolate_frames(realFrms[-1].dim(dim_factor), backFrm_with_shapes, num_frames = STATIC_FRAMES)
zoneScns = [zoneScn for _ in range(STATIC_FRAMES)]

transFrms, _ = vs.animate_sequence(zoneScns, start_cam = halfCam, stop_cam = halfCam, 
                                    **prmDict_annot, animate_over = itplFrms)
moveFrms, _ = vs.animate_cam_motion(zoneScn, start_cam = halfCam, stop_cam = heatCam, background_scene = fullScn, 
                                    num_frames = TRANSITION_FRAMES, **prmDict_annot_heat)

postFrms = BGRFrames.from_frames([transFrms, moveFrms])

prevRst.add(postFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

script_line =   """
                    In this situation alone, the topspin return is strictly preferred by the model:
                    The chip does buy the returner more time to reposition on court, but on this serve out wide,
                    this is not enough to offset the advantage of a simple topspin return cross-court, 
                    to the backhand of the server.
                """ # 17 sec

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

postFrms = BGRFrames.from_frames([lineFrms, statFrms])

prevRst.add(postFrms, i, j)
rastFrms = prevRst.render()
rastFrms = overlay(rastFrms, leg3Frm, top_left_point = tlPt)
all_frames.append(rastFrms)
script_lines.append(script_line)

prevRst = prevRst.to_frame_raster().to_frames_raster(25)

### total section seconds: 314
### total so far: 314 + 97 = 407

vis_ut.show(all_frames)

### INTERPRETATION ###





### First Example ###
scale_factor = 1.5

retPnts_in

retShots = retPnts_in.get_returns()
retShots_rectified = vis_ut.rectify_shots(retShots)
chip_retShots_rectified = retShots_rectified.filter('is_slice_visual')
tops_retShots_rectified = retShots_rectified.filter('is_slice_visual', negate = True)

dirFrm = regFrm[(regFrm['is_top_to_bot'])]
low_pred_exFrm = dirFrm[dirFrm.is_chipped].sort_values('pred_is_chipped').iloc[:8]
idx = int(low_pred_exFrm.index[0])
myShots = retPnts[idx][:2]

### Flight Splines
serveSF = myShots[0].feat_spline_cumul(which_endpoint = 'penultimate')
returnSF = myShots[1].feat_spline_cumul(which_endpoint = 'penultimate')

serve_scenes = [Scene([spLn.recolor(BALL_COLOR).resize(1.5) for spLn in splines if spLn is not None]) for splines in serveSF.values()]
return_scenes = [Scene([spLn.recolor(BALL_COLOR).resize(1.5) for spLn in splines if spLn is not None]) for splines in returnSF.values()]
spline_scenes = [*serve_scenes, *return_scenes]
splnScn = spline_scenes[-1]
bPt = splnScn[0].leftPt

# get topspin and chip dists, with axes
kwargDict = {'bandwidth': 2,
             'resolution': 1000,
             'axis_length': 40,
             'axis_origin_point': bPt,
             'x_unit_vector': Point3D((0, -1, 0)), 
             'y_unit_vector': Point3D((0, 0, 1)),
             'normalize_density': False, 
             'stretch_density': 4.5}

weightArr = regFrm_in.loc[regFrm_in.is_chipped == 1, 'weight_topspin'].values
weightArr_preds = 1 - regFrm_in.loc[regFrm_in.is_chipped == 1, 'pred_is_chipped'].values

tops_angleScn, tops_modes = vs.kde_angle_distribution(tops_retShots_rectified, **kwargDict, color = TOPS_COLOR)
chip_angleScn, chip_modes = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict, weight_func_or_arr = weightArr_preds, color = CHIP_COLOR)

chip_mode_lo, chip_mode_hi = chip_modes[[4, 10]]

### find similar shot of original, with higher launch angle V2 -> trajectory metrics
devArr_ang = chip_retShots_rectified.get_launch_angle() - chip_mode_hi
devArr_spd = chip_retShots_rectified.get_initial_speed() - myShots[1].get_initial_speed()
devArr_hgt = chip_retShots_rectified.get_ball_hit_height() - myShots[1].get_ball_hit_height()
devArr_spn = chip_retShots_rectified.get_initial_spin() - myShots[1].get_initial_spin()

std_ang = (devArr_ang ** 2).mean() ** 0.5
std_spd = (devArr_spd ** 2).mean() ** 0.5
std_hgt = (devArr_hgt ** 2).mean() ** 0.5
std_spn = (devArr_spn ** 2).mean() ** 0.5

dev_metric = np.abs(devArr_ang / std_ang) + np.abs(devArr_spd / std_spd) + np.abs(devArr_hgt / std_hgt) + np.abs(devArr_spn / std_spn)
dev_metric = dev_metric / 4 ### average standard deviations away from original shot

small_to_large = np.argsort(dev_metric)
best_idx = int(small_to_large[3]) #int(np.argmin(dev_metric)) ### not necessarily the smallest since we want a top to bot point!
dev_metric[best_idx]

bestShot = chip_retShots_rectified[best_idx]
mtc_idx, pnt_idx = bestShot.get_match_index(), bestShot.get_point_index()

bestPnt = retPnts.get(mtc_idx, pnt_idx)
bestShots = bestPnt[:2]
bestStm = bestShots.stream(confs = confs)

frames = bestStm.frmNums
new_frames = list(range(min(frames) - STATIC_FRAMES, min(frames)))
early_bestStm = bestStm.conform(new_frames)
early_bestDtm = early_bestStm.decap()
early_best_scaledFrms = early_bestDtm.frames.rescale(scale_factor)

early_bestStm.show()

