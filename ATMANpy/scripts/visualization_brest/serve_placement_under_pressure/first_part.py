
import os
import numpy as np
import pandas as pd
        
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame, MaskFrame
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Visual3D import Camera, View, Scene, DummyScene
from ATMANpy.types.Overlays import FramesRaster, overlay_center, overlay
from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D, Point3D

from ATMANpy.types.Abstract import NullPoint, Points, Shots, read_in_data_tables, read_in_placeholder_tables, read_in_colinear_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

########################################
########### SAVE FOLDER ################
########################################

SAVE = True

DESCRIPTOR = 'first_part'
SAVE_FOLDER = 'serve_pressure_new'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())

if SAVE:
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
cap = confs[0]['cap']


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

### TITLE CARD ###
from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.Lines import Region

title_dims = (850, 100)
eps = 2
title_x, title_y = title_dims

titleStr = 'On the Impact of Leverage on Serve Selection'

titleFrm = BGRFrame.make_black(*title_dims).put(titleStr, size = 2, where = (30, round(0.575 * title_y)))
# titleFrm.show()

blackFrm = BGRFrame.make_black()
black_with_titleFrm = overlay_center(blackFrm, titleFrm)
# black_with_titleFrm.show()

### TODO: maybe a little quote 
### TODO: maybe emphasize contrast between Playing every point like your life depends on it v. only point that matters is the last one.

all_frames = []
script_lines = []

script_line =   """
                    In tennis, not all points are of equal importance. 
                    Facetiously speaking, the only point that really matters is the last one. 
                    Though this might be a convenient oversimplification, 
                    there is more at stake in a close final set tiebreak than 2 breaks ahead in the first.

                    We quantify what is at stake by a notion of leverage: to what extent can a point swing the outcome of a match.

                    In this analysis, we want to find out whether leverage affects a player's serve selection.
                """

fadeFrms = vs.interpolate_frames(blackFrm, black_with_titleFrm, 100)
statFrms = BGRFrames.from_frame(black_with_titleFrm, 100)
currFrms = BGRFrames.from_frames([fadeFrms, statFrms])

all_frames.append(currFrms)
script_lines.append(script_line)

# script_line =   """
#                     The hypothesis is that at high leverage, a player will want to employ the serve that maximizes his or her chances of winning;
#                     at low leverage, a player might be more willing to go for sub-optimal serves in order to remain unpredictable, 
#                     and hopefully keep the receiver guessing even in high-leverage situations.
#                 """

# script_line =   """
#                     Before testing the hypothesis, let's take a quick look at the construction of the leverage metric.
#                 """

##############################
########## Raster ############
##############################

def maybe_get_pressure(somePnt):
    try:
        press = somePnt.get_total_match_score_pressure()
    except:
        press = None
    return press

pressures = dataPnts.apply(maybe_get_pressure)

### find a high pressure point ###
max_press = pressures[~np.isnan(pressures)].max()
max_idxs = pressures == max_press
idxs = np.arange(len(dataPnts))
print(idxs[max_idxs])

### 
dataPnts[1131].get_game_score()
dataPnts[1131].get_point_score()

hi_pressure_idx = 1131
hiPnt = dataPnts[hi_pressure_idx]
# hiPnt.stream(confs = confs).show()

### find a low pressure point from the same side by the same player in the same match
mtc_idx = hiPnt.get_match_index()
mtcPnts = dataPnts.get(mtc_idx)

def sousa_serves_ad(somePnt):
    is_sousa = somePnt.get_server().get_name() == 'Joao Sousa'
    is_ad = somePnt.is_serve_ad()
    return is_sousa and is_ad

filtPnts = mtcPnts.filter(sousa_serves_ad)

pressures_in_mtc_filt = filtPnts.apply(maybe_get_pressure)

which_lowest_pressure = 2
min_press = np.sort(pressures_in_mtc_filt)[which_lowest_pressure]

min_idxs = pressures_in_mtc_filt == min_press
mtc_idxs = np.arange(len(filtPnts))
min_idx = int(mtc_idxs[min_idxs][0])

loPnt = filtPnts[min_idx]
# loPnt.stream(confs = confs).show()

### initial frame raster ###
widen_by = 300
fade_by = 100
keep_for_end = 150
hiStm = hiPnt.stream(confs = confs).widen(widen_by)
loStm = loPnt.stream(confs = confs).widen(widen_by)

pre_hiFrms_all_unframed = hiStm.decap().frames[:widen_by].rescale(3/2)
pre_loFrms_all_unframed = loStm.decap().frames[:widen_by].rescale(3/2)

pre_hiFrms_all = pre_hiFrms_all_unframed.framed(thickness = 3, color = const.BGR['bright_grey'])
pre_loFrms_all = pre_loFrms_all_unframed.framed(thickness = 3, color = const.BGR['bright_grey'])

pre_hiFrms_fade = pre_hiFrms_all[:fade_by]
pre_loFrms_fade = pre_loFrms_all[:fade_by]
pre_hiFrms = pre_hiFrms_all[fade_by:widen_by-keep_for_end]
pre_loFrms = pre_loFrms_all[fade_by:widen_by-keep_for_end]
pre_hiFrms_post = pre_hiFrms_all[-keep_for_end:]
pre_loFrms_post = pre_loFrms_all[-keep_for_end:]

last_frame_unframed = pre_hiFrms_all_unframed[-keep_for_end]
last_frames_unframed = pre_hiFrms_all_unframed[-keep_for_end:]

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Arrow

### high leverage text frame
arrPt00 = Point2D((250, 235))
arrPt01 = Point2D((50, 235))
myArr0 = Arrow.make(arrPt00, arrPt01, hat_length = 20)

textFrm_hi = BGRFrame.make_black(cols = 960, rows = 540)
textFrm_hi_0 = textFrm_hi.put('High Leverage Point', color = const.BGR['intense_red'], 
                            size = 3, thickness = 2,
                            where = (300, 250))
textFrm_hi_1 = textFrm_hi_0.put(myArr0, color = const.BGR['intense_red'], thickness = 2)
textFrm_hi_2 = textFrm_hi_1.put('Server is well advised to employ\na serve that maximizes\nwin probability.', color = const.BGR['bright_grey'], 
                            size = 2, thickness = 1,
                            where = (300, 350))
# textFrm_hi_2.show()

### lo leverage text frame
arrPt10 = Point2D((650, 235))
arrPt11 = Point2D((910, 235))
myArr1 = Arrow.make(arrPt10, arrPt11, hat_length = 20)

textFrm_lo = BGRFrame.make_black(cols = 960, rows = 540)
textFrm_lo_0 = textFrm_lo.put('Low Leverage Point', color = const.BGR['leafy_green'], 
                            size = 3, thickness = 2,
                            where = (100, 250))
textFrm_lo_1 = textFrm_lo_0.put(myArr1, color = const.BGR['leafy_green'], thickness = 2)
textFrm_lo_2 = textFrm_lo_1.put('Server might mix in a serve that is not optimal\nto retain unpredictability.', color = const.BGR['bright_grey'], 
                            size = 2, thickness = 1,
                            where = (100, 350))
# textFrm_lo_2.show()

##########
myRst = FramesRaster(2, 2, borders = None)

myRst.add(pre_hiFrms_fade, 0, 0)
rstFrms_fade = myRst.render()
rstFrms_fade = rstFrms_fade.dim_progressively(0, 1) + BGRFrames.from_frame(black_with_titleFrm, fade_by).dim_progressively(1, 0)
myRst = myRst.to_frame_raster().to_frames_raster()

myRst.add(pre_hiFrms, 0, 0)
rstFrms0 = myRst.render()
# DecapStream(rstFrms0, 0, 100).show()

myRst = myRst.to_frame_raster().to_frames_raster()
myRst.add(BGRFrames.from_frame(textFrm_hi_1, 50), 0, 1)
rstFrms1 = myRst.render()
# DecapStream(rstFrms1, 0, 100).show()

myRst.add(BGRFrames.from_frame(textFrm_hi_2, 50), 0, 1)
rstFrms2 = myRst.render()
# DecapStream(rstFrms2, 0, 100).show()

myRst.add(pre_loFrms, 1, 1)
rstFrms3 = myRst.render()
# DecapStream(rstFrms3, 0, 100).show()

myRst = myRst.to_frame_raster().to_frames_raster()
myRst.add(BGRFrames.from_frame(textFrm_lo_1, 50), 1, 0)
rstFrms4 = myRst.render()
# DecapStream(rstFrms4, 0, 100).show()

myRst.add(BGRFrames.from_frame(textFrm_lo_2, 50), 1, 0)
rstFrms5 = myRst.render()
# DecapStream(rstFrms5, 0, 100).show()

rst_frames = [rstFrms_fade, rstFrms0, rstFrms1, rstFrms2, rstFrms3, rstFrms4, rstFrms5]
vis_ut.show(rst_frames)

###### ANIMATE #####

script_line =   """
                    Consider this match point on Joao Sousa's serve.
                """
currFrms = BGRFrames.from_frames([rstFrms_fade, rstFrms0])
all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    The hypothesis is that at such high leverage, 
                    a player will want to employ the serve that maximizes his or her chances of winning.
                """
currFrms = BGRFrames.from_frames([rstFrms1, rstFrms2])
all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    In a lower leverage situation, such as here, at 40-Love, a player might be more willing to go for sub-optimal serves in order to remain unpredictable, 
                    and hopefully keep the receiver guessing even in high-leverage situations.
                """
currFrms = BGRFrames.from_frames([rstFrms3, rstFrms4, rstFrms5])
all_frames.append(currFrms)
script_lines.append(script_line)

script_line = ""
holdFrms = BGRFrames.from_frame(currFrms[-1], 25)
all_frames.append(holdFrms)
script_lines.append(script_line)

script_line =   """
                    Before testing the hypothesis, let's take a quick look at the construction of the leverage metric, 
                    using as an example the high-leverage situation.
                """

# vis_ut.show(rst_frames)

### crop the score and fade out screen
from ATMANpy.types.Lines import Region, Line
x1, x2 = 20, 300
y1, y2 = 460, 525
p1, p2 = Point2D((x1, y1)), Point2D((x2, y2))
cropReg = Region.from_points(p1, p2)

# other crop score region (for later)
x1, x2 = x1 + 1920 // 2, x2 + 1920 // 2 - 30
y1, y2 = y1 + 1080 // 2, y2 + 1080 // 2
p1_2, p2_2 = Point2D((x1, y1)), Point2D((x2, y2))
cropReg_2 = Region.from_points(p1_2, p2_2)

lsFrm = rstFrms5[-1]
# lsFrm.put(cropReg).show()
# lsFrm.put(cropReg_2).show()
cropFrm = lsFrm.crop(cropReg)

from ATMANpy.types.Overlays import scoreboard, overlay, Table
lsFrms_fade = BGRFrames.from_frame(lsFrm, 50).dim_progressively()
lsFrms_fade = overlay(lsFrms_fade, cropFrm, top_left_point = p1)
# DecapStream(lsFrms_fade, 0, 100).show()

name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 5
point_score2 = 6
game_string1 = '376'
game_string2 = '666'

scoreFrm = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 2,
                      frame_thickness = 2, frame_color = const.BGR['bright_grey'])
# scoreFrm.show()

fade_frms = 50
fadeFrms = vs.interpolate_frames(cropFrm, scoreFrm, fade_frms)

start_point = p1
end_point = Point2D((200, 400))

lmbds = np.linspace(0, 1, fade_frms)

blackFrm = BGRFrame.make_black()
moving_frames = []
for idx in range(fade_frms):
    fadeFrm = fadeFrms[idx]
    lmbd = lmbds[idx]

    itplPt = start_point * (1 - lmbd) + end_point * lmbd
    itplPt = itplPt.round_point()

    move_fadeFrm = overlay(blackFrm, fadeFrm, top_left_point = itplPt)
    moving_frames.append(move_fadeFrm)

moveFrms = BGRFrames.from_single_frames(moving_frames)
# DecapStream(moveFrms, 0, 100).show()

fade_and_moveFrms = BGRFrames.from_frames([lsFrms_fade, moveFrms])
all_frames.append(fade_and_moveFrms)
script_lines.append(script_line)

###### MARKOV DIAGRAM ######
### 1st Order ###
name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 5
point_score2 = 7
game_string1 = '376'
game_string2 = '667'

scoreFrm_win = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 2,
                          frame_thickness = 2, frame_color = const.BGR['leafy_green'])
# scoreFrm_win.show()

name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 6
point_score2 = 6
game_string1 = '376'
game_string2 = '666'

scoreFrm_lose = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 2,
                           frame_thickness = 2, frame_color = const.BGR['bright_grey'])
# scoreFrm_lose.show()

dist_x = 400
dist_y = 100

tlPt_init = end_point
tlPt_win = tlPt_init.right(dist_x).up(dist_y)
tlPt_lose = tlPt_init.right(dist_x).down(dist_y)

### 2nd Order ###
tlPt_losewin = tlPt_lose.right(dist_x).up(dist_y)
tlPt_loselose = tlPt_lose.right(dist_x).down(dist_y)

name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 6
point_score2 = 7
game_string1 = '376'
game_string2 = '666'

scoreFrm_losewin = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 1,
                          frame_thickness = 2, frame_color = const.BGR['bright_grey'])
# scoreFrm_losewin.show()

name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 7
point_score2 = 6
game_string1 = '376'
game_string2 = '666'

scoreFrm_loselose = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 1,
                           frame_thickness = 2, frame_color = const.BGR['bright_grey'])
# scoreFrm_loselose.show()

### 3rd Order ###
tlPt_losewinwin = tlPt_losewin.right(dist_x).up(dist_y)
tlPt_losewinlose = tlPt_losewin.right(dist_x).down(dist_y)
tlPt_loseloselose = tlPt_loselose.right(dist_x).down(dist_y)

name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 6
point_score2 = 8
game_string1 = '376'
game_string2 = '667'

scoreFrm_losewinwin = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 1,
                          frame_thickness = 2, frame_color = const.BGR['leafy_green'])
# scoreFrm_losewin.show()

name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 7
point_score2 = 7
game_string1 = '376'
game_string2 = '666'

scoreFrm_losewinlose = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 1,
                           frame_thickness = 2, frame_color = const.BGR['bright_grey'])
# scoreFrm_loselose.show()

name1 = 'F.GAIO'
name2 = 'J.SOUSA'
point_score1 = 8
point_score2 = 6
game_string1 = '377'
game_string2 = '666'

scoreFrm_loseloselose = scoreboard(name1, name2, point_score1, point_score2, game_string1, game_string2, serving = 1,
                           frame_thickness = 2, frame_color = const.BGR['intense_red'])
# scoreFrm_loselose.show()


def make_arrow_from_to(tlPt_start, tlPt_end, dims = (280, 65), num_dashes = None):
    dx, dy = dims
    right_startPt = tlPt_start.right(dx).down(dy/2)
    left_endPt = tlPt_end.down(dy/2)
    return Arrow.make(right_startPt, left_endPt, shorten_by = 0.2, num_dashes = num_dashes)

def make_arrow_from_to_backwards(tlPt_start, tlPt_end, dims = (280, 65), num_dashes = None):
    dx, dy = dims
    left_startPt = tlPt_start.down(dy/2)
    right_endPt = tlPt_end.right(dx).down(dy/2)
    return Arrow.make(left_startPt, right_endPt, shorten_by = 0.1, num_dashes = num_dashes)

arr_win = make_arrow_from_to(tlPt_init, tlPt_win)
arr_lose = make_arrow_from_to(tlPt_init, tlPt_lose)

arr_losewin = make_arrow_from_to(tlPt_lose, tlPt_losewin)
arr_loselose = make_arrow_from_to(tlPt_lose, tlPt_loselose)

arr_losewinwin = make_arrow_from_to(tlPt_losewin, tlPt_losewinwin)
arr_losewinlose = make_arrow_from_to(tlPt_losewin, tlPt_losewinlose)
arr_loselosewin = make_arrow_from_to(tlPt_loselose, tlPt_losewinlose)
arr_loseloselose = make_arrow_from_to(tlPt_loselose, tlPt_loseloselose)

arr_color_win = const.BGR['grass_green']
arr_color_lose = const.BGR['sign_orange']

from ATMANpy.funcs.util_funcs import interpolate_tuples
dim_factor = 0.25
arr_color_win_dim = interpolate_tuples((0, 0, 0), arr_color_win, itpl_lambda = dim_factor)
arr_color_lose_dim = interpolate_tuples((0, 0, 0), arr_color_lose, itpl_lambda = dim_factor)

### win percentage boxes
def make_win_perc_box(win_percent = 0.5, back_color = const.BGR['bright_grey']):
    blackFrm = BGRFrame.make_black(280, 20)
    colorFrm = blackFrm + back_color
    if isinstance(win_percent, float):
        textStr = f'Win Probability: {win_percent:.0%}'
    else:
        textStr = f'Win Probability: {win_percent}'
    return colorFrm.put(textStr, color = const.BGR['darker_grey'], where = (100, 15))

percFrm_win = make_win_perc_box(1.0, back_color = const.BGR['leafy_green'])
percFrm_lose_known = make_win_perc_box(0.5)
percFrm_lose_unknown = make_win_perc_box('?')

scoreFrm_win_with_box = BGRFrame(np.concatenate([scoreFrm_win.array, percFrm_win.array], axis = 0))
# scoreFrm_lose_known_with_box = BGRFrame(np.concatenate([scoreFrm_lose.array, percFrm_lose_known.array], axis = 0))
# scoreFrm_lose_unknown_with_box = BGRFrame(np.concatenate([scoreFrm_lose.array, percFrm_lose_unknown.array], axis = 0))
scoreFrm_lose_known_with_box = BGRFrame(np.concatenate([percFrm_lose_known.array, scoreFrm_lose.array], axis = 0))
scoreFrm_lose_unknown_with_box = BGRFrame(np.concatenate([percFrm_lose_unknown.array, scoreFrm_lose.array], axis = 0))

######### WRITE SCRIPT DRAFT ########
curr_scoreFrm = moveFrms[-1]

script_line = "The leverage of a point should quantify to what extent the outcome of the point influences the outcome of the match."
currFrms = BGRFrames.from_frame(curr_scoreFrm, 50)

all_frames.append(currFrms)
script_lines.append(script_line)

# script_line = "The outcome of a point is among two possibilities: Joao Sousa can win the point, or he can lose it."
script_line = "Let's go through the possibilities."

curr_scoreFrm0 = overlay(curr_scoreFrm, scoreFrm_win, top_left_point = tlPt_win)
curr_scoreFrm0 = curr_scoreFrm0.put(arr_win, color = arr_color_win)
currFrms0 = BGRFrames.from_frame(curr_scoreFrm0, 40)

curr_scoreFrm1 = overlay(curr_scoreFrm0, scoreFrm_lose, top_left_point = tlPt_lose)
curr_scoreFrm1 = curr_scoreFrm1.put(arr_lose, color = arr_color_lose)
currFrms1 = BGRFrames.from_frame(curr_scoreFrm1, 40)

currFrms = BGRFrames.from_frames([currFrms0, currFrms1])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "If Joao Sousa wins this point, the outcome of the match is decided in his favor."
### brighten win side, dim lose side

curr_scoreFrm = overlay(curr_scoreFrm, scoreFrm_win, top_left_point = tlPt_win)
curr_scoreFrm = curr_scoreFrm.put(arr_win, color = arr_color_win)

curr_scoreFrm = overlay(curr_scoreFrm, scoreFrm_lose.dim(dim_factor), top_left_point = tlPt_lose)
curr_scoreFrm = curr_scoreFrm.put(arr_lose, color = arr_color_lose_dim)
# curr_scoreFrm.show()

fadeFrms = vs.interpolate_frames(currFrms[-1], curr_scoreFrm, 25)
statFrms = BGRFrames.from_frame(curr_scoreFrm, 40)

curr_scoreFrm = overlay(curr_scoreFrm, scoreFrm_win_with_box, top_left_point = tlPt_win)

fadeFrms2 = vs.interpolate_frames(statFrms[-1], curr_scoreFrm, 25)
statFrms2 = BGRFrames.from_frame(curr_scoreFrm, 40)

currFrms = BGRFrames.from_frames([fadeFrms, statFrms, fadeFrms2, statFrms2])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    If Joao Sousa loses this point, the outcome of the match is uncertain: it's now 6-all in the deciding tiebreak.

                    What we do in this case is try to figure out the probability of Joao Sousa winning the match.
                """
### brighten lose side, dim win side

curr_scoreFrm = overlay(curr_scoreFrm, scoreFrm_win_with_box.dim(dim_factor), top_left_point = tlPt_win)
curr_scoreFrm = curr_scoreFrm.put(arr_win, color = arr_color_win_dim)

curr_scoreFrm = overlay(curr_scoreFrm, scoreFrm_lose, top_left_point = tlPt_lose)
curr_scoreFrm = curr_scoreFrm.put(arr_lose, color = arr_color_lose)
# curr_scoreFrm.show()

fadeFrms = vs.interpolate_frames(currFrms[-1], curr_scoreFrm, 50)
statFrms = BGRFrames.from_frame(curr_scoreFrm, 50)

curr_scoreFrm = overlay(curr_scoreFrm, scoreFrm_lose_unknown_with_box, top_left_point = tlPt_lose.up(20))

fadeFrms2 = vs.interpolate_frames(statFrms[-1], curr_scoreFrm)
statFrms2 = BGRFrames.from_frame(curr_scoreFrm, 50)

currFrms = BGRFrames.from_frames([fadeFrms, statFrms, fadeFrms2, statFrms2])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "To this end, we follow the path of possible point outcomes a bit further."
### build out possible outcomes

curr_scoreFrm0 = overlay(curr_scoreFrm, scoreFrm_losewin, top_left_point = tlPt_losewin)
curr_scoreFrm0 = curr_scoreFrm0.put(arr_losewin, color = arr_color_win)

curr_scoreFrm1 = overlay(curr_scoreFrm0, scoreFrm_loselose, top_left_point = tlPt_loselose)
curr_scoreFrm1 = curr_scoreFrm1.put(arr_loselose, color = arr_color_lose)

curr_scoreFrm2 = overlay(curr_scoreFrm1, scoreFrm_losewinwin, top_left_point = tlPt_losewinwin)
curr_scoreFrm2 = curr_scoreFrm2.put(arr_losewinwin, color = arr_color_win)

curr_scoreFrm3 = overlay(curr_scoreFrm2, scoreFrm_losewinlose, top_left_point = tlPt_losewinlose)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_losewinlose, color = arr_color_lose)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_loselosewin, color = arr_color_win)
# curr_scoreFrm.show()

curr_scoreFrm4 = overlay(curr_scoreFrm3, scoreFrm_loseloselose, top_left_point = tlPt_loseloselose)
curr_scoreFrm4 = curr_scoreFrm4.put(arr_loseloselose, color = arr_color_lose)

currFrms = BGRFrames.from_frames([BGRFrames.from_frame(curr_scoreFrm0, 25),
                                  BGRFrames.from_frame(curr_scoreFrm1, 25),
                                  BGRFrames.from_frame(curr_scoreFrm2, 25),
                                  BGRFrames.from_frame(curr_scoreFrm3, 25),
                                  BGRFrames.from_frame(curr_scoreFrm4, 25)])
prevFrm = curr_scoreFrm4

all_frames.append(currFrms)
script_lines.append(script_line)


script_line = "If Joao Sousa wins two points consecutively, he wins the match 8-6."
### brighten top path

curr_scoreFrm0 = overlay(curr_scoreFrm, scoreFrm_losewin, top_left_point = tlPt_losewin)
curr_scoreFrm0 = curr_scoreFrm0.put(arr_losewin, color = arr_color_win)

curr_scoreFrm1 = overlay(curr_scoreFrm0, scoreFrm_loselose.dim(dim_factor), top_left_point = tlPt_loselose)
curr_scoreFrm1 = curr_scoreFrm1.put(arr_loselose, color = arr_color_lose_dim)

curr_scoreFrm2 = overlay(curr_scoreFrm1, scoreFrm_losewinwin, top_left_point = tlPt_losewinwin)
curr_scoreFrm2 = curr_scoreFrm2.put(arr_losewinwin, color = arr_color_win)

curr_scoreFrm3 = overlay(curr_scoreFrm2, scoreFrm_losewinlose.dim(dim_factor), top_left_point = tlPt_losewinlose)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_losewinlose, color = arr_color_lose_dim)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_loselosewin, color = arr_color_win_dim)
# curr_scoreFrm.show()

curr_scoreFrm4 = overlay(curr_scoreFrm3, scoreFrm_loseloselose.dim(dim_factor), top_left_point = tlPt_loseloselose)
curr_scoreFrm4 = curr_scoreFrm4.put(arr_loseloselose, color = arr_color_lose_dim)

fadeFrms = vs.interpolate_frames(prevFrm, curr_scoreFrm4, 50)
statFrms = BGRFrames.from_frame(curr_scoreFrm4, 50)
prevFrm = curr_scoreFrm4

currFrms = BGRFrames.from_frames([fadeFrms, statFrms])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "If Federico Gaio wins two points consecutively, Joao Sousa loses the match 6-8."
### brighten bottom path

curr_scoreFrm0 = overlay(curr_scoreFrm, scoreFrm_losewin.dim(dim_factor), top_left_point = tlPt_losewin)
curr_scoreFrm0 = curr_scoreFrm0.put(arr_losewin, color = arr_color_win_dim)

curr_scoreFrm1 = overlay(curr_scoreFrm0, scoreFrm_loselose, top_left_point = tlPt_loselose)
curr_scoreFrm1 = curr_scoreFrm1.put(arr_loselose, color = arr_color_lose)

curr_scoreFrm2 = overlay(curr_scoreFrm1, scoreFrm_losewinwin.dim(dim_factor), top_left_point = tlPt_losewinwin)
curr_scoreFrm2 = curr_scoreFrm2.put(arr_losewinwin, color = arr_color_win_dim)

curr_scoreFrm3 = overlay(curr_scoreFrm2, scoreFrm_losewinlose.dim(dim_factor), top_left_point = tlPt_losewinlose)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_losewinlose, color = arr_color_lose_dim)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_loselosewin, color = arr_color_win_dim)
# curr_scoreFrm.show()

curr_scoreFrm4 = overlay(curr_scoreFrm3, scoreFrm_loseloselose, top_left_point = tlPt_loseloselose)
curr_scoreFrm4 = curr_scoreFrm4.put(arr_loseloselose, color = arr_color_lose)

fadeFrms = vs.interpolate_frames(prevFrm, curr_scoreFrm4, 50)
statFrms = BGRFrames.from_frame(curr_scoreFrm4, 50)
prevFrm = curr_scoreFrm4

currFrms = BGRFrames.from_frames([fadeFrms, statFrms])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "If the next two points are shared between the two players, the tiebreak continues at 7-7."
### brighten middle paths

curr_scoreFrm0 = overlay(curr_scoreFrm, scoreFrm_losewin, top_left_point = tlPt_losewin)
curr_scoreFrm0 = curr_scoreFrm0.put(arr_losewin, color = arr_color_win)

curr_scoreFrm1 = overlay(curr_scoreFrm0, scoreFrm_loselose, top_left_point = tlPt_loselose)
curr_scoreFrm1 = curr_scoreFrm1.put(arr_loselose, color = arr_color_lose)

curr_scoreFrm2 = overlay(curr_scoreFrm1, scoreFrm_losewinwin.dim(dim_factor), top_left_point = tlPt_losewinwin)
curr_scoreFrm2 = curr_scoreFrm2.put(arr_losewinwin, color = arr_color_win_dim)

curr_scoreFrm3 = overlay(curr_scoreFrm2, scoreFrm_losewinlose, top_left_point = tlPt_losewinlose)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_losewinlose, color = arr_color_lose)
curr_scoreFrm3 = curr_scoreFrm3.put(arr_loselosewin, color = arr_color_win)
# curr_scoreFrm.show()

curr_scoreFrm4 = overlay(curr_scoreFrm3, scoreFrm_loseloselose.dim(dim_factor), top_left_point = tlPt_loseloselose)
curr_scoreFrm4 = curr_scoreFrm4.put(arr_loseloselose, color = arr_color_lose_dim)

fadeFrms = vs.interpolate_frames(prevFrm, curr_scoreFrm4, 50)
statFrms = BGRFrames.from_frame(curr_scoreFrm4, 50)
prevFrm = curr_scoreFrm4

currFrms = BGRFrames.from_frames([fadeFrms, statFrms])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "This is practically the same situation as at 6-6, except that Federico Gaio is now on serve."
### dashed arrow back

backArr = make_arrow_from_to_backwards(tlPt_losewinlose, tlPt_lose, num_dashes = 10)
arrFrm = prevFrm.put(backArr, color = const.BGR['bright_grey'])

currFrms = BGRFrames.from_frame(arrFrm, 50)

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "At 6-6, 7-7, 8-8, and so on, play continues until either player wins the match by winning two consecutive points, one on serve, one on return."
### ...

currFrms = BGRFrames.from_frame(arrFrm, 270)

all_frames.append(currFrms)
script_lines.append(script_line)


script_line = "By virtue of this symmetry, we therefore take the probability for Joao Sousa to win the match at 6-6 to be 50%."
### ... fade out after 6-6, and fade in box with win prob 50%.

curr_scoreFrm = overlay(curr_scoreFrm, scoreFrm_win_with_box.dim(dim_factor), top_left_point = tlPt_win)
curr_scoreFrm = curr_scoreFrm.put(arr_win, color = arr_color_win_dim)

curr_scoreFrm_unknown = overlay(curr_scoreFrm, scoreFrm_lose_unknown_with_box, top_left_point = tlPt_lose.up(20))
curr_scoreFrm_unknown = curr_scoreFrm_unknown.put(arr_lose, color = arr_color_lose)

curr_scoreFrm_known = overlay(curr_scoreFrm, scoreFrm_lose_known_with_box, top_left_point = tlPt_lose.up(20))
curr_scoreFrm_known = curr_scoreFrm_known.put(arr_lose, color = arr_color_lose)

statFrms = BGRFrames.from_frame(curr_scoreFrm_unknown, 25)
fadeFrms = vs.interpolate_frames(curr_scoreFrm_unknown, curr_scoreFrm_known, 25)
statFrms2 = BGRFrames.from_frame(curr_scoreFrm_known, 25)

currFrms = BGRFrames.from_frames([BGRFrames.from_frame(curr_scoreFrm4, 10),
                                  BGRFrames.from_frame(curr_scoreFrm3, 10),
                                  BGRFrames.from_frame(curr_scoreFrm2, 10),
                                  BGRFrames.from_frame(curr_scoreFrm1, 10),
                                  BGRFrames.from_frame(curr_scoreFrm0, 10),
                                  statFrms, fadeFrms, statFrms2])

# DecapStream(currFrms, 0, 100).show()

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "The difference between winning and losing the point is the difference between a match that is won and a match that is dead-even."
### highlight the difference, somehow?

resFrm = BGRFrame.make_black()
resFrm = overlay(resFrm, scoreFrm, tlPt_init)
resFrm = overlay(resFrm, percFrm_win, tlPt_win.down(65))
resFrm = overlay(resFrm, percFrm_lose_known, tlPt_lose.up(20))
resFrm = overlay(resFrm, scoreFrm_win.dim(dim_factor), tlPt_win)
resFrm = overlay(resFrm, scoreFrm_lose.dim(dim_factor), tlPt_lose)
resFrm = resFrm.put(arr_win, color = arr_color_win)
resFrm = resFrm.put(arr_lose, color = arr_color_lose)

fadeFrms = vs.interpolate_frames(currFrms[-1], resFrm, 50)
statFrms = BGRFrames.from_frame(resFrm, 50)

currFrms = BGRFrames.from_frames([fadeFrms, statFrms])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "Our measure of leverage is therefore at 50%."
### difference = 50%

winArr = make_arrow_from_to(tlPt_win.down(40), tlPt_losewin)
loseArr = make_arrow_from_to(tlPt_lose.up(40), tlPt_losewin)

meanPt = (winArr[0].rightPt + loseArr[0].rightPt) / 2
meanPt = meanPt.down(8).right(10)

resFrm2 = resFrm.put(winArr, color = const.BGR['bright_grey'])
resFrm2 = resFrm2.put(loseArr, color = const.BGR['bright_grey'])

lvrStr = 'Leverage = 100% - 50% = 50%'
resFrm2 = resFrm2.put(lvrStr, where = meanPt.round_coords(), size = 2, color = const.BGR['bright_grey'])

currFrms = BGRFrames.from_frame(resFrm2, 110)

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "In fact, this is the highest possible leverage in a standard tennis match; with no-ad scoring, a leverage of 100% is possible."
### ...

lastFrm = BGRFrame.make_black()
lastFrm = overlay(lastFrm, scoreFrm, tlPt_init)
lastFrm = lastFrm.put(lvrStr, where = meanPt.round_coords(), size = 2, color = const.BGR['bright_grey'])

lastArr = make_arrow_from_to(tlPt_init, tlPt_init.right(800))
lastFrm = lastFrm.put(lastArr, color = const.BGR['bright_grey'])
# lastFrm.show()

fadeFrms = vs.interpolate_frames(resFrm2, lastFrm, 50)
statFrms = BGRFrames.from_frame(lastFrm, 50)
currFrms = BGRFrames.from_frames([fadeFrms, statFrms])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = ""
holdFrms = BGRFrames.from_frame(statFrms[-1], 25)
all_frames.append(holdFrms)
script_lines.append(script_line)

############# PRESSURE CHART #############
script_line = "With just a few more fairly unrestrictive assumptions, we can quantify leverage for the entire match."

from ATMANpy.types.Overlays import Graph, LineChart, ScatterChart

ys = mtcPnts.apply(maybe_get_pressure)
xs = np.arange(len(ys))
pressChart = LineChart(xs, ys)

new_set = (mtcPnts.apply('get_game_score') == 0).all(axis = 1)
new_set_start = new_set & ~np.roll(new_set, 1)

new_set_start_idxs = np.arange(len(new_set_start))[new_set_start]
set0, set1, set2 = new_set_start_idxs

num_dots = 50
xs1 = np.ones(num_dots) * set1
xs2 = np.ones(num_dots) * set2
ys12 = np.linspace(0, 0.5, num = num_dots)

vChart1 = ScatterChart(xs1, ys12, color = const.BGR['bright_blue'])
vChart2 = ScatterChart(xs2, ys12, color = const.BGR['bright_blue'])

myGraph = Graph([pressChart, vChart1, vChart2], make_default_x_labels = False, y_label_format = '>5.0%', y_label_dist = 1.5, 
                header_text = 'Score Leverage: Gaio v. Sousa', header_horizontal_offset = 5, max_y_override = 0.5, 
                x_label_dist = 0, tick_label_size = 1, x_tick_label_left_shift = 0, header_size = 1, 
                color = const.BGR['bright_grey'],
                resolution = 100)

graphFrm = myGraph.render()

graphFrm = graphFrm.framed(2, const.BGR['bright_grey'])
# graphFrm.show()

last_diagFrm = currFrms[-1]
last_rastFrm = rstFrms5[-1]

from ATMANpy.types.Overlays import overlay_center

fade_backFrm0 = last_rastFrm.dim(dim_factor)
fade_backFrm1 = overlay(fade_backFrm0, last_rastFrm.crop(cropReg), p1)
fade_backFrm2 = overlay(fade_backFrm1, last_rastFrm.crop(cropReg_2), p1_2)
# fade_backFrm.show()

statFrm_black = overlay_center(BGRFrame.make_black(), graphFrm)
statFrm = overlay_center(fade_backFrm2, graphFrm)

fadeFrms = vs.interpolate_frames(last_diagFrm, statFrm_black, 50)
statFrms = BGRFrames.from_frame(statFrm_black, 50)
currFrms = BGRFrames.from_frames([fadeFrms, statFrms])
# DecapStream(currFrms, 0, 100).show()

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "... with our initial situations corresponding to very high and very low leverage, indeed."

# hi_idx = hiPnt.get_point_index()
# lo_idx = loPnt.get_point_index()
# hi_idx_norm = hi_idx / hi_idx ### since hi_idx is the last point
# lo_idx_norm = lo_idx / hi_idx

scoreArr_hi = Arrow.make(p2.up(30), p2.up(100).right(890), shorten_by = 0.99, hat_length = 20)
scoreArr_lo = Arrow.make(p1_2.right(100), p1_2.up(300).left(32), shorten_by = 0.97, hat_length = 20)

arrFrm1 = statFrm.put(scoreArr_hi, thickness = 2)
arrFrm2 = arrFrm1.put(scoreArr_lo, thickness = 2)

fadeFrms = vs.interpolate_frames(statFrm_black, statFrm, 50)
arrFrms1 = BGRFrames.from_frame(arrFrm1, 50)
arrFrms2 = BGRFrames.from_frame(arrFrm2, 150)

currFrms = BGRFrames.from_frames([fadeFrms, 
                                  arrFrms1,
                                  arrFrms2])

all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    So then let's take a look at serve placement. 
                    Our sample consists of twenty matches played during the 2021 Brest Challenger Tournament, 
                    of which Joao Sousa played four.
                """
### remove all elements except the top left situation.

from ATMANpy.types.Overlays import reverse_cascade

arrFrms1_back = BGRFrames.from_frame(arrFrm1, 15)
statFrms_back = BGRFrames.from_frame(statFrm, 10)
fade_backFrms2 = BGRFrames.from_frame(fade_backFrm2, 5)
fade_backFrms1 = BGRFrames.from_frame(fade_backFrm1, 5)
fade_backFrms0 = BGRFrames.from_frame(fade_backFrm0, 5)
revRst, revFrms = reverse_cascade(myRst, keep_which = [(0, 0)], frames_per_element = 5)
revFrms = revFrms.dim(dim_factor)

revRst.add(last_frame_unframed.dim(dim_factor), 0, 0)
unFrm = revRst.render()

fadeFrms = vs.interpolate_frames(revFrms[-1], unFrm, 10)

currFrms = BGRFrames.from_frames([arrFrms1_back, statFrms_back, 
                                  fade_backFrms2, fade_backFrms1, fade_backFrms0, 
                                  revFrms, fadeFrms])

all_frames.append(currFrms)
script_lines.append(script_line)

# DecapStream(currFrms, 0, 100).show()

script_line = "In all of his tournament matches, when serving from the ad side..."
### grow frame

factors = np.linspace(1, 2, num = last_frames_unframed.length())

out_frames = []
blackFrm = BGRFrame.make_black()
for idx, f in enumerate(factors):
    ### curr_frame
    currFrm = last_frames_unframed[idx]
    ### calculate dim factor
    lmbd = f - 1
    curr_dim_factor = 0.25 * (1 - lmbd) + 1 * lmbd
    ### dim and rescale
    currFrm = currFrm.dim(curr_dim_factor).rescale(f)
    ### overlay on black frame so we have consistent size
    outFrm = overlay(blackFrm, currFrm, top_left_point = Point2D((0, 0)))
    out_frames.append(outFrm)

outFrms = BGRFrames.from_single_frames(out_frames)

all_frames.append(outFrms)
script_lines.append(script_line)


vis_ut.show(all_frames)


### Saving the separate MultiFrame objects and the Joint Script
if SAVE:
    assert len(script_lines) == len(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        print(idx)
        subFrms, script_line = frames_and_script
        subFrms.save(fullPath, f'seg_{idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{idx}.txt'), mode = 'w') as f:
            f.write(script_line)
