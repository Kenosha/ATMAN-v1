
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
import ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils as vis_ut

########################################
########### SAVE FOLDER ################
########################################

SAVE = True

DESCRIPTOR = 'short'
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

### PARAMETERS
RESOLUTION_MODIFIER = 30 # 2
RESOLUTION_BACKGROUND = 1.5 * RESOLUTION_MODIFIER
RESOLUTION_OBJECTS = 1 * RESOLUTION_MODIFIER
RESOLUTION_KDE = 1 * RESOLUTION_MODIFIER
MIX_COLORS_KDE = True
BANDWIDTH_KDE = 35
SCENE_FADE = 0.9
CONTOUR_PROP = 0.5
EPSILON = 0.002
EMPHASIS_DIM_FACTOR = 1.5
ARGUMENT_LUMINOSITY = 1.25
ARGUMENT_BLINKS_PER_SECOND = None
ADAPTIVE_RESOLUTION_EXPONENT = 1.5
BLINK_RATE = 1
prmDict = {'background_resolution': RESOLUTION_BACKGROUND,
            'argument_resolution': RESOLUTION_OBJECTS,
            'foreground_resolution': RESOLUTION_OBJECTS,
            'dummy_resolution': RESOLUTION_KDE,
            'adaptive_resolution_exponent': ADAPTIVE_RESOLUTION_EXPONENT,
            'argument_luminosity': ARGUMENT_LUMINOSITY,
            'argument_blinks_per_second': ARGUMENT_BLINKS_PER_SECOND,
            }

## camera
from ATMANpy.auxil.court import onto_service_box_cam
from ATMANpy.auxil.court import back_view_asymmetric_cam
from ATMANpy.types.ShapesTennis3D import Court

boxCam_ad = onto_service_box_cam(which = 'Ad', side = 'Bot', dist = 150).tilt(0.15)
boxCam_deuce = onto_service_box_cam(which = 'Deuce', side = 'Bot', dist = 150).tilt(0.15)
backCam = back_view_asymmetric_cam(which = 'Ad', side = 'Bot')
brestCam = Camera.calibrated_brest_behind().resize(3)


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

firstPnts = dataPnts.filter('is_first_serve').enforce_direction('Top')
adPnts = firstPnts.filter('is_serve_ad')
deucePnts = firstPnts.filter('is_serve_deuce')

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

all_frames = []
script_lines = []

script_line =   """                    
                    In this analysis, we take a quick look at whether players change their serve selection depending 
                    on how much is at stake in a given point. 
                """

fadeFrms = vs.interpolate_frames(blackFrm, black_with_titleFrm, 100)
statFrms = BGRFrames.from_frame(black_with_titleFrm, 100)
currFrms = BGRFrames.from_frames([fadeFrms, statFrms])

all_frames.append(currFrms)
script_lines.append(script_line)


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

script_line =   """
                    Why might this be interesting? Well, consider this situation here: 
                    In the quartefinals of the 2021 Brest Challenger Tournament, where all our data is taken from, 
                    Joao Sousa has match point on his serve.
                """
currFrms = BGRFrames.from_frames([rstFrms_fade, rstFrms0])
all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    The hypothesis is that when so much is at stake in a point,
                    a player will want to employ the serve that maximizes his or her chances of winning.
                """
currFrms = BGRFrames.from_frames([rstFrms1, rstFrms2])
all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    When there is much less at stake, such as here, at 40-Love, 
                    a player might be more willing to go for sub-optimal serves in order to 
                    hopefully keep the receiver guessing at a later time, when the point really matters.
                """
currFrms = BGRFrames.from_frames([rstFrms3, rstFrms4, rstFrms5])
all_frames.append(currFrms)
script_lines.append(script_line)

script_line = ""
holdFrms = BGRFrames.from_frame(currFrms[-1], 25)
all_frames.append(holdFrms)
script_lines.append(script_line)


############# PRESSURE CHART #############

script_line =   """
                    We quantify what is at stake by a notion of leverage, 
                    which essentially measures the impact of the outcome of a point on the outcome of the match.
                """

from ATMANpy.types.Overlays import Graph, LineChart, ScatterChart, overlay_footnote

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
                header_text = 'Score Leverage: Gaio v. Sousa', header_horizontal_offset = 5, 
                header_vertical_offset = 0.5, 
                max_y_override = 0.5, 
                x_label_dist = 0, tick_label_size = 1, x_tick_label_left_shift = 0, header_size = 1, 
                color = const.BGR['bright_grey'],
                resolution = 100)

graphScn, graphPnts = myGraph.preview()

p1 = graphScn[1].rightPt
p2 = graphScn[3].devectorize()[0]
p3 = graphScn[4].devectorize()[0]

p1 = p1.annotate('Set 1').recolor(const.BGR['bright_blue']).back(0.5).right(0.2)
p2 = p2.annotate('Set 2').back(0.5).right(0.2)
p3 = p3.annotate('Set 3').back(0.5).right(0.2)

add_graphPnts = [*graphPnts, p1, p2, p3]

graphFrm = View.make(myGraph.cam, graphScn, annotatedPoints = add_graphPnts, resolution = myGraph.resolution).snap()

graphFrm = graphFrm.framed(2, const.BGR['bright_grey'])
# graphFrm.show()

### put set denoters


dim_factor = 0.25
last_rastFrm = rstFrms5[-1]

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

# lsFrm.put(cropReg).show()
# lsFrm.put(cropReg_2).show()
cropFrm = last_rastFrm.crop(cropReg)

fade_backFrm0 = last_rastFrm.dim(dim_factor)
fade_backFrm1 = overlay(fade_backFrm0, last_rastFrm.crop(cropReg), p1)
fade_backFrm2 = overlay(fade_backFrm1, last_rastFrm.crop(cropReg_2), p1_2)

statFrm = overlay_center(fade_backFrm2, graphFrm)
statFrms = BGRFrames.from_frame(statFrm, 50)
fadeFrms = vs.interpolate_frames(last_rastFrm, statFrm, 50)

# currFrms = BGRFrames.from_frames([fadeFrms, statFrms])
# DecapStream(currFrms, 0, 100).show()

scoreArr_hi = Arrow.make(p2.up(30), p2.up(100).right(890), shorten_by = 0.99, hat_length = 20)
scoreArr_lo = Arrow.make(p1_2.right(100), p1_2.up(300).left(32), shorten_by = 0.97, hat_length = 20)

arrFrm1 = statFrm.put(scoreArr_hi, thickness = 2)
arrFrm2 = arrFrm1.put(scoreArr_lo, thickness = 2)
arrFrms1 = BGRFrames.from_frame(arrFrm1, 50)
arrFrms2 = BGRFrames.from_frame(arrFrm2, 100)

currFrms = BGRFrames.from_frames([fadeFrms, statFrms,
                                  arrFrms1, arrFrms2])

currFrms = overlay_footnote(currFrms, "00:56-02:49:", "For more info on\nthe construction and interpretation\nof the leverage metric.")


all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    So then let's take a look at serve placement. 
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


def server_is(somePnt, player_name):
    return somePnt.get_serve().get_hit_player_name() == player_name

##################################
########### JOAO SOUSA ###########
##################################
player_name = 'Joao Sousa'

### COLLECT SHOTS FOR PLAYER
serveShots = adPnts.filter(server_is, player_name = player_name).get_serves()
print(f'{len(serveShots)} ad serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_ad, side = 'Bottom')
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness, 'Ad')
tlPt_leg = Point2D((50, 50))

### FLYING BALL FEATURES 
line_slomo = 5
ball_bounces_when = 12 * line_slomo
ball_flight_over_when = 17 * line_slomo

### PREVIOUS FRAME 
hiPnt = dataPnts[1131]

lastFrm = hiPnt.stream(confs = confs).decap().frames[0].rescale(3)

shapes = vs.ball_flight(serveShots_sample, line_interpolation_factor = 5, which_endpoint = 'penultimate')

# splineSF = serveShots_sample.feat_spline_simul(color = const.BGR['ball_yellow'], which_endpoint = 'penultimate')
# shapes = [someLns for someLns in splineSF.values() if someLns is not None]

script_line = "... it seems that Joao Sousa goes for serves out wide somewhat more often than for serves down the middle."

myFrms_pre, animDict = vs.animate_sequence(shapes[:ball_bounces_when], 
                                        start_cam = brestCam, stop_cam = brestCam,
                                        cumulate_argument = True, cumulative_foreground_transparence = 0.9, 
                                        **prmDict,
                                        animate_over = lastFrm 
                                        )
# DecapStream(myFrms, 0, 100).show(ms_between_frames = 50)

### SCATTERS
MARK_COLOR = const.BGR['strong_blue']
ZERO_COLOR = const.BGR['grass_green']
ONE_COLOR = const.BGR['intense_red']
MARK_SIZE = 2

from ATMANpy.auxil.court import courtLines
adReg = courtLines.make().add_singles_netline().get_service_box(which = 'Ad', side = 'Bottom')
adReg_enlarged = adReg.scale(1.25)
bounceScn_mono = vs.scatter_markmap(serveShots_clean_bounce, showReg = adReg_enlarged,
                                    mix_colors = False, one_color = const.BGR['strong_blue'])
bounceScn_poly = vs.scatter_markmap(serveShots_clean_bounce, showReg = adReg_enlarged, 
                                    zero_color = ZERO_COLOR, one_color = ONE_COLOR,
                                    color_func_or_arr = 'get_total_match_score_pressure',
                                    mark_size = MARK_SIZE)


netScn_mono = vs.scatter_markmap(serveShots_net, position_func = 'get_net_crossing_position', flat_along_dimension = 'y',
                                    mix_colors = False, one_color = MARK_COLOR)
netScn_poly = vs.scatter_markmap(serveShots_net, position_func = 'get_net_crossing_position', flat_along_dimension = 'y',
                                    zero_color = ZERO_COLOR, one_color = ONE_COLOR,
                                    color_func_or_arr = 'get_total_match_score_pressure')

markScn_mono = bounceScn_mono.combine(netScn_mono)
markScn_poly = bounceScn_poly.combine(netScn_poly)

myFrms_post, animDict = vs.animate_sequence(shapes[ball_bounces_when:ball_flight_over_when], 
                                        start_cam = brestCam, stop_cam = brestCam,
                                        background_scene = markScn_mono,
                                        cumulate_argument = True, cumulative_foreground_transparence = 0.9, 
                                        **prmDict,
                                        **animDict,
                                        animate_over = lastFrm 
                                        )

### FADE OUT COURT
courtScn = Scene.make_court(singles_net = False, net_mesh_density = 3)
lastFrm = myFrms_post[-1]
nextFrm = View.make(brestCam, courtScn.combine(markScn_mono)).snap()
nextFrm = overlay(nextFrm, legFrm, tlPt_leg)
itplFrms = vs.interpolate_frames(lastFrm, nextFrm, 50)

jointFrms = BGRFrames.from_frames([myFrms_pre, myFrms_post, itplFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "To be able to quantify serve location more precisely, we record where any given serve would have crossed the service line."

zoomFrms, animDict = vs.animate_cam_motion(markScn_mono, background_scene = courtScn, start_cam = brestCam, stop_cam = boxCam_ad, num_frames = 25,
                                            **prmDict)
zoomFrms = overlay(zoomFrms, legFrm, tlPt_leg)

### FLIGHT LINE
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line
from ATMANpy.types.Shapes3D import Line3D
from ATMANpy.types.ShapesTennis3D import Mark
from ATMANpy.funcs.util_funcs import interpolate_tuples

srvLn, indDict = CourtLines.make().add_singles_netline().get_service_line_and_indicator('Bottom', 'Ad')

pt_idx = 5
someShot = serveShots_clean_bounce[pt_idx]

hit_coords = someShot.get_ball_hit_position()
pressPt_hit_2D = Point2D(hit_coords[:2])
bnc_coords = someShot.get_first_bounce_position()
pressPt_bnc_2D = Point2D(bnc_coords[:2])

press = someShot.get_total_match_score_pressure()

bncMark = Mark.make(Point3D.from_point2D(pressPt_bnc_2D, z = 0), size = MARK_SIZE, color = MARK_COLOR)

flightLn_2D = Line(pressPt_hit_2D, pressPt_bnc_2D)
interPt_2D = flightLn_2D.weak_intersect(srvLn)

pressPt_hit = Point3D.from_point2D(pressPt_hit_2D, z = 0)
interPt = Point3D.from_point2D(interPt_2D, z = 0)
flightLn = Line3D(pressPt_hit, interPt, color = const.BGR['ball_yellow'])    

flight_itpl_line_segments = vs.divide_line(flightLn, 25, dashed = True)
flight_line_and_mark_scenes = [Scene([line_seg, bncMark]) for line_seg in flight_itpl_line_segments]
flight_line_and_mark_scene = Scene([*flight_itpl_line_segments, bncMark])

lineFrms, animDict = vs.animate_sequence(flight_line_and_mark_scenes, background_scene = courtScn, cumulate_argument = True, 
                                    start_cam = boxCam_ad, stop_cam = boxCam_ad,
                                    cumulative_foreground_transparence = 0.95,
                                    **prmDict,
                                    **animDict)

lineFrms = overlay(lineFrms, legFrm, tlPt_leg)

# DecapStream(lineFrms, 0, 100).show()

statFrms_pre, animDict = vs.animate_static(flight_line_and_mark_scene, background_scene = courtScn, static_cam = boxCam_ad,
                                           num_frames = 40, **prmDict, **animDict)
# statFrms[-1].show()

statFrms_pre = overlay(statFrms_pre, legFrm, tlPt_leg)

### PROJECT MARK ONTO SERVICE LINE
bncMark_on_line = Mark.make(interPt, color = MARK_COLOR)
flight_line_and_mark_scene_end = Scene([*flight_itpl_line_segments, bncMark_on_line])

mark_moving_scenes = vs.interpolate_scenes(flight_line_and_mark_scene, flight_line_and_mark_scene_end, num_frames = 20)

animDict['previous_foreground_scene'] = Scene([shape for shape in animDict['previous_foreground_scene'].shapes if shape != bncMark])

moveFrms, animDict = vs.animate_sequence(mark_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam_ad, stop_cam = boxCam_ad,
                                    **prmDict,
                                    **animDict)

moveFrms = overlay(moveFrms, legFrm, tlPt_leg)

statFrms_post = BGRFrames.from_frame(moveFrms[-1], 40)
statFrms_post = overlay(statFrms_post, legFrm, tlPt_leg)

jointFrms1 = BGRFrames.from_frames([zoomFrms, lineFrms, statFrms_pre, moveFrms, statFrms_post])

### FADE OTHER MARKS BACK IN
markScn_mono_bar_one = Scene([*[mk for mk in markScn_mono if mk != bncMark], bncMark_on_line])

monoView = View.make(boxCam_ad, markScn_mono_bar_one, resolution = prmDict['argument_resolution'], adaptive_resolution_exponent = prmDict['adaptive_resolution_exponent'])

backView = View.make(boxCam_ad, courtScn, resolution = prmDict['background_resolution'])
monoFrm = backView.combine(monoView).snap()

itplFrms = vs.interpolate_frames(moveFrms[-1], monoFrm, 40)

itplFrms = overlay(itplFrms, legFrm, tlPt_leg)

### MOVE OTHER MARKS
hitArr_bnc = serveShots_clean_bounce.get_ball_hit_position()[:, :2]
posArr_bnc = serveShots_clean_bounce.get_first_bounce_position()

hitArr_net = serveShots_net.get_ball_hit_position()[:, :2]
posArr_net = serveShots_net.get_ball_last_position()[:, :2]

hitArr = np.concatenate([hitArr_bnc, hitArr_net], axis = 0)
posArr = np.concatenate([posArr_bnc, posArr_net], axis = 0)

hitPts = [Point2D(tuple(coords)) for coords in hitArr]
posPts = [Point2D(tuple(coords)) for coords in posArr]
flightLns = [Line(hitPt, posPt) for hitPt, posPt in zip(hitPts, posPts)]
interPts = [flightLn.weak_intersect(srvLn) for flightLn in flightLns]
interPts_3D = [Point3D.from_point2D(pt, z = 0) for pt in interPts]
interMarks = [Mark.make(pt, color = prev_mark.color) for pt, prev_mark in zip(interPts_3D, markScn_mono.shapes)]
markScn_mono_moved = Scene(interMarks)

markScn_mono_box = Scene([mark if mark != bncMark else bncMark for mark in markScn_mono.shapes])

all_marks_moving_scenes = vs.interpolate_scenes(markScn_mono_box, markScn_mono_moved, 20)

moveFrms, animDict = vs.animate_sequence(all_marks_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam_ad, stop_cam = boxCam_ad,
                                    **prmDict)
moveFrms = overlay(moveFrms, legFrm, tlPt_leg)

statFrms, animDict = vs.animate_static(markScn_mono_moved, background_scene = courtScn, static_cam = boxCam_ad, num_frames = 20,
                                        **prmDict)
statFrms = overlay(statFrms, legFrm, tlPt_leg)

jointFrms2 = BGRFrames.from_frames([itplFrms, moveFrms, statFrms])

jointFrms = BGRFrames.from_frames([jointFrms1, jointFrms2])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "Indeed, the serve out wide is favored against the serve down the middle."

### MAKE KDE WITH ALL ON BASELINE
kde_kwargs = {'position_func': vis_ut.find_intersection_with_service_line, 'normalize_density': True, 
              'mix_colors': False, 'mix_alpha': False, 
              'resolution': RESOLUTION_KDE, 'bandwidth': BANDWIDTH_KDE}

nullScn, jointScn = vis_ut.joint_kde(serveShots_sample, 'Ad', **kde_kwargs)
splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, 'Ad', **kde_kwargs)

itplScns = vs.interpolate_dummy_scenes(nullScn, jointScn, num_frames = 25)

kdeFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam_ad, stop_cam = boxCam_ad,
                                    **prmDict)
kdeFrms = overlay(kdeFrms, legFrm, tlPt_leg)

statFrms, animDict = vs.animate_static(jointScn, background_scene = courtScn, static_cam = boxCam_ad, num_frames = 25,
                                        **prmDict)
statFrms = overlay(statFrms, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([kdeFrms, statFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "We can now split this distribution by point leverage."

jointScn = jointScn.combine(jointScn).combine(jointScn)
itplScns = vs.interpolate_dummy_scenes(jointScn, splitScn, num_frames = 25)

### FADE OUT COURT ###
court_fadeout_scenes = vs.interpolate_scenes(courtScn, courtScn.transparify(0), num_frames = 25)
prmDict_no_illum = {k: v for k, v in prmDict.items()}
prmDict_no_illum['argument_luminosity'] = 1
prmDict_no_illum['argument_resolution'] = prmDict['background_resolution']

fadeFrms, animDict = vs.animate_sequence(court_fadeout_scenes, background_scene = xyScn, cumulate_argument = False, 
                                        start_cam = boxCam_ad, stop_cam = boxCam_ad,
                                        **prmDict_no_illum,
                                        **animDict)
fadeFrms = overlay(fadeFrms, legFrm, tlPt_leg)

splitFrms, animDict = vs.animate_sequence(itplScns, background_scene = xyScn, cumulate_argument = False, 
                                        start_cam = boxCam_ad, stop_cam = boxCam_ad,
                                        annotated_points = annPnts[:-1],
                                        **prmDict)
splitFrms = overlay(splitFrms, legFrm, tlPt_leg)

statFrms, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam_ad, num_frames = 25,
                                        annotated_points = annPnts[:-1],
                                        **prmDict)

statFrms = overlay(statFrms, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([fadeFrms, splitFrms, statFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    In Joao Sousa's case, this segmented view is quite revealing. 
                """

statFrms_fadeback = vs.interpolate_frames(statFrms[-1], statFrms[0], 50)
jointFrms = BGRFrames.from_frames([statFrms_fadeback, statFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

alpha = 0.25
loScn = splitScn[0].combine(splitScn[1].realpha(alpha)).combine(splitScn[2].realpha(alpha))
midScn = splitScn[0].realpha(alpha).combine(splitScn[1]).combine(splitScn[2].realpha(alpha))
hiScn = splitScn[0].realpha(alpha).combine(splitScn[1].realpha(alpha)).combine(splitScn[2])

script_line =   """
                    At low leverage, he serves down the middle as often as he serves out wide; 
                """

annPnts_dim = [annPnts[0], annPnts[1].dim(alpha), annPnts[2].dim(alpha), *annPnts[3:-1]]

fadedFrms, _ = vs.animate_static(loScn, static_cam = boxCam_ad, background_scene = xyScn, num_frames = 50,
                                 annotated_points = annPnts_dim, **prmDict)
fadedFrms = overlay(fadedFrms, legFrm, tlPt_leg)

fadeFrms = vs.interpolate_frames(statFrms[-1], fadedFrms[0], 25)

jointFrms = BGRFrames.from_frames([fadeFrms, fadedFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    However, as leverage increases, he reduces his serves down the middle.
                """

annPnts_dim = [annPnts[0].dim(alpha), annPnts[1], annPnts[2].dim(alpha), *annPnts[3:-1]]

fadedFrms, _ = vs.animate_static(midScn, static_cam = boxCam_ad, background_scene = xyScn, num_frames = 50,
                                 annotated_points = annPnts_dim, **prmDict)
fadedFrms = overlay(fadedFrms, legFrm, tlPt_leg)

fadeFrms = vs.interpolate_frames(fadeFrms[-1], fadedFrms[0], 25)

jointFrms = BGRFrames.from_frames([fadeFrms, fadedFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """ 
                    At high leverage, he serves out wide almost exclusively. 
                """

annPnts_dim = [annPnts[0].dim(alpha), annPnts[1].dim(alpha), annPnts[2], *annPnts[3:-1]]

fadedFrms, _ = vs.animate_static(hiScn, static_cam = boxCam_ad, background_scene = xyScn, num_frames = 50,
                                 annotated_points = annPnts_dim, **prmDict)
fadedFrms = overlay(fadedFrms, legFrm, tlPt_leg)

fadeFrms = vs.interpolate_frames(fadeFrms[-1], fadedFrms[0], 25)

jointFrms = BGRFrames.from_frames([fadeFrms, fadedFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = ""

itplFrms = vs.interpolate_frames(fadedFrms[-1], statFrms[0])
statFrms = BGRFrames.from_frame(statFrms[0], 75)
jointFrms = BGRFrames.from_frames([itplFrms, statFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

# script_line =   """
#                     Two statistical tests establish that there is a statistically significant difference between high and low leverage serve placement.
#                 """

# statFrms, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam_ad, num_frames = 25,
#                                         annotated_points = annPnts,
#                                         **prmDict)
# statFrms = overlay(statFrms, legFrm, tlPt_leg)

# statFrms = overlay_footnote(statFrms, "XX:XX-XX:XX:", "For more info on\nthe Kolmogorov-Smirnov and\nAnderson-Darling tests.")

# all_frames.append(statFrms)
# script_lines.append(script_line)

### fade out kde, fade in court.
script_line = "Let's take a look at another interesting case."

courtScn = Scene.make_court(singles_net = False, net_mesh_density = 3)
courtFrm = View.make(boxCam_ad, courtScn, resolution = prmDict['background_resolution']).snap()
itplFrms = vs.interpolate_frames(statFrms[-1], courtFrm, 50)

### move over
moveFrms, animDict = vs.animate_cam_motion(None, background_scene = courtScn, start_cam = boxCam_ad, stop_cam = boxCam_deuce, 
                                           num_frames = 50,
                                           **prmDict)

jointFrms = BGRFrames.from_frames([itplFrms, moveFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)


player_name = 'Holger Rune'
### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.filter(server_is, player_name = player_name).get_serves()
print(f'{len(serveShots)} deuce serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce, side = 'Bottom')
serveShots_net = serveShots.filter('is_in_net')
serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness, 'Deuce')
tlPt_leg = Point2D((50, 50))

splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, 'Deuce', **kde_kwargs)
_, _, annPnts_join = vis_ut.split_kde(serveShots_sample, 'Deuce', join_mid_and_high = True, **kde_kwargs)

### first deuce player
script_line = "Holger Rune is serving from the deuce side. On aggregate, it seems he prefers the serve out wide, here."

jointScn_null, jointScn = vis_ut.joint_kde(serveShots_sample, 'Deuce', **kde_kwargs)

xyFrm = View.make(boxCam_deuce, xyScn, resolution = prmDict['background_resolution']).snap()
itplFrms = vs.interpolate_frames(moveFrms[-1], xyFrm, 50)

itplScns = vs.interpolate_dummy_scenes(jointScn_null, jointScn, 50)

morphFrms, _ = vs.animate_sequence(itplScns, start_cam = boxCam_deuce, stop_cam = boxCam_deuce, 
                                **prmDict, animate_over = itplFrms)
morphFrms = overlay(morphFrms, legFrm, tlPt_leg)

all_frames.append(morphFrms)
script_lines.append(script_line)

### transition to hi/mid/lo
script_line =   """
                    However, the breakdown shows that the deuce serve out wide is predominantly a high-leverage serve;
                    at low leverage, Holger Rune switches it up, serving mostly down the middle.
                """

jointScn_triple = jointScn.combine(jointScn).combine(jointScn)
itplScns = vs.interpolate_dummy_scenes(jointScn_triple, splitScn, num_frames = 50)

splitFrms, _ = vs.animate_sequence(itplScns, start_cam = boxCam_deuce, stop_cam = boxCam_deuce, 
                                   background_scene = xyScn, annotated_points = annPnts[:-1], **prmDict)
splitFrms = overlay(splitFrms, legFrm, tlPt_leg)

all_frames.append(splitFrms)
script_lines.append(script_line)

script_line = ""
statFrms = BGRFrames.from_frame(splitFrms[-1], 25)
all_frames.append(statFrms)
script_lines.append(script_line)

script_line =   """
                    In this video, we have given a quick overview over two cases 
                    in which players seem to adjust their serve patterns based on leverage.
                    
                    As an opponent, figuring this out during the match can be costly, or even outright impossible,
                    since high leverage points are consequential, but few in number.

                    On the other hand, knowing the serve mix at different leverages ahead of time, there is no need to guess.

                    For a deeper look into a few more players' serve mixes, and the methodology employed, consult the longer version of this video.
                """

stillFrms = BGRFrames.from_frame(splitFrms[-1], 250)
fadeoutFrms = stillFrms.dim_progressively(1, 0)
jointFrms = BGRFrames.from_frames([stillFrms, fadeoutFrms])

all_frames.append(jointFrms)
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
