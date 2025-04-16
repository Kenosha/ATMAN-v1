
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
from ATMANpy.types.Points import Point2D
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

DESCRIPTOR = 'ad_side_example'
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

firstPnts = dataPnts.filter('is_first_serve').enforce_direction('Top')
adPnts = firstPnts.filter('is_serve_ad')

### PARAMETERS
RESOLUTION_MODIFIER = 1 # 2
RESOLUTION_BACKGROUND = 1 * RESOLUTION_MODIFIER
RESOLUTION_OBJECTS = 1 * RESOLUTION_MODIFIER
RESOLUTION_KDE = 2 * RESOLUTION_MODIFIER
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

boxCam = onto_service_box_cam(which = 'Ad', side = 'Bot', dist = 150).tilt(0.15)
backCam = back_view_asymmetric_cam(which = 'Ad', side = 'Bot')
brestCam = Camera.calibrated_brest_behind().resize(3)

## service line and region
srvLn, indDict = CourtLines.make().add_singles_netline().get_service_line_and_indicator('Bottom', 'Ad')

##### SIDE CALCULATATION ######
# How much more important is a point in the high pressure bucket on average than a point in the low pressure bucket?
hiShots = firstPnts.get_serves().filter('get_total_match_score_pressure', is_above = 0.08)
loShots = firstPnts.get_serves().filter('get_total_match_score_pressure', is_below = 0.04)

hi_avg = hiShots.get_total_match_score_pressure().mean()
lo_avg = loShots.get_total_match_score_pressure().mean()
ratio = hi_avg / lo_avg

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

all_frames = []
script_lines = []

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
courtScn = Scene.make_court(singles_net = False)
lastFrm = myFrms_post[-1]
nextFrm = View.make(brestCam, courtScn.combine(markScn_mono)).snap()
nextFrm = overlay(nextFrm, legFrm, tlPt_leg)
itplFrms = vs.interpolate_frames(lastFrm, nextFrm, 50)

jointFrms = BGRFrames.from_frames([myFrms_pre, myFrms_post, itplFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

### ZOOM ONTO BOX
script_line = "To understand if serve placement correlates with leverage, we color in each bounce: the redder a mark, the higher the leverage."

myFrms, animDict = vs.animate_cam_motion(markScn_mono, background_scene = courtScn, start_cam = brestCam, stop_cam = boxCam, num_frames = 25,
                                            **prmDict)
myFrms = overlay(myFrms, legFrm, tlPt_leg)

### MONO TO POLY
backView = View.make(boxCam, courtScn, resolution = prmDict['background_resolution'])
monoView = View.make(boxCam, markScn_mono, resolution = prmDict['argument_resolution'], adaptive_resolution_exponent = prmDict['adaptive_resolution_exponent'])
polyView = View.make(boxCam, markScn_poly, resolution = prmDict['argument_resolution'], adaptive_resolution_exponent = prmDict['adaptive_resolution_exponent'])
monoFrm = backView.combine(monoView).snap()
polyFrm = backView.combine(polyView).snap()
itplFrms = vs.interpolate_frames(monoFrm, polyFrm, num_frames = 50)
itplFrms = overlay(itplFrms, legFrm, tlPt_leg)

animDict['previous_foreground_scene'] = markScn_poly

polyFrms = BGRFrames.from_frame(polyFrm, 50)
polyFrms = overlay(polyFrms, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([myFrms, itplFrms, polyFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "To be able to quantify serve location more precisely, we record where any given serve would have crossed the service line."

### FLIGHT LINE
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line
from ATMANpy.types.Shapes3D import Line3D
from ATMANpy.types.ShapesTennis3D import Mark
from ATMANpy.funcs.util_funcs import interpolate_tuples

pt_idx = 5
someShot = serveShots_clean_bounce[pt_idx]

hit_coords = someShot.get_ball_hit_position()
pressPt_hit_2D = Point2D(hit_coords[:2])
bnc_coords = someShot.get_first_bounce_position()
pressPt_bnc_2D = Point2D(bnc_coords[:2])

press = someShot.get_total_match_score_pressure()
color = interpolate_tuples(ZERO_COLOR, ONE_COLOR, press)

bncMark = Mark.make(Point3D.from_point2D(pressPt_bnc_2D, z = 0), size = MARK_SIZE, color = color)

flightLn_2D = Line(pressPt_hit_2D, pressPt_bnc_2D)
interPt_2D = flightLn_2D.weak_intersect(srvLn)

pressPt_hit = Point3D.from_point2D(pressPt_hit_2D, z = 0)
interPt = Point3D.from_point2D(interPt_2D, z = 0)
flightLn = Line3D(pressPt_hit, interPt, color = const.BGR['bright_blue'])    

flight_itpl_line_segments = vs.divide_line(flightLn, 25, dashed = True)
flight_line_and_mark_scenes = [Scene([line_seg, bncMark]) for line_seg in flight_itpl_line_segments]
flight_line_and_mark_scene = Scene([*flight_itpl_line_segments, bncMark])

lineFrms, animDict = vs.animate_sequence(flight_line_and_mark_scenes, background_scene = courtScn, cumulate_argument = True, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    cumulative_foreground_transparence = 0.95,
                                    **prmDict,
                                    **animDict)

lineFrms = overlay(lineFrms, legFrm, tlPt_leg)

# DecapStream(lineFrms, 0, 100).show()

statFrms_pre, animDict = vs.animate_static(flight_line_and_mark_scene, background_scene = courtScn, static_cam = boxCam,
                                        **prmDict, 
                                        **animDict)
# statFrms[-1].show()

statFrms_pre = overlay(statFrms_pre, legFrm, tlPt_leg)

### PROJECT MARK ONTO SERVICE LINE
bncMark_on_line = Mark.make(interPt, color = color)
flight_line_and_mark_scene_end = Scene([*flight_itpl_line_segments, bncMark_on_line])

mark_moving_scenes = vs.interpolate_scenes(flight_line_and_mark_scene, flight_line_and_mark_scene_end, num_frames = 25)

animDict['previous_foreground_scene'] = Scene([shape for shape in animDict['previous_foreground_scene'].shapes if shape != bncMark])

moveFrms, animDict = vs.animate_sequence(mark_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    **prmDict,
                                    **animDict)

moveFrms = overlay(moveFrms, legFrm, tlPt_leg)

statFrms_post = BGRFrames.from_frame(moveFrms[-1], 50)
statFrms_post = overlay(statFrms_post, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([lineFrms, statFrms_pre, moveFrms, statFrms_post])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "By projecting onto the service line along the flight path..."

### FADE OTHER MARKS BACK IN
markScn_poly_bar_one = Scene([*[mk for mk in markScn_poly if mk != bncMark], bncMark_on_line])

polyView = View.make(boxCam, markScn_poly_bar_one, resolution = prmDict['argument_resolution'], adaptive_resolution_exponent = prmDict['adaptive_resolution_exponent'])

polyFrm = backView.combine(polyView).snap()

itplFrms = vs.interpolate_frames(moveFrms[-1], polyFrm, 50)

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
interMarks = [Mark.make(pt, color = prev_mark.color) for pt, prev_mark in zip(interPts_3D, markScn_poly.shapes)]
markScn_poly_moved = Scene(interMarks)

markScn_poly_box = Scene([mark if mark != bncMark else bncMark for mark in markScn_poly.shapes])

all_marks_moving_scenes = vs.interpolate_scenes(markScn_poly_box, markScn_poly_moved, 25)

moveFrms, animDict = vs.animate_sequence(all_marks_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    **prmDict)
moveFrms = overlay(moveFrms, legFrm, tlPt_leg)

statFrms, animDict = vs.animate_static(markScn_poly_moved, background_scene = courtScn, static_cam = boxCam, num_frames = 25,
                                        **prmDict)
statFrms = overlay(statFrms, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([itplFrms, moveFrms, statFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "... we can easily construct a distribution of serve placement. Indeed, the serve out wide is favored against the serve down the middle."

### MAKE KDE WITH ALL ON BASELINE
kde_kwargs = {'position_func': vis_ut.find_intersection_with_service_line, 'normalize_density': True, 
              'mix_colors': False, 'mix_alpha': False, 
              'resolution': RESOLUTION_KDE, 'bandwidth': BANDWIDTH_KDE}

nullScn, jointScn = vis_ut.joint_kde(serveShots_sample, 'Ad', **kde_kwargs)
splitScn, xyScn, annPnts = vis_ut.split_kde(serveShots_sample, 'Ad', **kde_kwargs)

itplScns = vs.interpolate_dummy_scenes(nullScn, jointScn, num_frames = 25)

kdeFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    **prmDict)
kdeFrms = overlay(kdeFrms, legFrm, tlPt_leg)

statFrms, animDict = vs.animate_static(jointScn, background_scene = courtScn, static_cam = boxCam, num_frames = 25,
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
                                        start_cam = boxCam, stop_cam = boxCam,
                                        **prmDict_no_illum,
                                        **animDict)
fadeFrms = overlay(fadeFrms, legFrm, tlPt_leg)

splitFrms, animDict = vs.animate_sequence(itplScns, background_scene = xyScn, cumulate_argument = False, 
                                        start_cam = boxCam, stop_cam = boxCam,
                                        annotated_points = annPnts[:-1],
                                        **prmDict)
splitFrms = overlay(splitFrms, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([fadeFrms, splitFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    In red, we chart the distribution of serves in high leverage situations; 
                    in yellow, medium leverage situations; 
                    in green, low leverage situations.
                """

statFrms, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 25,
                                        annotated_points = annPnts[:-1],
                                        **prmDict)

statFrms = overlay(statFrms, legFrm, tlPt_leg)

all_frames.append(statFrms)
script_lines.append(script_line)


script_line =   """ 
                    Over the entire sample, the average low leverage point swings match outcome probabilities by around 2,2%, ...
                """

statFrms_fading = statFrms.dim_progressively(1, 0.5)

loPt, hiPt = annPnts[0], annPnts[2]
loPt_avg = loPt.retext('Mean: 2.2%').right(20).back(1).rethicken(2)
hiPt_avg = hiPt.retext('Mean: 12.9%').right(30).back(3).rethicken(2)

statFrms1, animDict = vs.animate_static(Scene.make_empty(), static_cam = boxCam, num_frames = 50,
                                        annotated_points = [loPt_avg],
                                        animate_over = statFrms_fading[-1],
                                        **prmDict)

jointFrms = BGRFrames.from_frames([statFrms_fading, statFrms1])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    ... Whereas the average high leverage swings match outcomes by 12,9%. 
                """

statFrms2, animDict = vs.animate_static(Scene.make_empty(), static_cam = boxCam, num_frames = 50,
                                        annotated_points = [loPt_avg, hiPt_avg],
                                        animate_over = statFrms_fading[-1],
                                        **prmDict)

all_frames.append(statFrms2)
script_lines.append(script_line)

script_line =   """
                    This makes a typical high leverage point six times more relevant to the match outcome than a typical low leverage point.
                """

from ATMANpy.types.ShapesTennis3D import Arrow
myArr = Arrow.make(loPt_avg.right(7).front(10), hiPt_avg.right(4).back(20), rot_angle = np.pi / 1.75, hat_length = 5)

dashLns = [*myArr[0].dash(25), myArr[1], myArr[2]]

arrFrms, _ = vs.animate_sequence(dashLns, start_cam = boxCam, stop_cam = boxCam,
                                 cumulate_argument = True,
                                 annotated_points = [loPt_avg, hiPt_avg], 
                                 animate_over = statFrms_fading[-1],
                                 **prmDict)

midPt_avg = (loPt_avg + hiPt_avg) / 2
midPt_avg = midPt_avg.annotate('x6').resize(2).back(8).right(7)

statFrms3, animDict = vs.animate_static(dashLns, static_cam = boxCam, num_frames = 50,
                                        annotated_points = [loPt_avg, hiPt_avg, midPt_avg],
                                        animate_over = statFrms_fading[-1],
                                        **prmDict)

jointFrms = BGRFrames.from_frames([arrFrms, statFrms3])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    In Joao Sousa's case, this segmented view is very revealing. 
                """

statFrms_fadeback = vs.interpolate_frames(statFrms3[-1], statFrms[0], 50)
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

fadedFrms, _ = vs.animate_static(loScn, static_cam = boxCam, background_scene = xyScn, num_frames = 50,
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

fadedFrms, _ = vs.animate_static(midScn, static_cam = boxCam, background_scene = xyScn, num_frames = 50,
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

fadedFrms, _ = vs.animate_static(hiScn, static_cam = boxCam, background_scene = xyScn, num_frames = 50,
                                 annotated_points = annPnts_dim, **prmDict)
fadedFrms = overlay(fadedFrms, legFrm, tlPt_leg)

fadeFrms = vs.interpolate_frames(fadeFrms[-1], fadedFrms[0], 25)

jointFrms = BGRFrames.from_frames([fadeFrms, fadedFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

vis_ut.show(all_frames)

script_line =   """
                    In fact, two statistical tests show that there is a highly significant difference between high and low leverage serve placement.
                """

statFrms, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 25,
                                        annotated_points = annPnts,
                                        **prmDict)
statFrms = overlay(statFrms, legFrm, tlPt_leg)

all_frames.append(statFrms)
script_lines.append(script_line)

script_line =   """
                    The Kolmogorov-Smirnov and the Anderson-Darling test both quantify how likely it is that
                    two sets of samples (in our case, low and high leverage serve locations) are generated by the same underlying distribution.
                """
annPnts_curr = [pt for pt in annPnts]

annPnts_curr.pop(2)
statFrms1, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 15,
                                        annotated_points = annPnts_curr,
                                        **prmDict)
statFrms1 = overlay(statFrms1, legFrm, tlPt_leg)

annPnts_curr.pop(1)
statFrms2, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 15,
                                        annotated_points = annPnts_curr,
                                        **prmDict)
statFrms2 = overlay(statFrms2, legFrm, tlPt_leg)

annPnts_curr.pop(0)
statFrms3, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 15,
                                        annotated_points = annPnts_curr,
                                        **prmDict)
statFrms3 = overlay(statFrms3, legFrm, tlPt_leg)

### maybe for KS, AD: high and low are superimposed, while camera moves to direct front view; then we make arrows between the distros.
old_splitScn = splitScn.copy()
new_splitScn = splitScn.copy()

new_splitScn.rastArrs[2] += [0, 120, 0]
new_splitScn.alphaArrs[1] = new_splitScn.alphaArrs[1] * 0

straightCam = boxCam.tilt(-0.15).down(10)

itplScns = vs.interpolate_dummy_scenes(old_splitScn, new_splitScn, 50)

moveFrms, _ = vs.animate_sequence(itplScns, start_cam = boxCam, stop_cam = straightCam, background_scene = xyScn, **prmDict,
                                annotated_points = annPnts_curr, text_cam = boxCam)
moveFrms = overlay(moveFrms, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([statFrms1, statFrms2, statFrms3, moveFrms])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

### arrows to show distance
script_line =   """
                    In this case, the tests argue that it is very unlikely that low and high leverage serve locations are generated by the same distribution.
                    Essentially, Joao Sousa's decision process about where to serve in low leverage situations is different than in high leverage situations.
                """
from ATMANpy.types.Lines import Arrow

aPt1, bPt1 = Point2D((400, 480)), Point2D((400, 560))
aPt2, bPt2 = Point2D((900, 545)), Point2D((900, 630))
aPt3, bPt3 = Point2D((1200, 540)), Point2D((1200, 665))
aPt4, bPt4 = Point2D((1440, 445)), Point2D((1440, 670))

arr_color = const.BGR['bright_grey']

arr1a = Arrow.make(aPt1, bPt1, num_dashes = 10)
arr2a = Arrow.make(aPt2, bPt2, num_dashes = 10)
arr3a = Arrow.make(aPt3, bPt3, num_dashes = 15)
arr4a = Arrow.make(aPt4, bPt4, num_dashes = 25)

arr1b = Arrow.make(bPt1, aPt1, num_dashes = 10)
arr2b = Arrow.make(bPt2, aPt2, num_dashes = 10)
arr3b = Arrow.make(bPt3, aPt3, num_dashes = 15)
arr4b = Arrow.make(bPt4, aPt4, num_dashes = 25)

arrs = [arr1a, arr1b, arr2a, arr2b, arr3a, arr3b, arr4a, arr4b]

arrFrm0 = moveFrms[-1]
arrFrm1 = moveFrms[-1].put(arrs[:2], color = arr_color)
arrFrm2 = moveFrms[-1].put(arrs[:4], color = arr_color)
arrFrm3 = moveFrms[-1].put(arrs[:6], color = arr_color)
arrFrm4 = moveFrms[-1].put(arrs[:8], color = arr_color)
# arrFrm4.show()

arrFrms0 = BGRFrames.from_frame(arrFrm0, 15)
arrFrms1 = BGRFrames.from_frame(arrFrm1, 15)
arrFrms2 = BGRFrames.from_frame(arrFrm2, 15)
arrFrms3 = BGRFrames.from_frame(arrFrm3, 15)
arrFrms4 = BGRFrames.from_frame(arrFrm4, 15)

arrFrms = BGRFrames.from_frames([arrFrms0, arrFrms1, arrFrms2, arrFrms3, arrFrms4])

all_frames.append(arrFrms)
script_lines.append(script_line)

### now move back to previous view
script_line =   """
                    Joao Sousa might well be maximising win probability on high leverage serves 
                    by going for serves out wide to the receiver's backhand almost exclusively;
                """

arrFrms_back = BGRFrames.from_frames([arrFrms4, arrFrms3, arrFrms2, arrFrms1, arrFrms0])

redScn = new_splitScn[0].realpha(0.25).combine(new_splitScn[1]).combine(new_splitScn[2])
grnScn = new_splitScn[0].combine(new_splitScn[1]).combine(new_splitScn[2].realpha(0.25))

itplScns = vs.interpolate_dummy_scenes(new_splitScn, redScn, 50)

itplFrms, _ = vs.animate_sequence(itplScns, start_cam = straightCam, stop_cam = straightCam, background_scene = xyScn, 
                                    annotated_points = annPnts_curr, text_cam = boxCam)
itplFrms = overlay(itplFrms, legFrm, tlPt_leg)

statFrms, _ = vs.animate_static(redScn, static_cam = straightCam, text_cam = boxCam, background_scene = xyScn, annotated_points = annPnts_curr,
                                num_frames = 100)
statFrms = overlay(statFrms, legFrm, tlPt_leg)

jointFrms = BGRFrames.from_frames([arrFrms_back, itplFrms, statFrms])
DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    In low leverage situations, he mixes it up pretty much 50/50, serving down the middle to retain a measure of unpredictability.
                """

itplScns = vs.interpolate_dummy_scenes(redScn, grnScn, 50)

itplFrms, _ = vs.animate_sequence(itplScns, start_cam = straightCam, stop_cam = straightCam, background_scene = xyScn, 
                                annotated_points = annPnts_curr, text_cam = boxCam)
itplFrms = overlay(itplFrms, legFrm, tlPt_leg)

statFrms, _ = vs.animate_static(grnScn, static_cam = straightCam, text_cam = boxCam, background_scene = xyScn, annotated_points = annPnts_curr,
                                num_frames = 100)
statFrms = overlay(statFrms, legFrm, tlPt_leg)

itplScns = vs.interpolate_dummy_scenes(grnScn, old_splitScn, 25)

moveFrms_back, _ = vs.animate_sequence(itplScns, start_cam = straightCam, stop_cam = boxCam, background_scene = xyScn, **prmDict,
                                    annotated_points = annPnts_curr, text_cam = boxCam)
moveFrms_back = overlay(moveFrms_back, legFrm, tlPt_leg)

annPnts_curr = [annPnts[0], *annPnts_curr]
statFrms3, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 10,
                                        annotated_points = annPnts_curr,
                                        **prmDict)
statFrms3 = overlay(statFrms3, legFrm, tlPt_leg)

annPnts_curr = [annPnts[1], *annPnts_curr]
statFrms2, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 10,
                                        annotated_points = annPnts_curr,
                                        **prmDict)
statFrms2 = overlay(statFrms2, legFrm, tlPt_leg)

annPnts_curr = [annPnts[2], *annPnts_curr]
statFrms1, animDict = vs.animate_static(splitScn, background_scene = xyScn, static_cam = boxCam, num_frames = 10,
                                        annotated_points = annPnts_curr,
                                        **prmDict)
statFrms1 = overlay(statFrms1, legFrm, tlPt_leg)


jointFrms = BGRFrames.from_frames([itplFrms, statFrms, moveFrms_back, statFrms3, statFrms2, statFrms1])
DecapStream(jointFrms, 0, 100).show()

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
