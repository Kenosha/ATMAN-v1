
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import scipy

import ATMANpy.auxil.constants as const
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Abstract import Points, Point, read_in_all_tables

from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player, Racquet, Mark, Court
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D, Lines3D
from ATMANpy.types.Overlays import overlay

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from ATMANpy.auxil.court import courtLines
from ATMANpy.auxil.court import back_view_asymmetric_cam, onto_service_box_cam

import ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils as vis_ut

#### READ IN CONFIGS ####
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

### SAVE FOLDER 
DESCRIPTOR = 'intro'
SAVE_FOLDER = 'serve_pressure'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

### PARAMETERS
RESOLUTION_MODIFIER = 3 # 2
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

prmDict_back = {**prmDict}
prmDict_back['background_resolution'] /= 2 ### reduce the background resolution when the full background is in view, to lower the Moiree effect on the net.

STATIC_FRAMES = 100 ### does not matter much once we record: the static frames will be fit to the narration
TRANSITION_FRAMES = 25 

### READ IN DATA
allTable = read_in_all_tables(confs)
allPnts = Points.from_datatable(allTable, split_by_match = False)
firstPnts = allPnts.filter('is_first_serve').enforce_direction('Bot')
deucePnts = firstPnts.filter('is_serve_deuce')

## camera
boxCam = onto_service_box_cam(which = 'Deuce', side = 'Top', dist = 150).tilt(0.15)
backCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Top')

## service line and region
srvLn, indDict = vis_ut.get_service_line_and_indicator('Top', 'Deuce') 
lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

##### SIDE CALCULATATION ######
# How much more important is a point in the high pressure bucket on average than a point in the low pressure bucket?

hiShots = firstPnts.get_serves().filter('get_total_match_score_pressure', is_above = 0.08)
loShots = firstPnts.get_serves().filter('get_total_match_score_pressure', is_below = 0.04)

hi_avg = hiShots.get_total_match_score_pressure().mean()
lo_avg = loShots.get_total_match_score_pressure().mean()
ratio = hi_avg / lo_avg

##### MAKE FOOTNOTES ####

# Markov Model for Score
# -> recursively calculate probability of winning the game given that a point is won versus lost
# -> .................................    winning the set given that the game is won versus lost
# -> .................................    winning the match given that the set is won versus lost.
# and combines these probabilities under the law of total probability to calculate the probability of winning the match if the point is won, and winning the match if the point is lost.
# The score pressure is then simply the difference between these two probabilities, effectively saying: how important is the point at hand, as measured by the impact it has on the probability of winning the whole match.
# the unconditional probability of winning a single point is taken to be 65% for the server and 35% for the returner, which is ATP average, as per: 
    #  "Impact of game elements on tennis match outcome in Wimbledon and Rolland Garros 2009" (Katic et al. 2011)

fn1 = ['The Markov Model for Score Pressure recursively calculates, at a given current scoreline:',
       '    - the probability of winning the game given that the point is won, and given that the point is lost.',
       '    - the probability of winning the set given that the game is won, and given that the game is lost.',
       '    - the probability of winning the match given that the set is won, and given that the set is lost.',
       'It then combines these probabilities under the law of total probability and calculates:',
       '    - the probability of winning the match given that the point is won.',
       '    - the probability of winning the match given that the point is lost.',
       'By subtracting these two quantities, we arrive at the score pressure, which effectively says:',
       'How important is the current point, as measured by the impact it has on the probability of winning the whole match?',
       'The unconditional probability of winning a single point is taken to be 65% for the server and 35% for the returner,',
       'which is ATP average, as per:',
       '"Impact of game elements on tennis match outcome in Wimbledon and Rolland Garros 2009" (Katic et al. 2011)']

fn2 = ['The two statistical tests are the Kolmogorov-Smirnov (KS) Test and the Anderson-Darling (AD) test.',
      'Both the KS and the AD test allow one to check whether two samples derive from the same underlying probability distribution.',
      'To this end, the test statistics they produce are checked two-sided against a significance level a = 5%.',
      'Both tests produce a test statistic, from which a p-value can be calculated.',
      'This p-value indicates the confidence with which the test rejects the null hypothesis,',
      'which is that both samples derive from the same distribution.',
      'If the p-value produced by one of these tests falls below the significance level of 5%,',
      'the null hypothesis that both samples derive from the same distribution is said to be rejected at a = 5%.',
      'In this case, the two samples are said to statistically significantly derive from two different distributions.',
      'If this occurs when checking low-pressure samples against high-pressure samples,',
      'serve placement credibly differs in low-pressure and high-pressure situations.',
      ]

from ATMANpy.types.Overlays import Legend
myLeg1 = Legend.make_empty(legend_dimensions = (1120, 300), borders = False, header_text = '', split_denominator = 60)
fnDict = {'text_color': const.BGR['bright_grey'], 'text_size': 1, 'text_thickness': 1, 'text_offset': 0}
for ln in fn1:
    myLeg1 = myLeg1.add({'text_string': ln, **fnDict})

fnFrm1 = myLeg1.render()
# fnFrm1.show()

fnFrm_rescaled1 = fnFrm1.rescale(0.75)

from ATMANpy.types.Overlays import Legend
myLeg2 = Legend.make_empty(legend_dimensions = (1120, 300), borders = False, header_text = '', split_denominator = 60)
fnDict = {'text_color': const.BGR['bright_grey'], 'text_size': 1, 'text_thickness': 1, 'text_offset': 0}
for ln in fn2:
    myLeg2 = myLeg2.add({'text_string': ln, **fnDict})

fnFrm2 = myLeg2.render()
# fnFrm2.show()

fnFrm_rescaled2 = fnFrm2.rescale(0.75)

###### BEGIN ANIMATION ########
script_lines = []
all_frames = []

##################################
########### JOAO SOUSA ###########
##################################
player_name = 'Joao Sousa'

### COLLECT SHOTS FOR PLAYER
serveShots = deucePnts.get_serves().filter(vis_ut.is_player, player_name = player_name)
print(f'{len(serveShots)} deuce serves for {player_name}')
serveShots = serveShots.filter('get_initial_speed', is_above = 130)
print(f'{len(serveShots)} after speed filtering')

serveShots_clean_bounce = serveShots.filter(vis_ut.is_bounce_in_deuce)
serveShots_net = serveShots.filter('is_in_net')

serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

### PREP LEGEND
handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
legFrm = vis_ut.make_legend_frame(player_name, handedness)

### SCENES 
plyScn = vs.players(serveShots, hit_player_color = const.BGR['light_brown'], rec_player_color = const.BGR['light_brown'])
courtScn = Scene.make_court()
court_and_player_scene = courtScn.combine(plyScn)

### FLYING BALL FEATURES 
line_slomo = 5
ball_bounces_when = 10 * line_slomo
ball_flight_over_when = 17 * line_slomo

script_line = 'In this analysis, we investigate whether servers modify their first serve placement depending on the importance of the point that is about to be played.'
myFrms, animDict = vs.animate_static(None, num_frames = STATIC_FRAMES, static_cam = backCam, background_scene = courtScn, 
                                     **prmDict_back)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'To illuminate our methodology, consider our sample of first serves delivered by Joao Sousa on the deuce side.'
myFrms, animDict = vs.animate_static(plyScn.drop(1), num_frames = STATIC_FRAMES, static_cam = backCam, background_scene = courtScn,
                                     **prmDict_back)

myFrms = vs.blinking(overlay(myFrms, legFrm), myFrms)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'We first register where the ball bounced or hit the net.'
myFrms, animDict = vs.animate_static(plyScn.drop(1), num_frames = STATIC_FRAMES, static_cam = backCam, background_scene = courtScn,
                                     **prmDict_back)

myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = ''
ball_flight_shapes = vs.ball_flight(serveShots_sample, line_interpolation_factor = line_slomo)
myFrms, animDict = vs.animate_sequence(ball_flight_shapes[:ball_bounces_when], background_scene = court_and_player_scene, 
                                        start_cam = backCam, stop_cam = backCam,
                                        cumulate_argument = True, cumulative_foreground_transparence = 0.9, 
                                        **prmDict_back 
                                        )
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### SCATTERS
bounceScn_mono = vs.scatter_markmap(serveShots, showReg = vis_ut.deuceReg_enlarged,
                                    mix_colors = False, one_color = const.BGR['strong_blue'], 
                                    color_func = None)
bounceScn_poly = vs.scatter_markmap(serveShots, showReg = vis_ut.deuceReg_enlarged, 
                                    zero_color = const.BGR['grass_green'], one_color = const.BGR['intense_red'],
                                    color_func = 'get_total_match_score_pressure')

netScn_mono = vs.scatter_markmap(serveShots_net, position_func = 'get_net_crossing_position', flat_along_dimension = 'y',
                                    mix_colors = False, one_color = const.BGR['strong_blue'])
netScn_poly = vs.scatter_markmap(serveShots_net, position_func = 'get_net_crossing_position', flat_along_dimension = 'y',
                                    zero_color = const.BGR['grass_green'], one_color = const.BGR['intense_red'],
                                    color_func = 'get_total_match_score_pressure')

markScn_mono = bounceScn_mono.combine(netScn_mono)
markScn_poly = bounceScn_poly.combine(netScn_poly)

### CONTINUE BALL FLIGHT
script_line = ''
myFrms, animDict = vs.animate_sequence(ball_flight_shapes[ball_bounces_when:ball_flight_over_when], 
                                        background_scene = court_and_player_scene.combine(markScn_mono), 
                                        start_cam = backCam, stop_cam = backCam,
                                        cumulate_argument = True, cumulative_foreground_transparence = 0.9,
                                        **prmDict_back,
                                        **animDict
                                        )
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### ZOOM ONTO BOX
script_line = ''
myFrms, animDict = vs.animate_cam_motion(markScn_mono, background_scene = court_and_player_scene, start_cam = backCam, stop_cam = boxCam, num_frames = 25,
                                            **prmDict_back,
                                            **animDict,
                                            cumulative_foreground_transparence = 0.9)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'We then calculate a point pressure score. The higher the point pressure, the higher the importance of the point in question.'
myFrms, animDict = vs.animate_static(markScn_mono, background_scene = courtScn, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### MULTICOLOR MARKS
script_line = ''
itpl_scenes = vs.interpolate_scenes(markScn_mono, markScn_poly, num_frames = 25)
myFrms, animDict = vs.animate_sequence(itpl_scenes, background_scene = courtScn, start_cam = boxCam, stop_cam = boxCam,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Green indicates low pressure points, red indicates high pressure points.'
myFrms, animDict = vs.animate_static(markScn_poly, background_scene = courtScn, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'To illustrate how the point pressure is calculated, consider the highest pressure point in our sample.'
myFrms, animDict = vs.animate_static(markScn_poly, background_scene = courtScn, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

only_mark_dict = animDict

### FIND HIGHEST PRESSURE POINT
pressArr = serveShots_clean_bounce.get_total_match_score_pressure()
maxShot = serveShots_clean_bounce[int(np.argmax(pressArr))]
hi_coords = maxShot.get_first_bounce_position()
pressPt_bnc_2D = Point2D(hi_coords)

pressPt_lo = Point3D.from_point2D(pressPt_bnc_2D, z = 0)
pressPt_hi = Point3D.from_point2D(pressPt_bnc_2D, z = 30)
myLn = Line3D(pressPt_lo, pressPt_hi, color = const.BGR['intense_red'])

maxMark = Mark.make(pressPt_lo, color = const.BGR['intense_red'])

show_itpl_line_segments = vs.divide_line(myLn, 25, dashed = True)
show_line_and_mark_scenes = [Scene([line_seg, maxMark]) for line_seg in show_itpl_line_segments]
show_line_and_mark_scene = Scene([*show_itpl_line_segments, maxMark])

script_line = ''
myFrms, animDict = vs.animate_sequence(show_line_and_mark_scenes, background_scene = courtScn, cumulate_argument = True, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    **prmDict_back,
                                    **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### ANNOTATE HIGHEST PRESSURE POINT
scoreStr = maxShot.get_score_string()
p_w, p_l = maxShot.get_total_match_win_probabilities()
pressure = maxShot.get_total_match_score_pressure()

text_lines = [f'Score: {scoreStr}',
                f'Match win prob. if point is won: {round(p_w * 100)}%',
                f'Match win prob. if point is lost: {round(p_l * 100)}%',
                f'Score Pressure: {round(pressure * 100)}%']
annotPt = pressPt_hi.left(2).annotate('\n'.join(text_lines)).recolor(const.BGR['intense_red'])

script_line = 'The score is 3-5 in the deciding third-set tiebreak. According to a simple Markov Model, the probability of winning the match if the point is won is 36%. If the point is lost, the probability of winning the match drops to 7%. The given point can therefore swing the win probability by 29%, which we take to be the measure of pressure at this score.'
myFrms, animDict = vs.animate_static(show_line_and_mark_scene, background_scene = courtScn, annotated_points = [annotPt],
                                        static_cam = boxCam,
                                        **prmDict_back,
                                        **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

tlPt = Point2D((myFrms.cols() - fnFrm_rescaled1.cols - 1, myFrms.rows() - fnFrm_rescaled1.rows - 1))
script_line = 'To find out more about the model used to calculate score pressure, pause the video now and consult the footnote.'
myFrms, animDict = vs.animate_static(show_line_and_mark_scene, background_scene = courtScn, annotated_points = [annotPt],
                                        static_cam = boxCam,
                                        **prmDict_back,
                                        **animDict)
myFrms = overlay(myFrms, legFrm)
myFrms = overlay(myFrms, fnFrm_rescaled1, top_left_point = tlPt)
all_frames.append(myFrms)
script_lines.append(script_line)

animDict = only_mark_dict

### FLIGHT LINE
hit_coords = maxShot.get_ball_hit_position()
pressPt_hit_2D = Point2D(hit_coords[:2])

flightLn_2D = Line(pressPt_hit_2D, pressPt_bnc_2D)
interPt_2D = flightLn_2D.weak_intersect(srvLn)

pressPt_hit = Point3D.from_point2D(pressPt_hit_2D, z = 0)
interPt = Point3D.from_point2D(interPt_2D, z = 0)
flightLn = Line3D(pressPt_hit, interPt, color = const.BGR['ball_yellow'])    

flight_itpl_line_segments = vs.divide_line(flightLn, 25, dashed = True)
flight_line_and_mark_scenes = [Scene([line_seg, maxMark]) for line_seg in flight_itpl_line_segments]
flight_line_and_mark_scene = Scene([*flight_itpl_line_segments, maxMark])

script_line = ''
myFrms, animDict = vs.animate_sequence(flight_line_and_mark_scenes, background_scene = courtScn, cumulate_argument = True, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    cumulative_foreground_transparence = 0.95,
                                    **prmDict_back,
                                    **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'We now consider where the ball crosses the service line by projecting the bounce location along the flight path of the ball [cont.]'
myFrms, animDict = vs.animate_static(flight_line_and_mark_scene, background_scene = courtScn, static_cam = boxCam,
                                        **prmDict_back, 
                                        **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### PROJECT MARK ONTO SERVICE LINE
maxMark_on_line = Mark.make(interPt, color = const.BGR['intense_red'])
flight_line_and_mark_scene_end = Scene([*flight_itpl_line_segments, maxMark_on_line])

mark_moving_scenes = vs.interpolate_scenes(flight_line_and_mark_scene, flight_line_and_mark_scene_end, num_frames = 25)

animDict['previous_foreground_scene'] = Scene([shape for shape in animDict['previous_foreground_scene'].shapes if shape != maxMark])

script_line = ''
myFrms, animDict = vs.animate_sequence(mark_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    **prmDict_back,
                                    **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = '[cont.] and repeat this for all other balls, both those which bounced and those in the net.'
myFrms, animDict = vs.animate_static(mark_moving_scenes[-1], background_scene = courtScn, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                    **prmDict_back,
                                    **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### FADE OTHER MARKS BACK IN
animDict['previous_foreground_scene'] = Scene([shape for shape in animDict['previous_foreground_scene'].shapes if isinstance(shape, Mark)])

script_line = ''
myFrms, animDict = vs.animate_sequence([[]] * 25, background_scene = courtScn, cumulate_argument = False,
                                        cumulative_foreground_transparence = 1.2, start_cam = boxCam, stop_cam = boxCam,
                                        **prmDict_back,
                                        **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

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

markScn_poly_box = Scene([mark if mark != maxMark else maxMark_on_line for mark in markScn_poly.shapes])

all_marks_moving_scenes = vs.interpolate_scenes(markScn_poly_box, markScn_poly_moved, 25)

script_line = ''
myFrms, animDict = vs.animate_sequence(all_marks_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Now that all serves are projected onto the service line along the flight path, we can calculate a distribution of serve placement.'
myFrms, animDict = vs.animate_static(markScn_poly_moved, background_scene = courtScn, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### MAKE KDE WITH ALL ON BASELINE
lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

joint_kde_null_scene = vs.kde_hillmap(serveShots_sample, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line,
                                      bandwidth = BANDWIDTH_KDE, 
                                resolution = RESOLUTION_KDE, normalize_density = True, max_height = -0.5, one_color = const.BGR['strong_blue'])
joint_kde_scene = vs.kde_hillmap(serveShots_sample, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line, 
                                 bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['strong_blue'])

joint_itpl_scenes = vs.interpolate_dummy_scenes(joint_kde_null_scene, joint_kde_scene, num_frames = 25)

script_line = ''
myFrms, animDict = vs.animate_sequence(joint_itpl_scenes, background_scene = courtScn, cumulate_argument = False, 
                                    start_cam = boxCam, stop_cam = boxCam,
                                    **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'It appears that Joao Sousa generally prefers the serve out wide on the deuce side, employing it about twice as often as the serve down the T.'
myFrms, animDict = vs.animate_static(joint_kde_scene, background_scene = courtScn, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Now, we split the distribution according to the importance of the point at hand.'
myFrms, animDict = vs.animate_static(joint_kde_scene, background_scene = courtScn, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        **prmDict_back)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

### MAKE KDES WITH SPLIT CATEGORIES
lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

low_kde_scene = vs.kde_hillmap(lowShots, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
mid_kde_scene = vs.kde_hillmap(midShots, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line, 
							   bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
high_kde_scene = vs.kde_hillmap(hiShots, showReg = deucelineReg, position_func = vis_ut.find_intersection_with_service_line, 
								bandwidth = BANDWIDTH_KDE,
                                resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

mid_kde_scene.rastArrs[0] += [0, 60, 0]
high_kde_scene.rastArrs[0] += [0, 120, 0]

joint_kde_scenes = joint_kde_scene.combine(joint_kde_scene).combine(joint_kde_scene)
split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

split_itpl_scenes = vs.interpolate_dummy_scenes(joint_kde_scenes, split_kde_scenes, num_frames = 25)

### SERVICE BOX GRAPH
width = courtLines.see_dimensions()['otherline_width']
x_ax, y_ax = srvLn, courtLines['Top Center Line']
x_ax_3D = Region3D.from_line3D(Line3D.from_line2D(x_ax, z = 0), width = width, dim = 'y')
y_ax_3D = Region3D.from_line3D(Line3D.from_line2D(y_ax, z = 0), width = width, dim = 'x')
x_y_scene = Scene([x_ax_3D, y_ax_3D])

origPt = x_ax.weak_intersect(y_ax)
origPt_3D = Point3D.from_point2D(origPt, z = 0)

loPt = origPt_3D.right(25).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
midPt = origPt_3D.back(70).right(35).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
hiPt = origPt_3D.back(150).right(37).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

tPt = origPt_3D.left(5).front(10).annotate('Down the Middle')    
wPt = origPt_3D.left(100).front(10).annotate('Out Wide')

######## DISTRO TESTS #########
diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
jntStr = vis_ut.make_stat_string(lowShots, hiShots)
statPt = diagPt.annotate(jntStr).back(110).left(40)

### FADE OUT COURT ###
court_fadeout_scenes = vs.interpolate_scenes(courtScn, courtScn.transparify(0), num_frames = 25)
prmDict_no_illum = {k: v for k, v in prmDict.items()}
prmDict_no_illum['argument_luminosity'] = 1
prmDict_no_illum['argument_resolution'] = prmDict_back['background_resolution']

script_line = ''
myFrms, animDict = vs.animate_sequence(court_fadeout_scenes, background_scene = x_y_scene, cumulate_argument = False, 
                                       start_cam = boxCam, stop_cam = boxCam,
                                       **prmDict_no_illum,
                                       **animDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = ''
myFrms, animDict = vs.animate_sequence(split_itpl_scenes, background_scene = x_y_scene, cumulate_argument = False, 
                                        start_cam = boxCam, stop_cam = boxCam,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'In red, we chart the distribution of serves in high pressure situations; in yellow, medium pressure situations; in green, low pressure situations.'
myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Over the entire sample, the average low pressure point swings match outcome probabilities by around 2,2%, whereas the average high pressure swings match outcomes by 12,9%. making points in the high pressure category on average six times more important than points in the low pressure category.'
myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Beyond a slight reduction in serves to the body in situations of high importance. it appears that Joao Sousa does not significantly alter the placement of his first serves on the deuce side depending on pressure.'
myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'Indeed, two statistical tests find no significant difference in the distribution of serves in high and low pressure situations.'
myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt, statPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

tlPt = Point2D((myFrms.cols() - fnFrm_rescaled2.cols - 1, myFrms.rows() - fnFrm_rescaled2.rows - 1))
script_line = 'To find out more about the two statistical tests employed, and how to interpret their results, feel free to pause the video and consult the footnote.'
myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt, statPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
myFrms = overlay(myFrms, fnFrm_rescaled2, top_left_point = tlPt)
all_frames.append(myFrms)
script_lines.append(script_line)

script_line = 'We therefore move on to more interesting cases.'
myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = STATIC_FRAMES,
                                        annotated_points = [loPt, midPt, hiPt, tPt, wPt, statPt],
                                        **prmDict)
myFrms = overlay(myFrms, legFrm)
all_frames.append(myFrms)
script_lines.append(script_line)

assert len(script_lines) == len(all_frames)

jntFrms = BGRFrames.from_frames(all_frames)
DecapStream(jntFrms, 0, 100).show(ms_between_frames = 25)

### Saving the separate MultiFrame objects and the Joint Script
from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

# b) Save segments
for idx, subFrms in enumerate(all_frames):
    subFrms.save(fullPath, f'seg_{idx}', save_as = 'mp4', lossless = True)

# c) Save script
with open(os.path.join(fullPath, 'script.txt'), mode = 'w') as f:
    f.writelines([l + '\n' for l in script_lines])

