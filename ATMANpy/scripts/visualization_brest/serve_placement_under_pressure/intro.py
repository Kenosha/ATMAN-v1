
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import scipy

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Abstract import Points, Point, read_in_all_tables

from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player, Racquet, Mark
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D, Lines3D

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from ATMANpy.auxil.court import courtLines

def get_service_line_and_indicator(side, which):
    srvReg = courtLines.get_service_box(side = side, which = which)
    
    if side == 'Bot':
        srvLn = srvReg['Bottom']
    elif side == 'Top':
        srvLn = srvReg['Top']

    ### get indicator -> which is wide, which is down the middle
    srvLn.leftPt, srvLn.rightPt
    center_x, center_y = courtLines.get_center_point().coords

    if srvLn.leftPt.x == center_x:
        indicator = {'middle': srvLn.leftPt,
                        'wide': srvLn.rightPt}
    if srvLn.rightPt.x == center_x:
        indicator = {'middle': srvLn.rightPt,
                        'wide': srvLn.leftPt}

    return srvLn, indicator

def find_intersection_with_service_line(myShot):
    from ATMANpy.types.Points import Point2D
    from ATMANpy.types.Lines import Line

    assert myShot.is_serve()
    side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
    which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
    srvLn, _ = get_service_line_and_indicator(side, which)

    hit_pos = myShot.get_ball_hit_position()
    if myShot.is_in_net():
        x, y, _ = myShot.get_ball_last_position()
    else:
        x, y = myShot.get_first_bounce_position()
    bnc_pos = x, y
    
    if bnc_pos == (None, None):
        return (None, None)

    flightLn = Line(Point2D(tuple(hit_pos)), Point2D(tuple(bnc_pos)))

    interPt = flightLn.weak_intersect(srvLn)
    return interPt.coords

def find_lambda_of_service_line(myShot):
    from ATMANpy.types.Points import Point2D
    from ATMANpy.types.Lines import Line

    assert myShot.is_serve()
    side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
    which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
    srvLn, indicator = get_service_line_and_indicator(side, which)

    hit_pos, bnc_pos = myShot.get_ball_hit_position(), myShot.get_first_bounce_position()
    if bnc_pos == (None, None):
        return None

    flightLn = Line(Point2D(tuple(hit_pos)), Point2D(tuple(bnc_pos)))

    interPt = flightLn.weak_intersect(srvLn)

    pt_x = interPt.x
    wide_x, middle_x = indicator['wide'].x, indicator['middle'].x
    middle_dist = middle_x - pt_x
    extent_to_which_the_ball_is_wide = middle_dist / (middle_x - wide_x)
    return extent_to_which_the_ball_is_wide

def regress_lambda_on_pressure(lambdaArr_filt, pressArr_filt):    
    import statsmodels.api as sm

    X = np.stack([pressArr_filt, np.ones(len(pressArr_filt))]).T
    ols = sm.OLS(lambdaArr_filt, X)
    ols_result = ols.fit()
    return ols_result


def make_stat_string(lowShots, hiShots, alpha = 0.05):
    ######## Tests for similarity of distribution #########
    low_x = np.array([find_intersection_with_service_line(aShot) for aShot in lowShots], dtype = float)[:, 0]
    hi_x = np.array([find_intersection_with_service_line(aShot) for aShot in hiShots], dtype = float)[:, 0]

    ks_res = scipy.stats.kstest(low_x, hi_x)           ### kolmogorov-smirnov test (2-sample)
    ad_res = scipy.stats.anderson_ksamp([low_x, hi_x]) ### anderson darling test (2-sample)
    ks_p, ad_p = ks_res.pvalue, ad_res.pvalue

    if ks_p >= alpha and ad_p >= alpha:
        textStr = f'High-Pressure and Low-Pressure distributions \n can not be shown to be statistically different at a = {alpha}.'
    elif ks_p < alpha and ad_p < alpha:
        textStr = f'High-Pressure and Low-Pressure distributions \n are statistically different at a = {alpha}.'
    elif ks_p < alpha and ad_p >= alpha:
        textStr = f'High-Pressure and Low-Pressure distributions \n are statistically different at a = {alpha} \n only according to the Kolmogorov-Smirnov test.'
    elif ks_p >= alpha and ad_p < alpha:
        textStr = f'High-Pressure and Low-Pressure distributions \n are statistically different at a = {alpha} \n only according to the Anderson-Darling test.'

    ksStr = f'Kolmogorov-Smirnov 2-Sample p-value: {round(ks_p, 3)}'
    adStr = f'Anderson-Darling 2-Sample p-value: {round(ad_p, 3)}'
    jntStr = '\n'.join([textStr, '', ksStr, adStr])

    return jntStr


RESOLUTION_MODIFIER = 0.2 # 2
RESOLUTION_BACKGROUND = 0.5 * RESOLUTION_MODIFIER
RESOLUTION_OBJECTS = 1 * RESOLUTION_MODIFIER
RESOLUTION_KDE = 2 * RESOLUTION_MODIFIER
MIX_COLORS_KDE = True
BANDWIDTH_KDE = 30
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


boxReg = courtLines.make().add_singles_netline().get_service_box(which = 'Deuce', side = 'Top')
boxReg_enlarged = boxReg.scale(1.25)

def is_player(myShot, player_name = 'Joao Sousa'):
        return myShot.hitPlayer.get_name() == player_name

def is_bounce_in_region(myShot, myReg = boxReg_enlarged):
        coords = myShot.get_first_bounce_position()
        if coords == (None, None):
            return False
        else:
            bouncePt = Point2D(coords)
            return bouncePt.within(myReg)

def main(confs, name = None):
    
    from ATMANpy.auxil.court import back_view_asymmetric_cam, onto_service_box_cam

    ### READ IN DATA
    allTable = read_in_all_tables(confs)
    allPnts = Points.from_datatable(allTable, split_by_match = False)

    deucePnts = allPnts.filter('is_serve_deuce').filter('is_first_serve').enforce_direction('Bot')
    # adPnts = allPnts.filter('is_serve_ad').enforce_direction('Bot')
     
    srvLn, indDict = get_service_line_and_indicator('Top', 'Deuce')

    all_frames = []

    #### JOAO SOUSA #####
    serveShots = deucePnts.get_serves().filter(is_player, player_name = 'Joao Sousa')
    print(f'{len(serveShots)} deuce serves for Joao Sousa')
    serveShots = serveShots.filter('get_initial_speed', is_above = 130)
    print(f'{len(serveShots)} after speed filtering')

    serveShots_clean_bounce = serveShots.filter(is_bounce_in_region, myReg = boxReg_enlarged)
    serveShots_net = serveShots.filter('is_in_net')

    serveShots_sample = serveShots_clean_bounce.join(serveShots_net)


    backCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Top')
    boxCam = onto_service_box_cam(which = 'Deuce', side = 'Top', dist = 150).tilt(0.15)
    plyScn = vs.players(serveShots)
    courtScn = Scene.make_court()

    court_and_player_scene = courtScn.combine(plyScn)

    line_slomo = 5
    ball_bounces_when = 10 * line_slomo
    ball_flight_over_when = 17 * line_slomo

    ball_flight_shapes = vs.ball_flight(serveShots_sample, line_interpolation_factor = line_slomo)
    myFrms, animDict = vs.animate_sequence(ball_flight_shapes[:ball_bounces_when], background_scene = court_and_player_scene, 
                                           start_cam = backCam, stop_cam = backCam,
                                            cumulate_argument = True, cumulative_foreground_transparence = 0.9, 
                                            **prmDict 
                                            )
    all_frames.append(myFrms)

    bounceScn_mono = vs.scatter_markmap(serveShots, showReg = boxReg_enlarged,
                                        mix_colors = False, one_color = const.BGR['strong_blue'], 
                                        color_func = None)
    bounceScn_poly = vs.scatter_markmap(serveShots, showReg = boxReg_enlarged, 
                                        zero_color = const.BGR['grass_green'], one_color = const.BGR['intense_red'],
                                        color_func = 'get_total_match_score_pressure')
    
    netScn_mono = vs.scatter_markmap(serveShots_net, position_func = 'get_net_crossing_position', flat_along_dimension = 'y',
                                     mix_colors = False, one_color = const.BGR['strong_blue'])
    netScn_poly = vs.scatter_markmap(serveShots_net, position_func = 'get_net_crossing_position', flat_along_dimension = 'y',
                                     zero_color = const.BGR['grass_green'], one_color = const.BGR['intense_red'],
                                     color_func = 'get_total_match_score_pressure')
    
    markScn_mono = bounceScn_mono.combine(netScn_mono)
    markScn_poly = bounceScn_poly.combine(netScn_poly)

    myFrms, animDict = vs.animate_sequence(ball_flight_shapes[ball_bounces_when:ball_flight_over_when], 
                                           background_scene = court_and_player_scene.combine(markScn_mono), 
                                           start_cam = backCam, stop_cam = backCam,
                                            cumulate_argument = True, cumulative_foreground_transparence = 0.9,
                                            **prmDict,
                                            **animDict
                                            )
    all_frames.append(myFrms)


    # myFrms, animDict = vs.animate_sequence([], background_scene = courtScn.combine(markScn_mono), start_cam = backCam, stop_cam = backCam,
    #                                         cumulate_argument = True, cumulative_foreground_transparence = 0.9,
    #                                         **prmDict,
    #                                         **animDict
    #                                         )
    # all_frames.append(myFrms)

    myFrms, animDict = vs.animate_cam_motion(markScn_mono, background_scene = court_and_player_scene, start_cam = backCam, stop_cam = boxCam, num_frames = 25,
                                                **prmDict,
                                                **animDict,
                                                cumulative_foreground_transparence = 0.9)
    all_frames.append(myFrms)


    myFrms, animDict = vs.animate_static(markScn_mono, background_scene = courtScn, static_cam = boxCam, num_frames = 25,
                                            **prmDict)
    all_frames.append(myFrms)


    itpl_scenes = vs.interpolate_scenes(markScn_mono, markScn_poly, num_frames = 25)
    myFrms, animDict = vs.animate_sequence(itpl_scenes, background_scene = courtScn, start_cam = boxCam, stop_cam = boxCam,
                                           **prmDict)
    all_frames.append(myFrms)

    ### TODO: color shift by interpolating    
    myFrms, animDict = vs.animate_static(markScn_poly, background_scene = courtScn, static_cam = boxCam, num_frames = 25,
                                            **prmDict)
    all_frames.append(myFrms)

    only_mark_dict = animDict

    ### get the highest pressure point, and make a line upwards
    from ATMANpy.types.Points import Point2D
    pressArr = serveShots_clean_bounce.get_total_match_score_pressure()
    maxShot = serveShots_clean_bounce[np.argmax(pressArr)]
    hi_coords = maxShot.get_first_bounce_position()
    pressPt_bnc_2D = Point2D(hi_coords)

    pressPt_lo = Point3D.from_point2D(pressPt_bnc_2D, z = 0)
    pressPt_hi = Point3D.from_point2D(pressPt_bnc_2D, z = 30)
    myLn = Line3D(pressPt_lo, pressPt_hi, color = const.BGR['intense_red'])

    maxMark = Mark.make(pressPt_lo, color = const.BGR['intense_red'])
    
    show_itpl_line_segments = vs.divide_line(myLn, 25, dashed = True)
    show_line_and_mark_scenes = [Scene([line_seg, maxMark]) for line_seg in show_itpl_line_segments]
    show_line_and_mark_scene = Scene([*show_itpl_line_segments, maxMark])

    myFrms, animDict = vs.animate_sequence(show_line_and_mark_scenes, background_scene = courtScn, cumulate_argument = True, 
                                        start_cam = boxCam, stop_cam = boxCam,
                                        **prmDict,
                                        **animDict)
    all_frames.append(myFrms)

    ### annotate the highest point    
    scoreStr = maxShot.get_score_string()
    p_w, p_l = maxShot.get_total_match_win_probabilities()
    pressure = maxShot.get_total_match_score_pressure()

    text_lines = [f'Score: {scoreStr}',
                    f'Win prob. if point is won: {round(p_w * 100)}%',
                    f'Win prob. if point is lost: {round(p_l * 100)}%',
                    f'Score Pressure: {round(pressure * 100)}%']
    annotPt = pressPt_hi.left(2).annotate('\n'.join(text_lines)).recolor(const.BGR['intense_red'])

    myFrms, animDict = vs.animate_static(show_line_and_mark_scene, background_scene = courtScn, annotated_points = [annotPt],
                                         static_cam = boxCam,
                                         **prmDict,
                                         **animDict)
    all_frames.append(myFrms)

    animDict = only_mark_dict
    # animDict['previous_foreground_scene'] = animDict['previous_foreground_scene'].transparify(0.5)
    # ### TODO: maybe transparify the other marks at this point!

    ### line through the highest point, onto service line
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

    myFrms, animDict = vs.animate_sequence(flight_line_and_mark_scenes, background_scene = courtScn, cumulate_argument = True, 
                                        start_cam = boxCam, stop_cam = boxCam,
                                        cumulative_foreground_transparence = 0.95,
                                        **prmDict,
                                        **animDict)
    all_frames.append(myFrms)

    myFrms, animDict = vs.animate_static(flight_line_and_mark_scene, background_scene = courtScn, static_cam = boxCam,
                                         **prmDict, 
                                         **animDict)
    all_frames.append(myFrms)

    maxMark_on_line = Mark.make(interPt, color = const.BGR['intense_red'])
    flight_line_and_mark_scene_end = Scene([*flight_itpl_line_segments, maxMark_on_line])
    
    mark_moving_scenes = vs.interpolate_scenes(flight_line_and_mark_scene, flight_line_and_mark_scene_end, num_frames = 25)
    
    # # animDict = only_mark_dict
    animDict['previous_foreground_scene'] = Scene([shape for shape in animDict['previous_foreground_scene'].shapes if shape != maxMark])

    myFrms, animDict = vs.animate_sequence(mark_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                        start_cam = boxCam, stop_cam = boxCam,
                                        **prmDict,
                                        **animDict)
    all_frames.append(myFrms)

    myFrms, animDict = vs.animate_static(mark_moving_scenes[-1], background_scene = courtScn, static_cam = boxCam, num_frames = 25,
                                        **prmDict,
                                        **animDict)
    all_frames.append(myFrms)

    ### FADE OTHER MARKS BACK IN
    animDict['previous_foreground_scene'] = Scene([shape for shape in animDict['previous_foreground_scene'].shapes if isinstance(shape, Mark)])

    myFrms, animDict = vs.animate_sequence([[]] * 25, background_scene = courtScn, cumulate_argument = False,
                                           cumulative_foreground_transparence = 1.2, start_cam = boxCam, stop_cam = boxCam,
                                           **prmDict,
                                           **animDict)
    all_frames.append(myFrms)

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

    myFrms, animDict = vs.animate_sequence(all_marks_moving_scenes, background_scene = courtScn, cumulate_argument = False, 
                                        start_cam = boxCam, stop_cam = boxCam,
                                        **prmDict)
    all_frames.append(myFrms)

    myFrms, animDict = vs.animate_static(markScn_poly_moved, background_scene = courtScn, static_cam = boxCam, num_frames = 25,
                                         **prmDict)
    all_frames.append(myFrms)

    ### MAKE KDE WITH ALL ON BASELINE
    from ATMANpy.types.ShapesTennis3D import Court 
    lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
    deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!
    
    joint_kde_null_scene = vs.kde_hillmap(serveShots_sample, showReg = deucelineReg, position_func = find_intersection_with_service_line, 
                                 resolution = RESOLUTION_KDE, normalize_density = True, max_height = -0.5, one_color = const.BGR['strong_blue'])
    joint_kde_scene = vs.kde_hillmap(serveShots_sample, showReg = deucelineReg, position_func = find_intersection_with_service_line, 
                                 resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['strong_blue'])

    joint_itpl_scenes = vs.interpolate_dummy_scenes(joint_kde_null_scene, joint_kde_scene, num_frames = 25)

    myFrms, animDict = vs.animate_sequence(joint_itpl_scenes, background_scene = courtScn, cumulate_argument = False, 
                                        start_cam = boxCam, stop_cam = boxCam,
                                        **prmDict)
    all_frames.append(myFrms)

    myFrms, animDict = vs.animate_static(joint_kde_scene, background_scene = courtScn, static_cam = boxCam, num_frames = 50,
                                         **prmDict)
    all_frames.append(myFrms)

    ### MAKE KDES WITH SPLIT CATEGORIES
    lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = 0.04)
    midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
    hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = 0.08)

    low_kde_scene = vs.kde_hillmap(lowShots, showReg = deucelineReg, position_func = find_intersection_with_service_line, 
                                    resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
    mid_kde_scene = vs.kde_hillmap(midShots, showReg = deucelineReg, position_func = find_intersection_with_service_line, 
                                    resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
    high_kde_scene = vs.kde_hillmap(hiShots, showReg = deucelineReg, position_func = find_intersection_with_service_line, 
                                    resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

    mid_kde_scene.rastArrs[0] += [0, 40, 0]
    high_kde_scene.rastArrs[0] += [0, 80, 0]

    joint_kde_scenes = joint_kde_scene.combine(joint_kde_scene).combine(joint_kde_scene)
    split_kde_scenes = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

    split_itpl_scenes = vs.interpolate_dummy_scenes(joint_kde_scenes, split_kde_scenes, num_frames = 25)

    ## make courtlines with just service line and deuce service line -> this way, we can make it look more like a graph.
    width = courtLines.see_dimensions()['otherline_width']
    x_ax, y_ax = srvLn, courtLines['Top Center Line']
    x_ax_3D = Region3D.from_line3D(Line3D.from_line2D(x_ax, z = 0), width = width, dim = 'y')
    y_ax_3D = Region3D.from_line3D(Line3D.from_line2D(y_ax, z = 0), width = width, dim = 'x')
    x_y_scene = Scene([x_ax_3D, y_ax_3D])

    origPt = x_ax.weak_intersect(y_ax)
    origPt_3D = Point3D.from_point2D(origPt, z = 0)

    loPt = origPt_3D.right(20).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
    midPt = origPt_3D.back(50).right(25).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
    hiPt = origPt_3D.back(100).right(27).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

    tPt = origPt_3D.left(5).front(10).annotate('Down the Middle')    
    wPt = origPt_3D.left(100).front(10).annotate('Out Wide')
    
    ######## Tests for similarity of distribution #########
    diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
    jntStr = make_stat_string(lowShots, hiShots)
    statPt = diagPt.annotate(jntStr).back(110)

    ### animate
    myFrms, animDict = vs.animate_sequence(split_itpl_scenes, background_scene = x_y_scene, cumulate_argument = False, 
                                            start_cam = boxCam, stop_cam = boxCam,
                                            annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                            **prmDict)
    all_frames.append(myFrms)

    myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = 50,
                                         annotated_points = [loPt, midPt, hiPt, tPt, wPt, statPt],
                                         **prmDict)
    all_frames.append(myFrms)

    jntFrms = BGRFrames.from_frames(all_frames)
    DecapStream(jntFrms, 0, 100).show(ms_between_frames = 25)

    myFrms[-1].show()
    
if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

    name = ppl._name

    main(confs, name = None)

    
    