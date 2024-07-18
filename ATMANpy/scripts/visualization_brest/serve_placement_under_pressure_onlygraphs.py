
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

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

def find_intersection_with_service_line(myShot, second_position_func = 'get_first_bounce_position'):
    from ATMANpy.types.Points import Point2D
    from ATMANpy.types.Lines import Line

    assert myShot.is_serve()
    side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
    which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
    srvLn, _ = get_service_line_and_indicator(side, which)

    hit_pos, bnc_pos = myShot.get_ball_hit_position(), getattr(myShot, second_position_func)()
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

RESOLUTION_MODIFIER = 0.5 # 2
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
    # serveShots_clean_bounce = serveShots.filter(is_bounce_in_region, myReg = boxReg_enlarged)

    def bounce_intersection_func(myShot):
         return find_intersection_with_service_line(myShot, second_position_func = 'get_first_bounce_position')
    
    def smart_intersection_func(myShot):
        x1, y1 = find_intersection_with_service_line(myShot, second_position_func = 'get_ball_last_position')
        x2, y2 = find_intersection_with_service_line(myShot, second_position_func = 'get_ball_next_position')
    
    for idx, shot in enumerate(serveShots):
        if shot.is_in_net():
            print(idx)
            # print(shot.shotTraj.trajFrm)

        x, y, z = shot.get_ball_next_position()
        if np.isnan(x):
            print(idx)
            print(shot.shotTraj.trajFrm)

         print(idx, shot.get_ball_next_position())

    for shot in serveShots[:17]:
        print(shot.get_score_string())
        print(shot.get_frames())
        shot.stream(cap).show(ms_between_frames = 50)

    serveShots[27].get_ball_last_position()

    backCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Top')
    boxCam = onto_service_box_cam(which = 'Deuce', side = 'Top', dist = 150).tilt(0.15)
    plyScn = vs.players(serveShots)
    courtScn = Scene.make_court()

    ### MAKE KDE WITH ALL ON BASELINE
    from ATMANpy.types.ShapesTennis3D import Court 
    lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
    deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!
    
    joint_kde_null_scene = vs.kde_hillmap(serveShots, showReg = deucelineReg, position_func = intersection_func, 
                                 resolution = RESOLUTION_KDE, normalize_density = True, max_height = -0.5, one_color = const.BGR['strong_blue'])
    joint_kde_scene = vs.kde_hillmap(serveShots, showReg = deucelineReg, position_func = intersection_func, 
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
    lowShots = serveShots.filter('get_total_match_score_pressure', is_below = 0.04)
    midShots = serveShots.filter('get_total_match_score_pressure', is_above = 0.04, is_below = 0.08)
    hiShots = serveShots.filter('get_total_match_score_pressure', is_above = 0.08)

    low_kde_scene = vs.kde_hillmap(lowShots, showReg = deucelineReg, position_func = intersection_func, 
                                    resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
    mid_kde_scene = vs.kde_hillmap(midShots, showReg = deucelineReg, position_func = intersection_func, 
                                    resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
    high_kde_scene = vs.kde_hillmap(hiShots, showReg = deucelineReg, position_func = intersection_func, 
                                    resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

    mid_kde_scene.rastArrs[0] += [0, 30, 0]
    high_kde_scene.rastArrs[0] += [0, 60, 0]

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

    loPt = origPt_3D.right(25).annotate('Low Pressure \n (Point Impact <4%)').recolor(const.BGR['grass_green'])
    midPt = origPt_3D.back(40).right(25).annotate('Medium Pressure \n (Point Impact 4-8%)').recolor(const.BGR['soft_yellow'])
    hiPt = origPt_3D.back(80).right(25).annotate('High Pressure \n (Point Impact >8%)').recolor(const.BGR['intense_red'])

    tPt = origPt_3D.left(5).front(10).annotate('Down the Middle')    
    wPt = origPt_3D.left(100).front(10).annotate('Out Wide')

    myFrms, animDict = vs.animate_sequence(split_itpl_scenes, background_scene = x_y_scene, cumulate_argument = False, 
                                            start_cam = boxCam, stop_cam = boxCam,
                                            annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                            **prmDict)
    all_frames.append(myFrms)

    myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = 50,
                                         annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                         **prmDict)
    all_frames.append(myFrms)

    jntFrms = BGRFrames.from_frames(all_frames)
    DecapStream(jntFrms, 0, 100).show(ms_between_frames = 50)

    
if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

    name = ppl._name

    main(confs, name = None)

    
    