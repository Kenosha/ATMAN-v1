
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
from ATMANpy.types.Overlays import Legend, overlay

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from ATMANpy.auxil.court import courtLines

graphPath = ut.get_folder_struct('graphs')['serve_under_pressure']
sub_graphPath = os.path.join(graphPath, 'Ad')

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

RESOLUTION_MODIFIER = 2 # 2
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

boxReg = courtLines.make().add_singles_netline().get_service_box(which = 'Ad', side = 'Top')
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

    adPnts = allPnts.filter('is_serve_ad').filter('is_first_serve').enforce_direction('Bot')
    deucePnts = allPnts.filter('is_serve_deuce').filter('is_first_serve').enforce_direction('Bot')

    srvLn, indDict = get_service_line_and_indicator('Top', 'Ad')
    players = list(set([pnt.get_srv_player_name() for pnt in adPnts]))

    for player_name in players:
        all_frames = []

        ### COLLECT SHOTS FOR PLAYER
        serveShots = adPnts.get_serves().filter(is_player, player_name = player_name)
        print(f'{len(serveShots)} ad serves for {player_name}')
        serveShots = serveShots.filter('get_initial_speed', is_above = 130)
        print(f'{len(serveShots)} after speed filtering')

        serveShots_clean_bounce = serveShots.filter(is_bounce_in_region, myReg = boxReg_enlarged)
        serveShots_net = serveShots.filter('is_in_net')

        serveShots_sample = serveShots_clean_bounce.join(serveShots_net)

        ### INITIALISE CAMERAS ###
        backCam = back_view_asymmetric_cam(which = 'Ad', side = 'Top')
        boxCam = onto_service_box_cam(which = 'Ad', side = 'Top', dist = 150).tilt(0.15)
        plyScn = vs.players(serveShots)
        courtScn = Scene.make_court()

        ### MAKE KDE WITH ALL ON BASELINE
        from ATMANpy.types.ShapesTennis3D import Court 
        lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
        deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!
        
        joint_kde_null_scene = vs.kde_hillmap(serveShots_sample, showReg = adlineReg, position_func = find_intersection_with_service_line, 
                                    resolution = RESOLUTION_KDE, normalize_density = True, max_height = -0.5, one_color = const.BGR['strong_blue'])
        joint_kde_scene = vs.kde_hillmap(serveShots_sample, showReg = adlineReg, position_func = find_intersection_with_service_line, 
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

        low_kde_scene = vs.kde_hillmap(lowShots, showReg = adlineReg, position_func = find_intersection_with_service_line, 
                                        resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['grass_green'])
        mid_kde_scene = vs.kde_hillmap(midShots, showReg = adlineReg, position_func = find_intersection_with_service_line, 
                                        resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['soft_yellow'])
        high_kde_scene = vs.kde_hillmap(hiShots, showReg = adlineReg, position_func = find_intersection_with_service_line, 
                                        resolution = RESOLUTION_KDE, normalize_density = True, one_color = const.BGR['intense_red'])

        mid_kde_scene.rastArrs[0] += [0, 60, 0]
        high_kde_scene.rastArrs[0] += [0, 120, 0]

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

        loPt = origPt_3D.left(15).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
        midPt = origPt_3D.back(70).left(20).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
        hiPt = origPt_3D.back(150).left(25).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

        tPt = origPt_3D.right(15).front(10).annotate('Down the Middle')    
        wPt = origPt_3D.right(110).front(10).annotate('Out Wide')
        
        ######## Tests for similarity of distribution #########
        diagPt = Point3D((x_ax.leftPt.x, y_ax.rightPt.y, 0))
        jntStr = make_stat_string(lowShots, hiShots)
        statPt = diagPt.annotate(jntStr).back(110).left(40)

        ##### Player name and handedness ####
        text_size = 1
        thickness = 1
        offset = 0.5
        color = const.BGR['bright_grey']

        myLeg = Legend.make_empty(resolution = 100, header_offset = 2.5, legend_dimensions = (400, 200), header_thickness = 2)
        handedness = serveShots_clean_bounce[0].hitPlayer.get_hand().capitalize()
        l1 = {'text_string_shape': 'Player:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
              'text_string': player_name,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}        
        l2 = {'text_string_shape': 'Handedness:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
              'text_string': handedness,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}
        myLeg = myLeg.add(l1).add(l2)

        legFrm = myLeg.render()

        ### animate
        myFrms, animDict = vs.animate_sequence(split_itpl_scenes, background_scene = x_y_scene, cumulate_argument = False, 
                                                start_cam = boxCam, stop_cam = boxCam,
                                                annotated_points = [loPt, midPt, hiPt, tPt, wPt],
                                                **prmDict)
        myFrms = overlay(myFrms, legFrm)
        all_frames.append(myFrms)

        myFrms, animDict = vs.animate_static(split_kde_scenes, background_scene = x_y_scene, static_cam = boxCam, num_frames = 50,
                                            annotated_points = [loPt, midPt, hiPt, tPt, wPt, statPt],
                                            **prmDict)
        myFrms = overlay(myFrms, legFrm)
        all_frames.append(myFrms)

        jntFrms = BGRFrames.from_frames(all_frames)
        DecapStream(jntFrms, 0, 100).show(ms_between_frames = 25)

        ### save last frame for future watching
        lsFrm = myFrms[-1]
        # lsFrm.show()

        lsFrm.save_as_png(sub_graphPath, player_name)
        
if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

    name = ppl._name

    main(confs, name = None)

    
    