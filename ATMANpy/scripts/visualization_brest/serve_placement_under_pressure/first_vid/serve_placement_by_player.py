# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:09:57 2024

@author: gianl
"""

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
from ATMANpy.types.Abstract import Points, read_in_all_tables

from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player, Racquet
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D, Lines3D

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from ATMANpy.auxil.court import courtLines

def load(vid, sto):
    dataTable = ff.read_feature_frame('dataTable', vid, sto)
    return dataTable

def save(vid, sto):
    pass

def condition_on_handedness(myPnt, hit_player_hand = None, rec_player_hand = None):
    if hit_player_hand is None:
        hit_hand_correct = True
    else:
        hit_hand_correct = myPnt.get_serve().hitPlayer.get_hand() == hit_player_hand
    
    if rec_player_hand is None:
        rec_hand_correct = True
    else:
        rec_hand_correct = myPnt.get_serve().recPlayer.get_hand() == rec_player_hand

    return hit_hand_correct and rec_hand_correct

### PARAMETERS ### 
DESCRIPTOR = 'First Serves Deuce'
SAVE_FOLDER = 'testing'
savePath = ut.get_folder_struct('results_video')[SAVE_FOLDER]

QUANTILES =  [1/3, 1/2, 2/3]
FREEZE_FRAMES = 200
CAMERA_MOVE_FRAMES = 40
SHOW_COLOR = const.BGR['strong_pink']
TEXT_COLOR = const.BGR['grass_green']
TEXT_SIZE = 1.5
TEXT_WHERE = (50, 200)
TEXT_OFFSET = 50
RESOLUTION_MODIFIER = 2 # 2
RESOLUTION_BACKGROUND = 0.25 * RESOLUTION_MODIFIER
RESOLUTION_OBJECTS = 1 * RESOLUTION_MODIFIER
RESOLUTION_KDE = 0.5 * RESOLUTION_MODIFIER
MIX_COLORS_KDE = True
BANDWIDTH_KDE = 30
SCENE_FADE = 0.9
CONTOUR_PROP = 0.5
EPSILON = 0.002
EMPHASIS_DIM_FACTOR = 1.5
ARGUMENT_LUMINOSITY = 1.25
ADAPTIVE_RESOLUTION_EXPONENT = 1.5

KDE_HEIGHT = 20

CUTOFF_FLIGHT_FRAMES_1 = 6
CUTOFF_FLIGHT_FRAMES_2 = 14
EXPANSION_FACTOR = 8
C1 = CUTOFF_FLIGHT_FRAMES_1 * EXPANSION_FACTOR
C2 = CUTOFF_FLIGHT_FRAMES_2 * EXPANSION_FACTOR

backScn = Scene.make_court()
header_text = f'{DESCRIPTOR}: '
permanent_animation_params = {'background_resolution': RESOLUTION_BACKGROUND,
                                'argument_resolution': RESOLUTION_OBJECTS,
                                'foreground_resolution': RESOLUTION_OBJECTS,
                                'dummy_resolution': RESOLUTION_KDE,
                                'adaptive_resolution_exponent': ADAPTIVE_RESOLUTION_EXPONENT,
                                'argument_luminosity': ARGUMENT_LUMINOSITY,
                                'background_scene': backScn,
                                'text_size': TEXT_SIZE,
                                'text_color': TEXT_COLOR,
                                'text_where': TEXT_WHERE,
                                'text_offset': TEXT_OFFSET}


def main(confs, name = None):
    ### READ IN DATA
    allTable = read_in_all_tables(confs)
    allPnts = Points.from_datatable(allTable)

    fstPnts = allPnts.filter('is_first_serve')
    secPnts = allPnts.filter('is_second_serve')

    r2r_fstPnts = fstPnts.filter(condition_on_handedness, hit_player_hand = 'right', rec_player_hand = 'right')
    l2r_fstPnts = fstPnts.filter(condition_on_handedness, hit_player_hand = 'left', rec_player_hand = 'right')
    r2l_fstPnts = fstPnts.filter(condition_on_handedness, hit_player_hand = 'right', rec_player_hand = 'left')

    r2r_serveShots = r2r_fstPnts.get_serves().enforce_direction('Bot')    
    r2l_serveShots = r2l_fstPnts.get_serves().enforce_direction('Bot')

    r2r_deuceShots = r2r_serveShots.filter(f'is_serve_deuce')
    r2l_deuceShots = r2l_serveShots.filter(f'is_serve_deuce')
    r2r_adShots = r2r_serveShots.filter(f'is_serve_ad')
    r2l_adShots = r2l_serveShots.filter(f'is_serve_ad')

    deuceReg = courtLines.get_service_box(side = 'Top', which = 'Deuce')
    adReg = courtLines.get_service_box(side = 'Top', which = 'Ad')

    r2r_color = (35, 119, 203) #const.BGR['light_brown']
    r2l_color = const.BGR['bright_blue']

    #### VARIOUS KDE DISTRIBUTION SCENES ###
    r2r_nullScn = vs.kde_hillmap(r2r_deuceShots, showReg = deuceReg.scale(1.25), position_func = 'get_first_bounce_position',
                             one_color = r2r_color, max_height = -0.5, 
                             bandwidth = BANDWIDTH_KDE, resolution = RESOLUTION_KDE)
    r2r_hillScn = vs.kde_hillmap(r2r_deuceShots, showReg = deuceReg.scale(1.25), position_func = 'get_first_bounce_position',
                             one_color = r2r_color, max_height = KDE_HEIGHT, 
                             bandwidth = BANDWIDTH_KDE, resolution = RESOLUTION_KDE)
    r2r_clusterScn, r2r_cluster_data = vs.kde_contourmap(r2r_deuceShots, showReg = deuceReg.scale(1.25), position_func = 'get_first_bounce_position',
                                                 bandwidth = BANDWIDTH_KDE, resolution = RESOLUTION_KDE,
                                                 threshold_proportion = CONTOUR_PROP, epsilon = EPSILON, n_components = 2,
                                                 preserve_showarr_shape = True, max_height = KDE_HEIGHT,
                                                 contour_color = r2r_color, contour_alpha = 1, fill = True)
    r2r_itplScns_fadein = vs.interpolate_dummy_scenes(r2r_nullScn, r2r_hillScn, num_frames = 25)
    r2r_itplScns = vs.interpolate_dummy_scenes(r2r_hillScn, r2r_clusterScn, num_frames = 25, x_fill = 'other', y_fill = 'other', z_fill = -5)

    r2l_hillScn = vs.kde_hillmap(r2l_deuceShots, showReg = deuceReg.scale(1.25), position_func = 'get_first_bounce_position',
                             one_color = r2l_color, max_height = KDE_HEIGHT, 
                             bandwidth = BANDWIDTH_KDE, resolution = RESOLUTION_KDE)
    r2l_clusterScn, r2l_cluster_data = vs.kde_contourmap(r2l_deuceShots, showReg = deuceReg.scale(1.25), position_func = 'get_first_bounce_position',
                                                 bandwidth = BANDWIDTH_KDE, resolution = RESOLUTION_KDE,
                                                 threshold_proportion = CONTOUR_PROP, epsilon = EPSILON, n_components = 2,
                                                 preserve_showarr_shape = True, max_height = KDE_HEIGHT,
                                                 contour_color = r2l_color, contour_alpha = 1, fill = True)
    r2l_itplScns = vs.interpolate_dummy_scenes(r2l_hillScn, r2l_clusterScn, num_frames = 25, x_fill = 'other', y_fill = 'other', z_fill = -5)
    r2l_itplScns_reversed = vs.interpolate_dummy_scenes(r2l_clusterScn, r2l_hillScn, num_frames = 25, x_fill = 'other', y_fill = 'other', z_fill = -5)

    itpl_clusterScns = vs.interpolate_dummy_scenes(r2r_clusterScn, r2l_clusterScn, num_frames = 25, x_fill = 'other', y_fill = 'other', z_fill = -5)

    ##### ANNOTATED POINTS ####
    text_func = lambda share: f'{share * 100:.0f}% of serves \n land within \n this region'
    r2r_clusterPts = vs.cluster_data_to_points(r2r_cluster_data, text_func = text_func, color = const.BGR['bright_grey'])
    r2r_clusterPts[0], r2r_clusterPts[1] = r2r_clusterPts[0].front(2).right(10), r2r_clusterPts[1].front(2).right(10)

    r2l_clusterPts = vs.cluster_data_to_points(r2l_cluster_data, text_func = text_func, color = const.BGR['bright_grey'])
    r2l_clusterPts[0], r2l_clusterPts[1] = r2l_clusterPts[0].front(2).right(10), r2l_clusterPts[1].front(2).right(10)

    from ATMANpy.auxil.court import top_srv_2D, top_deu_2D
    r2r_descPt_fs = Point3D.from_point2D(top_srv_2D, z = 1).right(10).front(30).annotate('... will generally balance serves \n towards the middle and the outside.').recolor(r2r_color)
    r2r_descPt_sc = Point3D.from_point2D(top_srv_2D, z = 1).right(10).front(30).annotate('However, he will slightly favor serves towards the middle, \n to the right-handed receivers backhand!').recolor(r2r_color)
    r2l_descPt = Point3D.from_point2D(top_srv_2D, z = 1).right(10).front(30).annotate('The server will now favor serves out wide, \n to the left-handed receivers backhand!').recolor(r2l_color)
    # r2l_descPt = Point3D.from_point2D(top_srv_2D, z = 1).front(30).annotate('... will reposition slightly towards the outside \n indicating relatively more serves out wide.').recolor(r2l_color)
    trans_descPt = Point3D.from_point2D(top_deu_2D, z = 1).right(35).annotate('Now, consider how serve behaviour changes \n when facing a left-handed receiver...').recolor(r2l_color)

    # #### PLAYERS ####
    # r2r_plyScn = vs.players_and_racquets(r2r_deuceShots, hit_player_color = r2r_color, rec_player_color = r2r_color,
    #                         hit_player_dir_vert = 0.15, rec_player_dir_vert = -0.5)
    # r2l_plyScn = vs.players_and_racquets(r2l_deuceShots, hit_player_color = r2l_color, rec_player_color = r2l_color,
    #                         hit_player_dir_vert = 0.15, rec_player_dir_vert = +0.5)
    r2r_plyScn = vs.players(r2r_deuceShots, hit_player_color = r2r_color, rec_player_color = r2r_color)
    r2l_plyScn = vs.players(r2l_deuceShots, hit_player_color = r2l_color, rec_player_color = r2l_color)

    r2r_hit_descPt = r2r_plyScn.shapes[0].get_shoulder_point(which = 'left').up(20).annotate('A right-handed player serving...').recolor(r2r_color)
    r2r_rec_descPt = r2r_plyScn.shapes[1].get_shoulder_point(which = 'left').up(20).annotate('... to a right-handed player \n on the deuce side...').recolor(r2r_color)

    r2l_hit_descPt = r2l_plyScn.shapes[0].get_shoulder_point(which = 'left').up(30).annotate('Though the servers position is virtually unchanged ...').recolor(r2l_color)
    r2l_rec_descPt = r2l_plyScn.shapes[1].get_shoulder_point(which = 'left').up(30).right(10).annotate('... the left-handed receiver stands slightly to the right, \n anticipating more serves out wide, \n and giving himself more space on the forehand.').recolor(r2l_color)

    ### ANIMATION ###
    from ATMANpy.auxil.court import birds_eye_service_box_cam, back_view_asymmetric_cam
    top_deuCam = birds_eye_service_box_cam(zoom = 0.825, which = 'Deuce', side = 'Top').down(2)
    top_back_deuCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Top')

    my_frames = []
    
    myFrms, previous_foreground_scenes = vs.animate_static(r2r_plyScn.shapes[0], annotated_points = [r2r_hit_descPt],
                                           static_cam = top_back_deuCam, 
                                           num_frames = FREEZE_FRAMES // 2,
                                            **permanent_animation_params)
    my_frames.append(myFrms)
    
    myFrms, previous_foreground_scenes = vs.animate_static(r2r_plyScn.shapes[1], annotated_points = [r2r_rec_descPt],
                                           static_cam = top_back_deuCam,
                                           num_frames = FREEZE_FRAMES // 2,
                                           **previous_foreground_scenes,
                                           **permanent_animation_params)
    my_frames.append(myFrms)
    

    myFrms, previous_foreground_scenes = vs.animate_sequence(r2r_itplScns_fadein, 
                                                            start_cam = top_back_deuCam, stop_cam = top_back_deuCam,
                                                            **previous_foreground_scenes,
                                                            **permanent_animation_params)
    my_frames.append(myFrms)
    
    ### TODO: this permanent wiping of scenes has to be changed 
    ### -> need a dummy scene constituent (equivalent of Shape -> call it Raster)
    ### and it needs a name (like all shapes, also, will have to have). Then, we can filter out Shapes from Scenes (and Rasters from DummyScene) by name.
    previous_foreground_scenes['previous_dummy_scene'] = None 
    myFrms, previous_foreground_scenes = vs.animate_static(r2r_hillScn, annotated_points = [r2r_descPt_fs],
                                                           static_cam = top_back_deuCam, 
                                                           num_frames = FREEZE_FRAMES,
                                                            **previous_foreground_scenes,
                                                            **permanent_animation_params)
    my_frames.append(myFrms)

    previous_foreground_scenes['previous_dummy_scene'] = None
    myFrms, previous_foreground_scenes = vs.animate_sequence(r2r_itplScns, annotated_points = [r2r_descPt_fs],
                                                    start_cam = top_back_deuCam, stop_cam = top_deuCam,
                                                    cumulative_foreground_transparence = SCENE_FADE,
                                                    # cumulative_foreground_transparence = None,
                                                    **previous_foreground_scenes,
                                                    **permanent_animation_params)
    my_frames.append(myFrms)

    previous_foreground_scenes['previous_dummy_scene'] = None
    # previous_foreground_scenes = {'previous_annotated_points': [r2r_descPt_sc]} ### wipe almost all the history...
    myFrms, previous_foreground_scenes = vs.animate_static(r2r_clusterScn, annotated_points = [r2r_descPt_sc, *r2r_clusterPts], 
                                                           static_cam = top_deuCam, 
                                                           num_frames = FREEZE_FRAMES * 1.5,
                                                           **previous_foreground_scenes,
                                                           **permanent_animation_params)
    my_frames.append(myFrms)

    
    ##### ADD IN LEFT HANDED RECEIVER SITUATION #####
    # -> here, we begin with the left handed cluster -> chiastic!
    previous_foreground_scenes['previous_dummy_scene'] = None
    myFrms, previous_foreground_scenes = vs.animate_static(r2r_clusterScn, annotated_points = [trans_descPt],
                                                           static_cam = top_deuCam, 
                                                           num_frames = FREEZE_FRAMES,
                                                            **previous_foreground_scenes,
                                                            **permanent_animation_params)
    my_frames.append(myFrms)

    myFrms, previous_foreground_scenes = vs.animate_static(r2l_clusterScn, annotated_points = [trans_descPt, *r2l_clusterPts],
                                                           static_cam = top_deuCam, 
                                                           num_frames = FREEZE_FRAMES,
                                                           **permanent_animation_params)
    my_frames.append(myFrms)

    previous_foreground_scenes['previous_dummy_scene'] = None
    myFrms, previous_foreground_scenes = vs.animate_static(r2l_clusterScn, annotated_points = [r2l_descPt, *r2l_clusterPts],
                                                           static_cam = top_deuCam, 
                                                           num_frames = FREEZE_FRAMES,
                                                           **previous_foreground_scenes,
                                                           **permanent_animation_params)
    my_frames.append(myFrms)

    jnt_plyScn = r2r_plyScn.transparify(0.25).combine(r2l_plyScn)
    myFrms, previous_foreground_scenes = vs.animate_sequence(r2l_itplScns_reversed, annotated_points = [r2l_descPt],
                                                             previous_foreground_scene = jnt_plyScn,
                                                             start_cam = top_deuCam, stop_cam = top_back_deuCam,
                                                             **permanent_animation_params)
    my_frames.append(myFrms)

    myFrms, previous_foreground_scenes = vs.animate_static(None, annotated_points = [r2l_descPt],
                                                            static_cam = top_back_deuCam,
                                                            num_frames = FREEZE_FRAMES,
                                                            **previous_foreground_scenes,
                                                            **permanent_animation_params)
    my_frames.append(myFrms)

    ### split player scene into top and bottom, so we can put emphasis with the shapes on the appearing text
    hitScn = Scene([jnt_plyScn.shapes[0], jnt_plyScn.shapes[2]])
    myFrms, previous_foreground_scenes = vs.animate_static(hitScn, annotated_points = [r2l_hit_descPt],
                                                            static_cam = top_back_deuCam,
                                                            num_frames = FREEZE_FRAMES,
                                                            **previous_foreground_scenes,
                                                            **permanent_animation_params)
    my_frames.append(myFrms)

    recScn = Scene([jnt_plyScn.shapes[1], jnt_plyScn.shapes[3]])
    myFrms, previous_foreground_scenes = vs.animate_static(recScn, annotated_points = [r2l_rec_descPt], 
                                                           static_cam = top_back_deuCam,
                                                           num_frames = FREEZE_FRAMES,
                                                           **previous_foreground_scenes,
                                                           **permanent_animation_params)
    my_frames.append(myFrms)



    jntFrms = BGRFrames.from_frames(my_frames)
    DecapStream(jntFrms, 0, jntFrms.length()).show(ms_between_frames = 25)

    saveName = DESCRIPTOR + '_' + str(ut.make_timestamp())
    jntFrms.save(savePath, saveName, save_as = 'mp4', lossless = True)

if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

    name = ppl._name

    main(confs, name = None)

    
    
    
    