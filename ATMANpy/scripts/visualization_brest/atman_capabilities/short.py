import os
import numpy as np
import pandas as pd
        
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames, MaskFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Visual3D import Camera, View, Scene

from ATMANpy.types.Abstract import NullPoint, Points, Shots, read_in_data_tables, read_in_placeholder_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

from ATMANpy.models.ParticleFilter import ParticleFilter
from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'lines'
SAVE_FOLDER = 'atman_capabilities'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

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

def animate_bounce_and_mark_scenes(myPnt, start_cam, stop_cam, 
                                    background_scene = None,
                                    over = False, over_scale = 3, init_over_dim = 1, last_over_dim = 1,
                                    sub_region = True, region_color = const.BGR['smokey_grey'],
                                    **anim_kwargs):
    from ATMANpy.types.Abstract import Point

    spline_scenes, bounce_scenes = vis_ut.get_shot_scenes(myPnt)

    if over:
        myFrms = myPnt.stream(confs = confs).decap().frames
        if isinstance(myPnt, Point):
            myFrms = myFrms[:-1]
        myFrms = myFrms.rescale(over_scale)
        myFrms = myFrms.dim_progressively(fs_dim_factor = init_over_dim, ls_dim_factor = last_over_dim)
    else:
        myFrms = None

    if background_scene is None:
        courtScn = Scene.make_empty()
    else:
        courtScn = background_scene

    if sub_region:            
        from ATMANpy.types.Lines import CourtLines
        from ATMANpy.types.Shapes3D import Region3D
        subReg = CourtLines.make().get_court_region().scale(1.25)
        subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(region_color)
        courtScn = courtScn.put(subReg3D)

    markFrms, _ = vs.animate_sequence(bounce_scenes, start_cam = start_cam, stop_cam = stop_cam,
                                        background_scene = courtScn, 
                                        animate_over = myFrms, 
                                        **anim_kwargs)

    animFrms, _ = vs.animate_sequence(spline_scenes, start_cam = start_cam, stop_cam = stop_cam, 
                                        animate_over = markFrms,
                                        cumulative_foreground_transparence = 0.5, cumulate_argument = True,
                                        **anim_kwargs)

    return animFrms

def make_point_descriptor_box(myPnt, dims = (200, 100), origin = (10, 20), step_px = 20,
                              text_size = 1,
                              back_color = const.BGR['smokey_grey'], text_color = const.BGR['leafy_green']):
    
    cols, rows = dims
    o_x, o_y = origin

    textFrm = BGRFrame.make_black(cols, rows)
    textFrm = textFrm + back_color

    kwargs = {'size': text_size, 'color': text_color}

    shot_frames = []
    for myShot in myPnt:
        n = myShot.get_num_frames() - 1

        shotStr = myShot.get_shot_string()
        dataStr = myShot.get_data_string(with_descriptors = False, get_launch_angle = True)

        fullFrm = textFrm.put(shotStr, where = (o_x, o_y), **kwargs).put(dataStr, where = (o_x, o_y + step_px), **kwargs)

        fullFrms = BGRFrames.from_frame(fullFrm, n)
        shot_frames.append(fullFrms)

    fullFrms = BGRFrames.from_frames(shot_frames)
    return fullFrms


###################################
######### READ IN DATA ############
###################################
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

########## Filtering for ad wide serves before the current one

import ATMANpy.funcs.filter_shots_funcs as filt_s
import ATMANpy.funcs.filter_points_funcs as filt_p

filtPnts1 = dataPnts.filter(filt_p.is_returner, name = 'Brandon Nakashima').filter(filt_p.is_longer_than, threshold = 1)

filtPnts2 = filtPnts1.filter('is_first_serve')
winArr = np.array(vis_ut.get_winners_from_score(filtPnts2, allPnts))
decArr = winArr != 0

filtPnts3 = filtPnts2.filter(decArr)
filtPnts4 = filtPnts3.filter('is_serve_deuce')
filtPnts5 = filtPnts4.filter(filt_p.is_serve_out_wide)

filtPnts_before = filtPnts5.filter(filt_p.is_not_server, name = 'Joao Sousa')
filtPnts_after = filtPnts5.filter(filt_p.is_server, name = 'Joao Sousa')

##############################
######### CAMERAS ############
##############################

brestCam_third = Camera.calibrated_brest_behind()
brestCam_half = brestCam_third.resize(1.5)
brestCam_full = brestCam_third.resize(3)

######################################
### CHECK DISTRO OF BEFORE RETURNS ###
######################################

retShots = filtPnts_before.get_returns()
retShots_rect = retShots.enforce_direction('Top')

retShots_rect_out = retShots_rect.filter('is_fair', nextShot = None, negate = True)
retShots_rect_in = retShots_rect.filter('is_fair', nextShot = None)

heatScn_in = vs.kde_heatmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'],
                            bandwidth = 40, resolution = 2, normalize_density = True)

crtView = View.make(brestCam_full, Scene.make_court())
heatView = View.make(brestCam_full, heatScn_in)
# crtView.combine(heatView).snap().show()

markScn_in = vs.scatter_markmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'])
markScn_out = vs.scatter_markmap(retShots_rect_out, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'])

crtView = View.make(brestCam_full, Scene.make_court())
markView = View.make(brestCam_full, markScn_in)
# crtView.combine(markView).snap().show()

crtView = View.make(brestCam_full, Scene.make_court())
heatView = View.make(brestCam_full, markScn_out)
# crtView.combine(heatView).snap().show()

#############################
### CHECK CANDIDATES ########
#############################
CHECK_CANDIDATES = False
if CHECK_CANDIDATES:
    # idxs_checked = [1, 2, 3, 6, 7] ### for ad serves
    # best_idx = 3 ### for ad serves

    idxs_checked = None

    idxs_checked = [5, 6, 7, 8, 10, 14, 16, 17]
    # idxs_checked = [14]

    ### check out candidate points: 
    for idx, myPnt in enumerate(filtPnts_after):
        if idxs_checked is not None:
            if idx not in idxs_checked:
                continue
    
        print(idx)
        print(myPnt.get_match_index(), myPnt.get_point_index())
        animFrms = animate_bounce_and_mark_scenes(myPnt, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                sub_region = False, argument_resolution = 10)
        
        
        _ = input('any key to continue ...')
        DecapStream(animFrms, 0, 100).show()

    ### check out candidate points: 
    for idx, myPnt in enumerate(filtPnts_after):
        if idxs_checked is not None:
            if idx not in idxs_checked:
                continue
    
        print(idx)

        _ = input('any key to continue ...')
        myPnt.stream(confs = confs).widen(500).show()


    ## check out points before candidate points
    # -> have to be right before, 
    # since we introduce the idea of filtering for knowledge about this ad-wide serve and then go into the filtering overlay.
    # we then cascade the filters back out, to show we're back at the serve in question.

    idxDict = {idx: (pt.get_match_index(), pt.get_point_index()) for idx, pt in enumerate(filtPnts_after)}

    for idx, ptTup in idxDict.items():
        if idx in idxs_checked:
            print(f'Shot index {idx}')

            mtc_idx, pnt_idx = ptTup
            prevPnt1 = allPnts.get(mtc_idx, pnt_idx - 1)
            prevPnt2 = allPnts.get(mtc_idx, pnt_idx - 2)
            prevPnt3 = allPnts.get(mtc_idx, pnt_idx - 3)

            print('One point before...')        
            if not isinstance(prevPnt1, NullPoint):
                animFrms = animate_bounce_and_mark_scenes(prevPnt1, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                        sub_region = False, argument_resolution = 10)
                _ = input('any key to continue ...')
                DecapStream(animFrms, 0, 100).show()
            else:
                print('... is null')
            
            print('Two points before...')
            if not isinstance(prevPnt2, NullPoint):
                animFrms = animate_bounce_and_mark_scenes(prevPnt2, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                        sub_region = False, argument_resolution = 10)
                _ = input('any key to continue ...')
                DecapStream(animFrms, 0, 100).show()
            else:
                print('... is null')

            print('Three points before...')
            if not isinstance(prevPnt3, NullPoint):
                animFrms = animate_bounce_and_mark_scenes(prevPnt3, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                        sub_region = False, argument_resolution = 10)
                _ = input('any key to continue ...')
                DecapStream(animFrms, 0, 100).show()
            else:
                print('... is null')
            
            print('_________')
            print('_________')
            print('_________')
            print('_________')
else:
    # idxs_checked = [1, 2, 3, 6, 7]
    # best_idx = 3
    ### -> for ad side

    idxs_checked = [5, 6, 7, 8, 10, 14, 16, 17]
    best_idx = 14
    ### -> for deuce side

### LOOK AT CHOSEN CANDIDATES ####
lastPnt = filtPnts_after[best_idx]
mtc_idx, pnt_idx = lastPnt.get_match_index(), lastPnt.get_point_index()

prevPnt1 = allPnts.get(mtc_idx, pnt_idx - 1)
prevPnt2 = allPnts.get(mtc_idx, pnt_idx - 2)
prevPnt3 = allPnts.get(mtc_idx, pnt_idx - 3)
# prevPnt4 = allPnts.get(mtc_idx, pnt_idx - 4)

# animFrms4 = animate_bounce_and_mark_scenes(prevPnt4, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
#                                                     sub_region = False, argument_resolution = 10)
# DecapStream(animFrms4, 0, 100).show()

animFrms3 = animate_bounce_and_mark_scenes(prevPnt3, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                    sub_region = False, argument_resolution = 10)
# DecapStream(animFrms3, 0, 100).show()

animFrms2 = animate_bounce_and_mark_scenes(prevPnt2, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                    sub_region = False, argument_resolution = 10)
# DecapStream(animFrms2, 0, 100).show()

animFrms1 = animate_bounce_and_mark_scenes(prevPnt1, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                    sub_region = False, argument_resolution = 10)
# DecapStream(animFrms1, 0, 100).show()

animFrms0 = animate_bounce_and_mark_scenes(lastPnt, start_cam = brestCam_full, stop_cam = brestCam_full, over = True,
                                                    sub_region = False, argument_resolution = 10)
# DecapStream(animFrms0, 0, 100).show()

widen_by = 150
preFrms = prevPnt3.stream(confs = confs).widen(widen_by).decap().frames[:widen_by].rescale(3)

##################################
### SCRIPT_LINES #################
##################################

script_line =   """
                    The Automated Tennis Match Annotator, or ATMAN for short,
                    is a start-to-end framework for the generation of insights
                    from tennis matches. 
                    ATMAN visually recognizes tennis matches, 
                    transforms them into data shot-by-shot, 
                    and allows one to reveal patterns in the style of play of a player.  
                """
### TITLE CONTRACTION 

script_line =   """
                    ATMAN recognizes the lines, players and the ball
                    and employs a suite of articifial intelligence models
                    to analyze player swing motion and the sound of the rally
                    to generate ball trajectories even on low quality video. 
                """ ### 12 sec
### TRAJECTORIES AND LANDING SPOTS

script_line  =  """
                    Every shot is described, and its speed, spin and launch angle
                    off the racket is calculated.
                """ ## 4 secs
### SHOW DATA
# -> dim play, show box

script_line =   """
                    ATMAN also includes a visual engine, which allows
                    any point to be displayed in arbitrary virtual space.
                """ ## 6 secs
### TRANSFORM VISUAL INTO TOP DOWN
# -> keep play dim, bright View transforming into top down

script_line =   """
                    Points detected by ATMAN can be easily searched, filtered and aggregated.
                    The visual engine can then be seamlessly employed to display results.
                """
# -> transform top down into cooler 3D view, move into top right raster box.
# might need a little fade between the two views, 
# since we can approximately move them over each other, but not properly.
# alternative to fading in and out: just leave the full camera view, and have a hole in the raster.

script_line =   """
                    For example, in the final of the 2021 Brest Challenger Tournament, 
                    Joao Sousa is about to hit a first serve from the ad side out wide to Brandon Nakashima.
                    With ATMAN, one can filter for occurences in previous matches of the same tournament
                    in which Brandon Nakashima faced a similar serve, and take a look at where his return
                    previously landed.
                """
# raster with previous examples, trajectories and bounce points shown in the cool view top right.
# maybe bounce points are made into heatmap directly (or transform in a cool animation, like shockwave)

script_line =   """
                    [Comment on where the ball lands vs. "expectation"]
                """
# fade the 3D view in raster back to 3D view with full camera
# transform the cool 3D view in full camera back to the court, seamlessly superimposing the heatmap
# check where the ball landed

script_line =   """
                    [Closing statement]
                """
# resume rally, fade out with the rest of the rally playing.

all_frames = []
script_lines = []

#####################################
#### ATMAN contraction animation ####
#####################################

script_line =   """
                    The Automated Tennis Match Annotator, or ATMAN for short,
                    is a start-to-end framework for the generation of insights
                    from tennis matches. 
                    ATMAN visually recognizes tennis matches, 
                    transforms them into data shot-by-shot, 
                    and allows one to reveal patterns in the style of play of a player.  
                """

from ATMANpy.types.Shapes3D import Point3D
z = 1000
oPt, cPt = Point3D((0, 0, 0)), Point3D((0, 0, z))
downCam = Camera.from_to(cPt, oPt)
text_color = const.BGR['leafy_green']
text_size = 2
left_pts = 325

titleStr = 'The Automated Tennis Match ANnotator'
titlePt = oPt.annotate(titleStr).left(left_pts).recolor(text_color).resize(text_size)
titleView = View.make(downCam, Scene.make_empty(), annotatedPoints = [titlePt])
titleFrm = titleView.snap()
# titleFrm.show()

auxil_offset = 0
move1 = left_pts - 76
move2 = left_pts - 270
move3 = left_pts - 392
move4 = left_pts - 506
move5 = left_pts - 526

p1 = oPt.annotate('A').front(auxil_offset).left(move1).recolor(text_color).resize(text_size)
p2 = oPt.annotate('T').front(auxil_offset).left(move2).recolor(text_color).resize(text_size)
p3 = oPt.annotate('M').front(auxil_offset).left(move3).recolor(text_color).resize(text_size)
p4 = oPt.annotate('A').front(auxil_offset).left(move4).recolor(text_color).resize(text_size)
p5 = oPt.annotate('N').front(auxil_offset).left(move5).recolor(text_color).resize(text_size)

target1 = - 40
target2 = - 20
target3 = 0
target4 = 25
target5 = 47

p1_target = p1.right(target1 + move1)
p2_target = p2.right(target2 + move2)
p3_target = p3.right(target3 + move3)
p4_target = p4.right(target4 + move4)
p5_target = p5.right(target5 + move5)

titleView = View.make(downCam, Scene.make_empty(), annotatedPoints = [titlePt])
titleFrm = titleView.snap()
# titleFrm.show()

letterView = View.make(downCam, Scene.make_empty(), annotatedPoints = [p1, p2, p3, p4, p5])
letterFrm = letterView.snap()
# letterFrm.show()

itpl_func = lambda l: [vs.interpolate_shape(p1, p1_target, l), 
                       vs.interpolate_shape(p2, p2_target, l),
                       vs.interpolate_shape(p3, p3_target, l),
                       vs.interpolate_shape(p4, p4_target, l),
                       vs.interpolate_shape(p5, p5_target, l)]

N = 50
pre_dim_factor = 0.5

pre_dimFrms = BGRFrames.from_frame(titleFrm, N).dim_progressively(fs_dim_factor = 1, ls_dim_factor = 0.5)
post_dimFrms = BGRFrames.from_frame(titleFrm, N).dim_progressively(fs_dim_factor = 0.5, ls_dim_factor = 0)
lambdas = np.linspace(0, 1, N, endpoint = True)

### deal with pre move dim frames
pre_dimFrms_lettered, _ = vs.animate_sequence([Scene.make_empty() for _ in range(N)], start_cam = downCam, stop_cam = downCam,
                            annotated_points = [p1, p2, p3, p4, p5], animate_over = pre_dimFrms)

### deal with post move dim frames
pt_frames = []
for i in range(N):
    l = lambdas[i]
    frm = post_dimFrms[i]

    annotated_points = itpl_func(l)
   
    ptView = View.make(downCam, Scene.make_empty(), annotatedPoints = annotated_points)
    ptFrm = ptView.snap(over_array = frm.array)
    pt_frames.append(ptFrm)

post_dimFrms_lettered = BGRFrames.from_single_frames(pt_frames)

### deal with preFrms
fade_in_length = 50

post_dimFrms_fade_out = BGRFrames.from_frame(post_dimFrms_lettered[-1], fade_in_length).dim_progressively(1, 0)
preFrms_fade_in = preFrms[:fade_in_length].dim_progressively(0, 1)

preFrms_fade = post_dimFrms_fade_out + preFrms_fade_in
# DecapStream(preFrms_fade, 0, 100).show()

preFrms_rest = preFrms[fade_in_length:]

jointFrms = BGRFrames.from_frames([pre_dimFrms_lettered, post_dimFrms_lettered, preFrms_fade, preFrms_rest])
# DecapStream(jointFrms, 0, 100).show()

all_frames.append(jointFrms)
script_lines.append(script_line)

###############################
##### FIRST RALLY TO SHOW #####
###############################

num_frames = 300
fade_in_frames = 25

fade_factor = 0.5

cols, rows = 243, 100
back_color = (60, 60, 60)
text_color = const.BGR['leafy_green']
o_x, o_y = 10, 20
step_px = 30

script_line =   """
                    ATMAN recognizes the lines, players and the ball
                    and employs a suite of articifial intelligence models
                    to analyze player swing motion and the sound of the rally
                    to generate ball trajectories even on low quality video. 
                """ ### 12 sec

animFrms_pre = animFrms3[:num_frames]
animFrms_fade = animFrms3[num_frames:num_frames+fade_in_frames].dim_progressively(fs_dim_factor = 1, ls_dim_factor = fade_factor)
animFrms_post = animFrms3[num_frames+fade_in_frames:].dim(fade_factor)

all_frames.append(animFrms_pre)
script_lines.append(script_line)

# DecapStream(animFrms_bef, 0, 100).show()

script_line  =  """
                    Every shot is described, and its speed, spin and launch angle
                    off the racket is calculated.
                """ ## 4 secs

fullFrms = make_point_descriptor_box(prevPnt3, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = (255, 255, 255), text_size = 1)
# DecapStream(fullFrms, 0, 100).show()

fullFrms_fade = fullFrms[num_frames:num_frames+fade_in_frames]
fullFrms_post = fullFrms[num_frames+fade_in_frames:]

over_x, over_y = 1635, 650
tlPt = Point2D((over_x, over_y))
fadeFrms = overlay(animFrms_fade, fullFrms_fade, top_left_point = tlPt, start_alpha = 0, stop_alpha = 1)
postFrms = overlay(animFrms_post, fullFrms_post, top_left_point = tlPt)

jointFrms = BGRFrames.from_frames([fadeFrms, postFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

vis_ut.show(all_frames)

##############################
##### SEC. RALLY TO SHOW #####
##############################

script_line =   """
                    ATMAN also includes a visual engine, which allows
                    any point to be displayed in arbitrary virtual space.
                """ ## 6 secs

back_color = (60, 60, 60)

c_x, c_y, c_z = -900, 900, 1500

brestCam = Camera.calibrated_brest_behind().resize(3)
downCam = Camera.from_to(Point3D((c_x, c_y, c_z)), Point3D((c_x, c_y, 0)))

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D
subReg = CourtLines.make().get_court_region().scale(1.25)
subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(back_color)

# testScn = Scene.make_court().put(subReg3D)
# View.make(stopCam, testScn, resolution = 1).snap(over_array = prevPnt2.stream(confs = confs).decap().frames[0].rescale(3).array).show()

prmDict = {'background_resolution': 0.1, 'argument_resolution': 0.1, 'interpolate_func': vs.interpolate_camera_naive}

animFrms = animate_bounce_and_mark_scenes(prevPnt2, start_cam = brestCam, stop_cam = downCam, background_scene = Scene.make_court(), 
                                          over = True, sub_region = True, region_color = back_color,
                                          init_over_dim = 0.5, last_over_dim = 0.5,
                                          start_later = 0, stop_earlier = 300,
                                          **prmDict)

fullFrms = make_point_descriptor_box(prevPnt2, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = (255, 255, 255), text_size = 1)

over_x, over_y = 1635, 650
tlPt = Point2D((over_x, over_y))
postFrms = overlay(animFrms, fullFrms, top_left_point = tlPt)
# DecapStream(postFrms, 0, 100).show()

###############################
###### THIRD RALLY TO SHOW ####
###############################

script_line =   """
                    Points detected by ATMAN can be easily searched, filtered and aggregated.
                    The visual engine can then be seamlessly employed to display results.
                """
# -> while showing third (short) rally...
# ... transform top down into cooler 3D view, move into top right raster box.
# might need a little fade between the two views, 
# since we can approximately move them over each other, but not properly.
# alternative to fading in and out: just leave the full camera view, and have a hole in the raster.

animFrms = animate_bounce_and_mark_scenes(prevPnt1, start_cam = downCam, stop_cam = downCam, background_scene = Scene.make_court(), 
                                          over = True, sub_region = True, region_color = back_color,
                                          init_over_dim = 0.5, last_over_dim = 0.5,
                                          start_later = 0, stop_earlier = 300,
                                          **prmDict)

fullFrms = make_point_descriptor_box(prevPnt1, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
                                     back_color = back_color, text_color = (255, 255, 255), text_size = 1)

over_x, over_y = 1635, 650
tlPt = Point2D((over_x, over_y))
postFrms = overlay(animFrms, fullFrms, top_left_point = tlPt)
# DecapStream(postFrms, 0, 100).show()



cntPt = CourtLines.make().get_center_point()
cntPt3D = Point3D.from_point2D(cntPt, z = 0).left(750).down(600)
camPt3D = Point3D((-300, 2000, 200))

thirdCam = Camera.from_to(camPt3D, cntPt3D)

testFrm = View.make(thirdCam, Scene.make_court(), resolution = 5).snap()
testFrm.show()

rs, cs = testFrm.rows, testFrm.cols
testFrm[:rs//3, -cs//3:].show()

cntPt = CourtLines.make().get_center_point()
cntPt3D = Point3D.from_point2D(cntPt, z = 0).left(670).back(770)
camPt3D = Point3D((-300, 2000, 600))

thirdCam = Camera.from_to(camPt3D, cntPt3D).zoom(1.1)

testFrm = View.make(thirdCam, Scene.make_court(), resolution = 5).snap()
testFrm.show()

rs, cs = testFrm.rows, testFrm.cols
testFrm[:rs//3, -cs//3:].show()


moveFrms, _ = vs.animate_cam_motion(Scene.make_court(), start_cam = downCam, stop_cam = thirdCam, num_frames = 100)

DecapStream(moveFrms, 0, 100).show()

### test rotating court

courtScn = Scene.make_court()
myCrt = courtScn[0]

courtLines = CourtLines.make().add_singles_netline()

from ATMANpy.types.ShapesTennis3D import Court, Net


n = 50
pis = np.linspace(0, np.pi, n)

rot_scenes = []
rot_frms = []
for pi in pis:
    rotLns = courtLines.rotate(pi)
    rotScn = Scene.make_court(rotLns)
    rot_scenes.append(rotScn)

    rotFrm = View.make(thirdCam, rotScn, resolution = 1).snap()
    rot_frms.append(rotFrm)

rotFrms = BGRFrames.from_single_frames(rot_frms)
DecapStream(rotFrms, 0, 100).show()

top_to_bot_scene = rot_scenes[0]
bot_to_top_scene = rot_scenes[-1]

rotFrms[-1][:rs//3, -cs//3:].show()

def return_is_fair(somePnt):
    try:
        thirdShot = somePnt[2]
    except:
        thirdShot = None
    return somePnt.get_return().is_fair(thirdShot)

filtPnts_before_fair = filtPnts_before.filter(return_is_fair)
retShots_before_fair = filtPnts_before_fair.get_returns() ### unrectified!

from ATMANpy.funcs.util_funcs import interpolate_tuples
from ATMANpy.types.Visual3D import DummyScene

mark_color = const.BGR['light_brown']
heat_color = const.BGR['leafy_green']
bandwidth = 40
intermediate_lambdas = [0.25, 0.5, 0.75]

prmDict = {'background_resolution': 0.05, 'argument_resolution': 0.05, 'interpolate_func': vs.interpolate_camera_naive}

kdeDict = { 'bandwidth': 40, 'normalize_density': True, 
            'one_color': heat_color, 'zero_color': heat_color, 
            'resolution': 2}

courtScn = Scene.make_court(courtLines) ### todo: rotate this if needed!
region_color = (60, 60, 60)
from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D
subReg = CourtLines.make().get_court_region().scale(1.25)
subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(region_color)
courtScn = courtScn.put(subReg3D)

centerPt = courtLines.get_center_point()
centerPt3D = Point3D.from_point2D(centerPt, z = 0)

currentFrm = View.make(thirdCam, courtScn, resolution = prmDict['background_resolution']).snap() ### initialize a "current" frame

rotation_animation_length = 25
fade_length = 10
fade_thresh = 0.025

dim_factor = 0.5

idx = 0
pnt = filtPnts_before_fair[0]

raster_frames = []
raster_rot_frames = []
raster_fade_frames = []
corner_frames = []
rotate_frames = []
fade_frames =   []
for idx, myPnt in enumerate(filtPnts_before_fair[:4]):
    print(idx)
    
    ### get previous shots and current shot (as Shots, i.e: list of length 1)
    prevShots, allShots = retShots_before_fair[:idx], retShots_before_fair[:idx+1]
    
    ### figure out orientation
    if len(prevShots) > 0:
        previous_orientation = prevShots[-1].hitPlayer.get_side()
    else:
        previous_orientation = None
    current_orientation  = allShots[0].hitPlayer.get_side()

    ### rectify, so we gather everything in one direction (for corner graphic)
    prevShots_rect_curr = prevShots.enforce_direction(current_orientation.capitalize())
    prevShots_rect_next = allShots.enforce_direction(current_orientation.capitalize())

    heatScn_rect_next = vs.kde_hillmap(prevShots_rect_next, **kdeDict)

    ### previous heat scene
    if len(prevShots) > 0:
        heatScn_rect_curr = vs.kde_hillmap(prevShots_rect_curr, **kdeDict)
        # heatScn_rect_curr = vs.kde_heatmap(prevShots_rect_curr, one_color = heat_color, zero_color = heat_color,
        #                                     bandwidth = bandwidth, resolution = 2, normalize_density = True)
    else:
        zero_heightArr = heatScn_rect_next.rastArrs[0].copy()
        zero_heightArr[:, 2] = 0
        heatScn_rect_curr = DummyScene([zero_heightArr], heatScn_rect_next.colArrs, heatScn_rect_next.alphaArrs).transparify(0)
        heatScn_rect_prev = DummyScene([zero_heightArr], heatScn_rect_next.colArrs, heatScn_rect_next.alphaArrs).transparify(0)

    ### little intermezzos, with rotation
    if current_orientation == previous_orientation or previous_orientation is None:
        frameList = [currentFrm for _ in range(rotation_animation_length)]
    else:
        prevShots_rect_prev = prevShots.enforce_direction(previous_orientation.capitalize())
        
        heatScn_rect_prev = vs.kde_hillmap(prevShots_rect_prev, **kdeDict)
        # heatScn_rect_prev = vs.kde_heatmap(prevShots_rect_prev, one_color = heat_color, zero_color = heat_color,
        #                                     bandwidth = bandwidth, resolution = 2, normalize_density = True)

        pis = np.linspace(0, np.pi, rotation_animation_length)

        frameList = []
        for pi in pis:
            heatScn_rect_rot = heatScn_rect_prev.rotate_flat(pi, origPt = centerPt3D)
            rotLns = courtLines.rotate(pi)
            rotReg = subReg.rotate(pi, origPt = centerPt)
            rotReg3D = Region3D.from_region2D(rotReg, z = -1).recolor(region_color)
            rotScn = Scene.make_court(rotLns).put(rotReg3D)
            
            heatView = View.make(thirdCam, heatScn_rect_rot, resolution = prmDict['argument_resolution'])
            courtView = View.make(thirdCam, rotScn, resolution = prmDict['background_resolution'])

            rotFrm = heatView.combine(courtView).snap()
            frameList.append(rotFrm)
        
    rotateFrms = BGRFrames.from_single_frames(frameList)

    ### filter for serve and return
    serve_and_return = myPnt[:2]
    myStm = serve_and_return.stream(confs = confs).maybe_to_stream()

    ### real frames
    myFrms = myStm.decap().frames

    ### spline, bounce scenes
    spline_scenes, bounce_scenes = vis_ut.get_shot_scenes(serve_and_return, mark_color = mark_color)
    spline_and_bounce_scenes = [sScn.combine(bScn) for sScn, bScn in zip(spline_scenes, bounce_scenes)]

    bounce_scenes_return = [myScn.drop(0) if not myScn.is_empty() else myScn for myScn in bounce_scenes]
    bounce_scenes_only_return = [myScn for myScn in bounce_scenes_return if not myScn.is_empty()]
    num_bounce_scenes = len(bounce_scenes_only_return)
    num_flight_scenes = len(bounce_scenes_return) - num_bounce_scenes

    ### interpolate the extending bounce heatmap
    # intermediate_lambdas = np.linspace(0, 1, num_bounce_scenes, endpoint = True)
    # itpl_scenes = []
 
    # for lmbd in intermediate_lambdas:
    #     itpl_color = interpolate_tuples(mark_color, heat_color, lmbd)
    #     itpl_bw = bandwidth * lmbd

    #     kdeDict_itpl_bw = {**kdeDict}
    #     kdeDict_itpl_bw['bandwidth'] = itpl_bw
    #     # heatScn_itpl = vs.kde_heatmap(currShots, one_color = itpl_color, zero_color = itpl_color,
    #     #                                 bandwidth = itpl_bw, resolution = 2, normalize_density = True)
    #     heatScn_itpl = vs.kde_hillmap(currShots, **kdeDict_itpl_bw)

    #     heatScn_itpl = heatScn_itpl.combine(heatScn_rect_curr, keep_which = 'both')

    #     itpl_scenes.append(heatScn_itpl)

    itpl_scenes = vs.interpolate_dummy_scenes(heatScn_rect_curr, heatScn_rect_next, num_bounce_scenes)
    prev_scenes = [heatScn_rect_curr for _ in range(num_flight_scenes)]
    heat_scenes = [*prev_scenes, *itpl_scenes]

    heatFrms, _ = vs.animate_sequence(heat_scenes, background_scene = courtScn, start_cam = thirdCam, stop_cam = thirdCam, 
                                    **prmDict)
    # DecapStream(heatFrms, 0, 100).show()

    spline_and_mark_and_heatFrms, _ = vs.animate_sequence(spline_and_bounce_scenes, start_cam = thirdCam, stop_cam = thirdCam,
                                                animate_over = heatFrms, 
                                                **prmDict)
    # DecapStream(mark_and_heatFrms, 0, 100).show()

    ### fade out marks and splines
    cumulative_transparence = fade_thresh ** (1/fade_length)

    fadeFrms, _ = vs.animate_sequence([Scene.make_empty() for _ in range(fade_length)], start_cam = thirdCam, stop_cam = thirdCam,
                                        previous_foreground_scene = spline_and_bounce_scenes[-1], animate_over = heatFrms[-1], 
                                        cumulative_foreground_transparence = cumulative_transparence, **prmDict)

    ### raster frames
    rasterFrms = animate_bounce_and_mark_scenes(serve_and_return, start_cam = brestCam_third, stop_cam = brestCam_third,
                                            over = True, over_scale = 1, sub_region = False,
                                            **prmDict)
    
    ### pre and post frames
    # -> these correspond to the rotate and the fade_frames, respectively
    rasterFrms_rot = myStm.widen(rotation_animation_length).decap().frames[:rotation_animation_length]
    rasterFrms_fade = myStm.widen(fade_length).decap().frames[-fade_length:].dim_progressively(ls_dim_factor = dim_factor)

    ### populate lists
    corner_frames.append(spline_and_mark_and_heatFrms)
    rotate_frames.append(rotateFrms)
    fade_frames.append(fadeFrms)

    raster_frames.append(rasterFrms)
    raster_rot_frames.append(rasterFrms_rot)
    raster_fade_frames.append(rasterFrms_fade)

    ### set iteration variables
    previous_orientation = current_orientation
    currentFrm = fadeFrms[-1]


# for idx, frms in enumerate(corner_frames):
#     print(idx)
#     DecapStream(frms, 0, 100).show()

# for idx, frms in enumerate(raster_frames):
#     print(idx)    
#     DecapStream(frms, 0, 100).show()

# for idx, frms in enumerate(rotate_frames):
#     print(idx)    
#     DecapStream(frms, 0, 100).show()

# prevShots, currShots = retShots_before_fair[:idx], retShots_before_fair[:idx+1]
# heatScn_prev = vs.kde_hillmap(prevShots, **kdeDict)
# heatScn_curr = vs.kde_hillmap(currShots, **kdeDict)

# heatScns_itpl = vs.interpolate_dummy_scenes(heatScn_prev, heatScn_curr, 10)

# testFrms, _ = vs.animate_sequence(heatScns_itpl, start_cam = thirdCam, stop_cam = thirdCam, background_scene = courtScn,
#                                 **prmDict)

# DecapStream(testFrms, 0, 100).show(ms_between_frames = 100)


joint_raster_frames = [a.join(b).join(c) for a, b, c in zip(raster_rot_frames, raster_frames, raster_fade_frames)]
joint_corner_frames = [a.join(b).join(c) for a, b, c in zip(rotate_frames, corner_frames, fade_frames)]
joint_corner_frames = [a[:, :rs//3, -cs//3:] for a in joint_corner_frames]

myRst = FramesRaster(3, 3)
full_raster_frames = []
idx = 0
for rFrms, cFrms in zip(joint_raster_frames, joint_corner_frames):
    ### index manipulation
    if idx == 2:
        idx = 3 ### skip top right
    idx = idx % 9 ### reset at 9
    
    r, c = idx // 3, idx % 3
    myRst.add(rFrms, r, c)
    myRst.add(cFrms, 0, 2)

    ### generate frame
    rstFrms = myRst.render()
    full_raster_frames.append(rstFrms)
    
    ### freeze raster
    myRst = myRst.to_frame_raster().to_frames_raster()

    ### increment
    idx += 1

vis_ut.show(full_raster_frames)

DecapStream(rstFrms, 0, 100).show()

rstFrms[-1].show()


raster_frames

fade_frames
corner_frames
rotate_frames



#### TESTING ROTATION OF DUMMY SCENE ####

testView = View.make(thirdCam, heatScn_in, resolution = 1)
courtView = View.make(thirdCam, Scene.make_court(), resolution = 1) 
allView = testView.combine(courtView)
allView.snap().show()

self = heatScn_in
angle = np.pi

s_a, c_a = np.sin(angle), np.cos(angle)

R = np.array([  [c_a, -s_a, 0],
                [s_a,  c_a, 0],
                [0  ,    0, 1]])
origPt = Point3D((0, 0, 0))
origPt = Point3D.from_point2D(courtLines.get_center_point(), z = 0)

origArr = np.array(origPt.coords)

rastArrs_rot = []
for rastArr in heatScn_in.rastArrs:
    rastArr_norm = rastArr - origArr
    rastArr_norm_rot = np.einsum('ij, nj -> ni', R, rastArr_norm)
    rastArr_rot = rastArr_norm_rot + origArr
    rastArrs_rot.append(rastArr_rot)

from ATMANpy.types.Visual3D import DummyScene

rotScn = DummyScene(rastArrs_rot, self.colArrs, self.alphaArrs)

testView = View.make(thirdCam, rotScn, resolution = 1)
courtView = View.make(thirdCam, Scene.make_court(), resolution = 1) 
allView = testView.combine(courtView)
allView.snap().show()







heatScn_in.rastArrs

heatScn_in










myRst = FramesRaster(3, 3)



# rotCrt = Court.make()

# rotLns
# rotLns.add_singles_netline()



# self = normLns[0].leftPt


# courtLines[0].leftPt - courtLines.get_center_point()

# myCrt['Top Baseline']

# self = moveFrms[0]

# sub_moveFrms = moveFrms[:, :rs//3, cs//3:]

# DecapStream(sub_moveFrms, 0, 100).show()

# #####################
# ### OVERHEAD VIEW ###
# #####################

# ## make an overhead view of a point, similar to SwingVision.
# ## little box with verbal shot info below a larger box of overhead frame

# myPnt = dataPnts[10]

# myPnt = dataPnts[16]

# spline_scenes, bounce_scenes = get_shot_scenes(myPnt)

# len(bounce_scenes)
# len(spline_scenes)

# myFrms = myPnt.stream(confs = confs).decap().frames[:-1]

# myCam = Camera.calibrated_brest_behind()
# markFrms, _ = vs.animate_sequence(bounce_scenes, start_cam = myCam, stop_cam = myCam, animate_over = myFrms)
# DecapStream(markFrms, 0, 100).show()

# animFrms, _ = vs.animate_sequence(spline_scenes, start_cam = myCam, stop_cam = myCam, animate_over = markFrms,
#                                 cumulative_foreground_transparence = 0.5, cumulate_argument = True)
# DecapStream(animFrms, 0, 100).show()

# ### top down view:

# back_color = (60, 60, 60)

# c_x, c_y, c_z = -900, 900, 1500

# startCam = Camera.calibrated_brest_behind().resize(3)
# stopCam = Camera.from_to(Point3D((c_x, c_y, c_z)), Point3D((c_x, c_y, 0)))

# from ATMANpy.types.Lines import CourtLines
# from ATMANpy.types.Shapes3D import Region3D
# subReg = CourtLines.make().get_court_region().scale(1.25)
# subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(back_color)

# testScn = Scene.make_court().put(subReg3D)
# View.make(stopCam, testScn, resolution = 1).snap(over_array = myPnt.stream(confs = confs).decap().frames[0].rescale(3).array).show()

# prmDict = {'background_resolution': 0.05, 'argument_resolution': 0.1, 'interpolate_func': vs.interpolate_camera_naive}

# animFrms = animate_bounce_and_mark_scenes(myPnt, start_cam = startCam, stop_cam = stopCam, background = True, 
#                                           over = True, last_over_dim = 0.5, sub_region = True, region_color = back_color,
#                                           start_later = 0, stop_earlier = 100,
#                                           **prmDict)
# DecapStream(animFrms, 0, 100).show()

# animFrms[-1].show()

# #### Textbox with description of shots

# cols, rows = 243, 100
# back_color = (60, 60, 60)
# text_color = const.BGR['leafy_green']
# o_x, o_y = 10, 20
# step_px = 30

# fullFrms = make_point_descriptor_box(myPnt, dims = (cols, rows), origin = (o_x, o_y), step_px = step_px,
#                                      back_color = back_color, text_color = (255, 255, 255), text_size = 1)

# DecapStream(fullFrms, 0, 100).show()

# over_x, over_y = 1635, 650
# tlPt = Point2D((over_x, over_y))
# overFrms = overlay(animFrms, fullFrms, top_left_point = tlPt)

# overFrms[-1].show()
# DecapStream(overFrms, 0, 100).show()



# overFrms[-1].show()

# filtPnts4

