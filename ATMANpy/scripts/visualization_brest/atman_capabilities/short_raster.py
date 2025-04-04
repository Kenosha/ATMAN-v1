
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
from ATMANpy.types.Overlays import FramesRaster
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

SAVE = False

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'short_raster'
SAVE_FOLDER = 'atman_capabilities'
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


#############################
######### EXPERIMENT ########
#############################

### PARAMS ###
mark_size = 6
mark_color = const.BGR['bright_blue']
heat_color = const.BGR['leafy_green']
bandwidth = 40
intermediate_lambdas = [0.25, 0.5, 0.75]

# prmDict = {'background_resolution': 0.1, 'argument_resolution': 0.1, 'foreground_resolution': 0.1,
#            'interpolate_func': vs.interpolate_camera_naive}
prmDict = {'background_resolution': 2, 'argument_resolution': 10, 'foreground_resolution': 10,
           'interpolate_func': vs.interpolate_camera_naive}
prmDict_static = {k: v for k, v in prmDict.items() if k not in ['interpolate_func']}

### FILTER ###
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

def return_is_fair(somePnt):
    try:
        thirdShot = somePnt[2]
    except:
        thirdShot = None
    return somePnt.get_return().is_fair(thirdShot)

filtPnts_before_fair = filtPnts_before.filter(return_is_fair)
retShots_before_fair = filtPnts_before_fair.get_returns() ### unrectified!

### CAM ###

M_ext = ConstFeature.from_store(vid, sto, featname = 'M_ext').feature
M_int = ConstFeature.from_store(vid, sto, featname = 'M_int').feature

brestCam_third = Camera(M_int, M_ext, (640, 360))
brestCam_half = brestCam_third.resize(1.5)
brestCam_full = brestCam_third.resize(3)

cntPt = CourtLines.make().get_center_point()
cntPt3D = Point3D.from_point2D(cntPt, z = 0).left(670).back(770)
camPt3D = Point3D((-300, 2000, 600))
thirdCam = Camera.from_to(camPt3D, cntPt3D).zoom(1.1)

### COURT ###
courtLines = CourtLines.make()
courtScn = Scene.make_court(courtLines, net_mesh_density = 0.75) ### todo: rotate this if needed!
region_color = (60, 60, 60)
from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D
subReg = CourtLines.make().get_court_region().scale(1.25)
subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(region_color)
courtScn = courtScn.put(subReg3D)

### ANIMATION ###
def kde_func(someShots):    
    ### hill map
    kdeDict = { 'bandwidth': 40, 'normalize_density': True, 
                'one_color': heat_color, 'zero_color': heat_color, 
                'resolution': 2}
    return vs.kde_hillmap(someShots, **kdeDict)

def kde_func(someShots):
    ### just rings
    kdeDict = { 'bandwidth': 40, 'normalize_density': True, 
                'contour_color': heat_color, 'display_at_height': False,
                'resolution': 2, 'preserve_showarr_shape': True}
    props = [0.25, 0.5, 0.75]

    from functools import reduce
    cont_scenes = [vs.kde_contourmap(someShots, **kdeDict, threshold_proportion = prop)[0] for prop in props]
    contScn = reduce(DummyScene.combine, cont_scenes, DummyScene.make_empty())

    contScn = contScn.consolidate()

    return contScn

def kde_func(someShots, bandwidth = 40, normalize_density = True, display_at_height = False, resolution = 2,
             heat_bgr = const.BGR['intense_red'], props = [0.2, 0.4, 0.6, 0.8]):
    ### bulls eye look
    
    from ATMANpy.funcs.util_funcs import interpolate_tuples
    background_bgr = tuple(BGRFrame.from_video(cap, 1100).array[200, 350]) ### the color of the court.
    fill_colors = [interpolate_tuples(background_bgr, heat_bgr, itpl_lambda = p) for p in props]

    from functools import reduce
    cont_scenes = [vs.kde_contourmap(someShots, bandwidth = bandwidth,
                                     normalize_density = normalize_density, 
                                     display_at_height = display_at_height,
                                     resolution = resolution, 
                                     fill = True, zero_color = background_bgr,
                                     threshold_proportion = prop, contour_color = col)[0] for prop, col in zip(props, fill_colors)]
    contScn = reduce(DummyScene.combine, cont_scenes, DummyScene.make_empty())

    contScn = contScn.consolidate()

    return contScn

### TODO: maybe also a map that is like a marriage between heatmap and contourmap
# -> fixed colour areas, filled out within a contour
# -> kde_contourmap with fill = True!!!

centerPt = courtLines.get_center_point()
centerPt3D = Point3D.from_point2D(centerPt, z = 0)

currentFrm = View.make(thirdCam, courtScn, resolution = prmDict['background_resolution']).snap() ### initialize a "current" frame

rotation_animation_length = 25
fade_length = 10
fade_thresh = 0.025
dim_factor = 0.25

kdeDict = {'props': [0.25, 0.5, 0.75, 0.9], 'heat_bgr': const.BGR['intense_red'], 'bandwidth': 50, 'resolution': 10}

raster_frames = []
raster_rot_frames = []
raster_fade_frames = []
corner_frames = []
rotate_frames = []
fade_frames =   []
for idx, myPnt in enumerate(filtPnts_before_fair):
    print(idx)
    
    ### get previous shots and current shot (as Shots, i.e: list of length 1)
    prevShots, allShots = retShots_before_fair[:idx], retShots_before_fair[:idx+1]
    
    ### figure out orientation
    if len(prevShots) > 0:
        previous_orientation = prevShots[-1].hitPlayer.get_side()
    else:
        previous_orientation = None
    current_orientation = allShots[-1].hitPlayer.get_side()

    ### rectify, so we gather everything in one direction (for corner graphic)
    prevShots_rect_curr = prevShots.enforce_direction(current_orientation.capitalize())
    prevShots_rect_next = allShots.enforce_direction(current_orientation.capitalize())

    heatScn_rect_next = kde_func(prevShots_rect_next, **kdeDict)

    ### previous heat scene
    if len(prevShots) > 0:
        heatScn_rect_curr = kde_func(prevShots_rect_curr, **kdeDict)
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
        
        heatScn_rect_prev = kde_func(prevShots_rect_prev, **kdeDict)
        # heatScn_rect_prev = vs.kde_heatmap(prevShots_rect_prev, one_color = heat_color, zero_color = heat_color,
        #                                     bandwidth = bandwidth, resolution = 2, normalize_density = True)

        pis = np.linspace(0, np.pi, rotation_animation_length)

        frameList = []
        for pi in pis:
            heatScn_rect_rot = heatScn_rect_prev.rotate_flat(pi, origPt = centerPt3D)
            rotLns = courtLines.rotate(pi)
            rotReg = subReg.rotate(pi, origPt = centerPt)
            rotReg3D = Region3D.from_region2D(rotReg, z = -1).recolor(region_color)
            rotScn = Scene.make_court(rotLns, net_mesh_density = 0.75).put(rotReg3D)
            
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
    spline_scenes, bounce_scenes = vis_ut.get_shot_scenes(serve_and_return, mark_color = mark_color,
                                                          mark_size = mark_size)
    spline_and_bounce_scenes = [sScn.combine(bScn) for sScn, bScn in zip(spline_scenes, bounce_scenes)]

    bounce_scenes_return = [myScn.drop(0) if not myScn.is_empty() else myScn for myScn in bounce_scenes]
    bounce_scenes_only_return = [myScn for myScn in bounce_scenes_return if not myScn.is_empty()]
    num_bounce_scenes = len(bounce_scenes_only_return)
    num_flight_scenes = len(bounce_scenes_return) - num_bounce_scenes

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
    rasterFrms = vis_ut.animate_bounce_and_mark_scenes( serve_and_return, confs = confs, 
                                                        start_cam = brestCam_third, stop_cam = brestCam_third,
                                                        over = True, over_scale = 1, sub_region = False,
                                                        mark_size = mark_size, mark_color = mark_color,
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


joint_raster_frames = [a.join(b).join(c) for a, b, c in zip(raster_rot_frames, raster_frames, raster_fade_frames)]
joint_corner_frames = [a.join(b).join(c) for a, b, c in zip(rotate_frames, corner_frames, fade_frames)]
joint_corner_frames = [a[:, :1080//3, -1920//3:] for a in joint_corner_frames]

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

# vis_ut.show(full_raster_frames)

full_rstFrms = BGRFrames.from_frames(full_raster_frames) ### speed this up...

# DecapStream(full_rstFrms, 0, 100).show()
# full_rstFrms[-1].show()

###### ROTATE ONTO POINT ####

### chosen candidates
best_idx = 14
lastPnt = filtPnts_after[best_idx]
mtc_idx, pnt_idx = lastPnt.get_match_index(), lastPnt.get_point_index()

prevPnt1 = allPnts.get(mtc_idx, pnt_idx - 1)
prevPnt2 = allPnts.get(mtc_idx, pnt_idx - 2)
prevPnt3 = allPnts.get(mtc_idx, pnt_idx - 3)

frms3 = prevPnt3.get_frames()
frms2 = prevPnt2.get_frames()
frms1 = prevPnt1.get_frames()
frms0 = lastPnt.get_frames()

d32 = frms2[0] - frms3[-1]
d21 = frms1[0] - frms2[-1]
d10 = frms0[0] - frms1[-1]

interFrms10 = lastPnt.stream(confs = confs).widen(d10).decap().frames[:d10].rescale(3)
interFrms10_speed = vis_ut.speed_up_frames_progressively(interFrms10, ramp_up_frames = 150, ramp_down_frames = 100)
intermezzo_stop_early = 50 ### stop early so that we have the windup of the serve later.
interFrms10_speed_pre, interFrms_10_speed_post = interFrms10_speed[:-intermezzo_stop_early], interFrms10_speed[-intermezzo_stop_early:]
interFrms_10_speed_post_dim = interFrms_10_speed_post.dim(0.5)

retShots_before_fair_rectified = retShots_before_fair.enforce_direction('Top')
kde_whole = kde_func(retShots_before_fair_rectified, **kdeDict)

map_frames = 30
transparence = 0.05 ** (1 / (map_frames - 5))

from ATMANpy.types.Overlays import reverse_cascade
_, revFrms = reverse_cascade(myRst, keep_which = [(0, 2)], frames_per_element = 14, acceleration = 1.3)
# DecapStream(revFrms, 0, 100).show()

# myRst[0, 2] = None
# rastFrm = myRst.to_frame_raster().render()
# newFrm = interFrms_10_speed_post_dim[0]

# itplFrms = vs.interpolate_frames(rastFrm, newFrm, num_frames = map_frames)

# retShots_before_fair_rectified = retShots_before_fair.enforce_direction('Top')
# kde_whole = kde_func(retShots_before_fair_rectified)

# rotate_backFrms, _ = vs.animate_sequence([kde_whole for _ in range(map_frames)], start_cam = thirdCam, stop_cam = brestCam_full,
#                                         previous_foreground_scene = courtScn, cumulative_foreground_transparence = transparence,
#                                         animate_over = itplFrms,
#                                         **prmDict)
# DecapStream(rotate_backFrms, 0, 100).show()

prmDict['background_resolution'] = 12 ### first resolution at which the background is solid even in full cam.
prmDict_static['background_resolution'] = 12 ### first resolution at which the background is solid even in full cam.

####### SOME TESTING, DELETE #######

myScene = kde_whole
myCamera = brestCam_full

aView.snap()

####################################

rotate_backFrms_void, _ = vs.animate_cam_motion(kde_whole, start_cam = thirdCam, stop_cam = brestCam_full,
                                                background_scene = courtScn, num_frames = map_frames,
                                                **prmDict)

### idea: dim out the kde a bit, so we can re-establish what we're actually showing (serve out wide, deuce...)
kde_whole_dim = kde_whole.dim(0.5)
kde_whole_interp = vs.interpolate_dummy_scenes(kde_whole, kde_whole_dim, num_frames = map_frames)

rotate_backFrms_void_transp, _ = vs.animate_sequence(kde_whole_interp, start_cam = thirdCam, stop_cam = brestCam_full,
                                                        background_scene = courtScn,
                                                        **prmDict)
# DecapStream(rotate_backFrms_void_transp, 0, 100).show()

# kde_whole_dim = animDict['previous_dummy_scene']

### TODO: full heatmap with description of server, returner, trajectory of serve.
myPnts = filtPnts_before_fair.enforce_direction('Bot')
srvShots = myPnts.get_serves() 
retShots = myPnts.get_returns() 

exampleSplnSF = srvShots[2].feat_spline_cumul(color = const.BGR['ball_yellow'], thickness = 1.5)
example_splns = list(exampleSplnSF.values())

_, exampleSpLns = exampleSplnSF.last()
exampleSpLns = [ln for ln in exampleSpLns if ln is not None]

srv_x, srv_y = srvShots.get_player_hit_position().mean(axis = 0)
ret_x, ret_y = retShots.get_player_hit_position().mean(axis = 0) 

srv_y = srv_y + 15

srvPt = Point3D((srv_x, srv_y, 0)).annotate('Server').resize(3)
retPt = Point3D((ret_x, ret_y, 0)).annotate('Returner').resize(2)

diagFrms0, _ = vs.animate_static(Scene.make_empty(), static_cam = brestCam_full, num_frames = 25, 
                              background_scene = courtScn, **prmDict_static)
diagFrms1, _ = vs.animate_static(Scene.make_empty(), static_cam = brestCam_full, num_frames = 15, 
                              previous_dummy_scene = kde_whole_dim, 
                              annotated_points = [srvPt],
                              background_scene = courtScn, **prmDict_static)
diagFrms2, _ = vs.animate_static(Scene.make_empty(), static_cam = brestCam_full, num_frames = 15, 
                              previous_dummy_scene = kde_whole_dim, 
                              annotated_points = [srvPt, retPt],
                              background_scene = courtScn, **prmDict_static)
diagFrms3, _ = vs.animate_sequence(example_splns, start_cam = brestCam_full, stop_cam = brestCam_full,
                                    previous_dummy_scene = kde_whole_dim, 
                                    annotated_points = [srvPt, retPt],
                                    background_scene = courtScn, **prmDict_static)
diagFrms4_dim, _ = vs.animate_static(Scene(exampleSpLns), static_cam = brestCam_full, num_frames = 25, 
                              previous_dummy_scene = kde_whole_dim, 
                              annotated_points = [srvPt, retPt],
                              background_scene = courtScn, **prmDict_static)
diagFrms4_lit, _ = vs.animate_static(Scene(exampleSpLns), static_cam = brestCam_full, num_frames = 25, 
                              previous_dummy_scene = kde_whole, 
                              annotated_points = [srvPt, retPt],
                              background_scene = courtScn, **prmDict_static)

brighteningFrms = vs.interpolate_frames(diagFrms4_dim[0], diagFrms4_lit[0], num_frames = 15)
dimmingFrms = vs.interpolate_frames(diagFrms4_lit[0], diagFrms4_dim[0], num_frames = 15)

diagFrms4 = BGRFrames.from_frames([brighteningFrms, dimmingFrms, 
                                   brighteningFrms, dimmingFrms,
                                   brighteningFrms, dimmingFrms,
                                   brighteningFrms])

# diagFrms5 = BGRFrames.from_frame(brighteningFrms[-1], 25)

# diagFrms0[0].show()
# diagFrms1[0].show()
# diagFrms2[0].show()
# diagFrms3[0].show()
# DecapStream(diagFrms3, 0, 100).show()
# diagFrms4_dim[0].show()
# diagFrms4_lit[0].show()
# DecapStream(diagFrms4, 0, 100).show()
# diagFrms5[0].show()

diagFrms_all = BGRFrames.from_frames([diagFrms0, diagFrms1, diagFrms2, diagFrms3, diagFrms4])
# DecapStream(diagFrms_all, 0, 100).show()

#### more expressive diagram of the 

############ TESTING DIFFERENT LOOKS ###########

newFrm = interFrms_10_speed_post[0]
newFrm_dim = interFrms_10_speed_post_dim[0]

kde_whole = kde_func(retShots_before_fair_rectified, **kdeDict)

# View.make(brestCam_full, kde_whole, resolution = 1).snap(over_array = newFrm.array.copy()).show()

########### ANIMATE FINAL RALLY ###########
intermezzo_stop_early = 50
interFrms10_speed_pre, interFrms_10_speed_post = interFrms10_speed[:-intermezzo_stop_early], interFrms10_speed[-intermezzo_stop_early:]

interFrms_before, _ = vs.animate_sequence([kde_whole for _ in range(interFrms_10_speed_post.length())], 
                                            start_cam = brestCam_full, stop_cam = brestCam_full,
                                            animate_over = interFrms_10_speed_post,
                                            **prmDict_static)
# DecapStream(interFrms_before, 0, 100).show()

beforeFrms = lastPnt[:2].stream(confs = confs).maybe_to_stream().decap().frames.rescale(3)
afterFrms = lastPnt[2:].stream(confs = confs).maybe_to_stream().decap().frames.rescale(3)

kdeFrms_before, _ = vs.animate_sequence([kde_whole for _ in range(beforeFrms.length())], 
                                            start_cam = brestCam_full, stop_cam = brestCam_full,
                                            animate_over = beforeFrms,
                                            **prmDict_static)

brestCam_full_traj = Camera.calibrated_brest_behind().resize(3)

spline_scenes, bounce_scenes = vis_ut.get_shot_scenes(lastPnt[:2], mark_color = mark_color, mark_size = mark_size)

markFrms_before, _ = vs.animate_sequence(bounce_scenes, start_cam = brestCam_full_traj, stop_cam = brestCam_full_traj,
                                    animate_over = kdeFrms_before, 
                                    **prmDict)

# DecapStream(markFrms_before, 0, 100).show()

animFrms_before, _ = vs.animate_sequence(spline_scenes, start_cam = brestCam_full_traj, stop_cam = brestCam_full_traj, 
                                            animate_over = markFrms_before,
                                            cumulative_foreground_transparence = 0.7, cumulate_argument = True,
                                            **prmDict)
# DecapStream(animFrms_before, 0, 100).show()
animFrms_before_with_inter = BGRFrames.from_frames([interFrms_before, animFrms_before])
# DecapStream(animFrms_before_with_inter, 0, 100).show()

animFrms_after = vis_ut.animate_bounce_and_mark_scenes(lastPnt[2:], brestCam_full_traj, brestCam_full_traj, confs, over = True, 
                                                sub_region = False, 
                                                mark_size = mark_size, mark_color = mark_color,
                                                spline_cumulative_transparence = 0.7, **prmDict)
# DecapStream(animFrms_after, 0, 100).show()
widen_by = 100
finalFrms = lastPnt.stream(confs = confs).widen(widen_by).decap().frames[-widen_by:].rescale(3)
# DecapStream(finalFrms, 0, 100).show()
animFrms_after_with_final = BGRFrames.from_frames([animFrms_after, finalFrms])
# DecapStream(animFrms_after_with_final, 0, 100).show()
animFrms_after_with_final_dim = animFrms_after_with_final.dim_progressively(1, 0, exp = 0.5)
# DecapStream(animFrms_after_with_final_dim, 0, 100).show()

fadeinFrms = vs.interpolate_frames(rotate_backFrms_void[-1], interFrms_before[0], 50)
# DecapStream(fadeinFrms, 0, 100).show()

### make it so the players are above the heatmap

cntSF = StreamFeatures.from_store(vid, sto, featnameList = ['botPlayerCnt', 'cand2Cnts'])

frms = lastPnt.get_frames()
pre_frms = list(range(frms[0] - intermezzo_stop_early, frms[0]))
frms_extended = [*pre_frms, *frms]

sub_cntSF = cntSF.conform(frms_extended)
myStm = Stream.from_SF(sub_cntSF, cap)

comb_frames = []
idx = 0
for frm_num, myFrm, featDict in myStm.generator():
    if idx >= animFrms_before_with_inter.length():
        break
    
    plCnt = featDict['botPlayerCnt']
    if plCnt is None:
        plMask = MaskFrame.false(like = myFrm)
    else:
        plMask = myFrm.within_contour(plCnt)
    plMask = plMask.rescale(3)

    ballMask = MaskFrame.false(like = myFrm)
    ballCnts = featDict['cand2Cnts']

    if ballCnts is not None:
        for cnt in featDict['cand2Cnts']:
            ballMask = ballMask | myFrm.within_contour(cnt)
    
    ballMask = ballMask.rescale(3)
    jointMask = plMask | ballMask

    combFrm = animFrms_before_with_inter[idx].mask(~jointMask) + myFrm.rescale(3).mask(jointMask) 
    comb_frames.append(combFrm)

    idx += 1

combFrms = BGRFrames.from_single_frames(comb_frames)
# DecapStream(combFrms, 0, 100).show()

### hold
staticFrms = BGRFrames.from_frame(combFrms[-1], 100)

### fade
ls_frm = lastPnt[1].get_frames()[-2]
lsFrm = Stream(cap, ls_frm, ls_frm).get(ls_frm).rescale(3)
itplFrms = vs.interpolate_frames(combFrms[-1], lsFrm, num_frames = 50)

### join
jointFrms = BGRFrames.from_frames([combFrms, staticFrms, itplFrms])

# DecapStream(jointFrms, 0, 100).show()

#### END SCREEN ####

from ATMANpy.types.Overlays import FrameRaster
myRst = FrameRaster(2, 2)

myFrm = BGRFrame.make_black(960, 540)

textFrm0 = myFrm.put('More Videos', color = const.BGR['bright_grey'], size = 3, where = (325, 280))
textFrm0 = textFrm0.framed(4, const.BGR['bright_grey'])
# textFrm0.show()

textFrm1 = myFrm.put('The Automated Tennis Match ANnotator', size = 2, where = (150, 280))
textFrm1 = textFrm1.framed(4, const.BGR['bright_grey'])
# textFrm1.show()

textFrm2 = myFrm.put('On the Effectiveness of the Chipped Return', size = 2, where = (100, 280))
textFrm2 = textFrm2.framed(4, const.BGR['bright_grey'])
# textFrm2.show()

textFrm3 = myFrm.put('On the Impact of Pressure on Serve Selection', size = 2, where = (80, 280))
textFrm3 = textFrm3.framed(4, const.BGR['bright_grey'])
# textFrm3.show()

myRst.add(textFrm0, 0, 0)
rstFrm0 = myRst.render()
# rstFrm0.show()

myRst.add(textFrm1, 1, 0)
rstFrm1 = myRst.render()
# rstFrm1.show()

myRst.add(textFrm2, 0, 1)
rstFrm2 = myRst.render()
# rstFrm2.show()

myRst.add(textFrm3, 1, 1)
rstFrm3 = myRst.render()
# rstFrm3.show()

############ ADD TO ALL FRAMES ##########
# available frames in order: 
# full_rstFrms -> rasters, up until cam movement
# revFrms -> reverse raster the frame elements of play
# rotate_backFrms_void -> move the schematic view to center
# diagFrms0, diagFrms1, diagFrms2, diagFrms3, diagFrms4, diagFrms5 -> for re-explaining the setup
# fadeinFrms -> fade in play.
# combFrms -> serve and return, with heatmap overlayed
# staticFrms -> freeze on return impact
# itplFrms -> disappear heatmap
# animFrms_after_with_final_dim -> play out rest of the point

all_frames = []
script_lines = []

script_line =   """
                    With ATMAN, one can filter for occurences in previous matches of this tournament
                    in which Brandon Nakashima faced a similar serve, and take a look at where his return
                    previously landed.
                """ # 16 seconds

ramp_up_frames = joint_raster_frames[0].length() + joint_raster_frames[1].length()
ramp_down_frames = 1

full_rstFrms_accel = vis_ut.speed_up_frames_progressively(full_rstFrms, 
                                                          ramp_up_frames = ramp_up_frames,
                                                          ramp_down_frames = ramp_down_frames)
# DecapStream(full_rstFrms_accel, 0, 100).show()

all_frames.append(full_rstFrms_accel)
script_lines.append(script_line)

script_line =   """
                    In the course of his previous matches in this tournament, 
                    when facing deuce serves out wide...
                """ # 5 seconds

jointFrms = BGRFrames.from_frames([revFrms, rotate_backFrms_void_transp, diagFrms1, diagFrms2, diagFrms3])
# DecapStream(jointFrms, 0, 100).show()
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    ... it seems that Brandon Nakashima typically goes for a safe return played to the center of the court
                    right back at the server, or a down-the line return to the backhand.
                """ # 9 seconds
# DecapStream(diagFrms4, 0, 100).show()
all_frames.append(diagFrms4)
script_lines.append(script_line)

### idea: cascade away the other frames, and rotate the heatmap in virtual space...

script_line =   """
                    Let's return to our serve in question.
                """ # 3 seconds
### ... then, blend it into the court
all_frames.append(fadeinFrms)
script_lines.append(script_line)

script_line =   ""
all_frames.append(combFrms[:40])
script_lines.append(script_line)

script_line =   """
                    The ball lands close to where previous returns landed, going down the line to Joao Sousa's backhand.
                """ # 6 seconds
jointFrms = BGRFrames.from_frames([combFrms[40:], staticFrms])
all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   ""
all_frames.append(itplFrms)
script_lines.append(script_line)

script_line =   """
                    This example is a short but indicative look into the capabilities of ATMAN, 
                    the Automated Tennis Match Annotator. 
                """ # 8 seconds
all_frames.append(animFrms_after_with_final_dim)
script_lines.append(script_line)

script_line =   """
                    If you would like to learn more about its workings...
                """ # 4 seconds

rstFrms0 = BGRFrames.from_frame(rstFrm0, 75)
all_frames.append(rstFrms0)
script_lines.append(script_line)

script_line =   """
                    ... consult the longer version of this video.
                """ # 4 seconds

rstFrms1 = BGRFrames.from_frame(rstFrm1, 100)
all_frames.append(rstFrms1)
script_lines.append(script_line)

script_line =   """
                    If you would like to see it employed in actual case studies,
                    take a look at the analysis videos.
                """ # 8 seconds

rstFrms2 = BGRFrames.from_frame(rstFrm2, 100)
rstFrms3 = BGRFrames.from_frame(rstFrm3, 100)
jointFrms = BGRFrames.from_frames([rstFrms2, rstFrms3])

all_frames.append(jointFrms)
script_lines.append(script_line)


vis_ut.show(all_frames)


### Saving the separate MultiFrame objects and the Joint Script
if SAVE:
    assert len(script_lines) == len(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        subFrms.save(fullPath, f'seg_{idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{idx}.txt'), mode = 'w') as f:
            f.write(script_line)




