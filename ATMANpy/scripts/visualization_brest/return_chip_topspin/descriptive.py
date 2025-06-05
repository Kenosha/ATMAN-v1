
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


import ATMANpy.auxil.constants as const
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Shots, read_in_data_tables, read_in_placeholder_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line, Region
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature, StreamFeature
from ATMANpy.types.Visual3D import DummyScene
from ATMANpy.types.Shapes3D import Line3D, Circle3D, Circles3D, Region3D, Regions3D
from ATMANpy.types.Overlays import FrameRaster, FramesRaster, Table
from ATMANpy.types.Frames import BGRFrame, MaskFrame

from statsmodels.discrete.discrete_model import Logit

from ATMANpy.funcs.util_funcs import interpolate_tuples

#############################
#### READ IN CONFIGS ########
#############################
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import collect_video_frames
import ATMANpy.scripts.visualization_brest.return_chip_topspin.utils as vis_ut

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

###################################
######### READ IN DATA ############
###################################
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

firstPnts = dataPnts.filter('is_first_serve')

### Remove aces, faults and net balls.
def get_returned_balls(myPnt):
    return len(myPnt) > 1
all_retPnts = firstPnts.filter(get_returned_balls)

############################################
####### FILTER OUT NON-DECISIVE POINTS #####
############################################
# store the winner codes, so we don't have to rerun this all the time
codes_path = os.path.join(get_folder_struct('ATMANpy')['scripts'], 'visualization_brest', 'return_chip_topspin')
codes_name = 'codes.txt'
codes_file = os.path.join(codes_path, codes_name)

if codes_name in os.listdir(codes_path):
    all_winSer = pd.read_csv(codes_file, index_col = 0).squeeze()
else:
    winner_codes = vis_ut.get_winners_from_score(all_retPnts, allPnts)
    all_winSer = pd.Series(winner_codes, name = 'winner_code')
    all_winSer.to_csv(codes_file)

dec_retPnts = Points([all_retPnts.pointList[idx] for idx in range(len(all_retPnts)) if all_winSer[idx] != 0])
dec_winSer = all_winSer[all_winSer != 0].reset_index(drop = True)

############################################
####### FILTER OUT PATHOLOGICAL BALLS ######
############################################
# a) serve speed too low!
# b) no bounce recorded!
# c) too many frames!
THRESHOLD_SPEED = 140
THRESHOLD_FRAMES = 35

retPnts = dec_retPnts.filter(vis_ut.is_not_pathological, threshold_speed = THRESHOLD_SPEED, threshold_frames = THRESHOLD_FRAMES)
pathPnts = dec_retPnts.filter(vis_ut.is_pathological, threshold_speed = THRESHOLD_SPEED, threshold_frames = THRESHOLD_FRAMES)

pathSer = pd.Series(dec_retPnts.apply(vis_ut.is_pathological))
winSer = dec_winSer[pathSer != 1].reset_index(drop = True)

##### ENFORCE DIRECTION
change_direction_array = retPnts.apply('is_top_to_bot', dtype = bool) ### for convenience, store whether we'll change the direction of the point.
indices = [(pt.get_match_index(), pt.get_point_index()) for pt in retPnts]
idxFrm = pd.DataFrame(indices, columns = ['match_idx', 'point_idx'])
idxFrm['is_top_to_bot'] = change_direction_array

retPnts = retPnts.enforce_direction('Top') ### Server is at the Bottom, Receiver at the Top. UPDATE: HAS BEEN SWITCHED!
indices_new = [(pt.get_match_index(), pt.get_point_index()) for pt in retPnts]
idxFrm_new = pd.DataFrame(indices, columns = ['match_idx', 'point_idx'])
idxFrm_new = idxFrm_new.merge(idxFrm, on = ['match_idx', 'point_idx'])

######################################
### PROBABILITY TO CHIP ESTIMATION ###
######################################
### Have to filter out chipped returns on the stretch -> compare only returns which are chipped by choice!
### But then, on the other hand -> if we try to match them to topspin returns, we won't find any, since on the stretch, most returns are blocked!
### -> automatically reduced!!

### Logistic Regression 
varDict = vis_ut.get_psm_var_dict(retPnts)
y = varDict.pop('is_chipped')
X = np.stack(list(varDict.values()), axis = 1)

xFrm = pd.DataFrame(X, columns = varDict.keys())
xFrm['const'] = 1
ySer = pd.Series(y, name = 'is_chipped').astype(bool)

xFrm = xFrm.drop(['serve_width', 'serve_spin'], axis = 1) ### these two are highly insignificant

logReg = Logit(ySer, xFrm)
logReg = logReg.fit()
logReg.summary()

#########################################
##### CREATE REGRESSORS AND WEIGHTS #####
#########################################
DECAY_RATE = 0.9
regFrm = vis_ut.make_regressor_frame(retPnts, logReg, winSer, idxFrm_new, decay_rate = DECAY_RATE)

#### From here on out, we look at returns made
# -> filter!
in_col = 'return_is_fine' #'did_bounce_in'
retPnts_in = retPnts.filter(regFrm[in_col].values)
regFrm_in = regFrm[regFrm[in_col]]

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'descriptive'
SAVE_FOLDER = 'return_chip_topspin'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

current_idx = 0

########################################
############# PARAMETERS ###############
########################################

def make_prm_dict(resolution_modifier = 1, argument_luminosity = 1, adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None):
    resolution_background = 1 * resolution_modifier
    resolution_objects = 1 * resolution_modifier
    resolution_kde = 2 * resolution_modifier

    return {'background_resolution': resolution_background,
            'argument_resolution': resolution_objects,
            'foreground_resolution': resolution_objects,
            'dummy_resolution': resolution_kde,
            'adaptive_resolution_exponent': adaptive_resolution_exponent,
            'argument_luminosity': argument_luminosity,
            'argument_blinks_per_second': argument_blinks_per_second,
            }


RESOLUTION_MODIFIER_NORMAL = 15 #15
RESOLUTION_MODIFIER_HEATMAP = 1.25
RESOLUTION_MODIFIER_ANGLE = 15
prmDict = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_NORMAL, argument_luminosity = 1.25, 
                        adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
prmDict_nolum = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_NORMAL, argument_luminosity = 1, 
                              adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
prmDict_heat = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_HEATMAP, argument_luminosity = 1, 
                             adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
prmDict_angle = make_prm_dict(resolution_modifier = RESOLUTION_MODIFIER_ANGLE, argument_luminosity = 1, 
                             adaptive_resolution_exponent = 1.5, argument_blinks_per_second = None)
RESOLUTION_KDE = prmDict_heat['dummy_resolution']

MIX_COLORS_KDE = True
BANDWIDTH_KDE = 35
SCENE_FADE = 0.9
CONTOUR_PROP = 0.5
EPSILON = 0.002
EMPHASIS_DIM_FACTOR = 1.5
ADAPTIVE_RESOLUTION_EXPONENT = 1.5
BLINK_RATE = 1

STATIC_FRAMES = 100 ### does not matter much once we record: the static frames will be fit to the narration
TRANSITION_FRAMES = 25 

INVISIBLE_ALPHA = 0.01 ### which alpha is accepted as invisible
FULL_ALPHA = 1 ### which alpha is accepted as fully visible --> FULL_ALPHA as parameter not needed except for FADE_IN, FADE_OUT_FACTOR calculation

FADE_IN_FACTOR = (FULL_ALPHA / INVISIBLE_ALPHA) ** (1 / TRANSITION_FRAMES)
FADE_OUT_FACTOR = 1 / FADE_IN_FACTOR ### easy to show that this must hold.

BALL_COLOR = const.BGR['ball_yellow']
CHIP_LO_COLOR = interpolate_tuples(const.BGR['marine_blue'], (0, 0, 0), 0.1)
CHIP_HI_COLOR = interpolate_tuples(const.BGR['bright_blue'], (255, 255, 255), 0.3)
TOPS_COLOR = const.BGR['light_brown']

BANDWIDTH = 45

FRAME_THICKNESS = 2
FRAME_COLOR = const.BGR['bright_grey']

########################################
###### LOOK AT RETURNS MADE ############
########################################
# -> idea: 
    # -> take a video serve and show the chipped return, as an example
    # -> as the chipped return flies, the video interpolates to virtual space, to show the trajectory, and where the ball lands.
    # (or maybe: superimpose the ball feature (with spline?) on the real video, then freeze, then fade to virtual space)
    # as soon as we are in virtual space, mark out where the ball lands.
    # then, show many more marks, and fade into heatmap
    ### -> explain heatmaps, yada yada yada
    # then, back to trajectry arc, and move camera to side of the returning player.
    ### -> launch angles!

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)

from ATMANpy.types.Visual3D import Camera, Scene, View
from ATMANpy.types.Streams import StreamList
from ATMANpy.types.Features import SFList

### Shots
dirFrm = regFrm[(regFrm['is_top_to_bot'])]
low_pred_exFrm = dirFrm[dirFrm.is_chipped].sort_values('pred_is_chipped').iloc[:8]
idx = int(low_pred_exFrm.index[0])
myShots = retPnts[idx][:2]

retShots = retPnts_in.get_returns()
retShots_rectified = vis_ut.rectify_shots(retShots)
chip_retShots_rectified = retShots_rectified.filter('is_slice_visual')
tops_retShots_rectified = retShots_rectified.filter('is_slice_visual', negate = True)

### Real Stream
myDtm = myShots.stream(confs = confs).join().maybe_to_stream().decap()
scaledFrms = myDtm.frames.rescale(3)

pre_frames = 75
preDtm = myShots[0].stream(confs = confs).widen(pre_frames).decap()
preFrms = preDtm.frames[:pre_frames].rescale(3)
# DecapStream(preFrms, 0, 100).show()

preFrms_dimmed = preFrms.dim_progressively(0, 1, exp = 2)
# DecapStream(preFrms_dimmed, 0, 100).show()

### Cams
brestCam = Camera.calibrated_brest_behind()
brestCam_resized = brestCam.resize(3)

from ATMANpy.auxil.court import cnt_2D, top_cnt_2D
from ATMANpy.types.Shapes3D import Point3D
p_orig, p_target = Point3D.from_point2D(cnt_2D, z = 100), Point3D.from_point2D(top_cnt_2D, z = 0)
heatCam = Camera.from_to(p_orig, p_target, f = 1000, sensor_dims = (1920, 1080)).zoom(1)

### Static Scenes
courtScn = Scene.make_court()

### Legends
from ATMANpy.types.Overlays import Legend, LegendRow, overlay, overlay_center

from ATMANpy.funcs.util_funcs import interpolate_tuples
CHIP_COLOR = interpolate_tuples(CHIP_LO_COLOR, CHIP_HI_COLOR)

from ATMANpy.types.Overlays import Legend, LegendRow, overlay
tpRow = LegendRow('... Topspin', shape_color = TOPS_COLOR, text_size = 2)
loRow = LegendRow('... Low Chip', shape_color = CHIP_LO_COLOR, text_size = 2)
hiRow = LegendRow('... High Chip', shape_color = CHIP_HI_COLOR, text_size = 2)
cpRow = LegendRow('... (Choice) Chip', shape_color = CHIP_COLOR, text_size = 2)

### faint colored rows
TOPS_COLOR_FAINT = interpolate_tuples(TOPS_COLOR, (0, 0, 0), 0.5)
CHIP_COLOR_FAINT = interpolate_tuples(CHIP_COLOR, (0, 0, 0), 0.5)
tpRow_faint = LegendRow('... Topspin', shape_color = TOPS_COLOR_FAINT, text_size = 2, text_color = (110, 110, 110))
cpRow_faint = LegendRow('... (Choice) Chip', shape_color = CHIP_COLOR_FAINT, text_size = 2, text_color = (110, 110, 110))

leg_dims = (375, 300)
# leg_dims = (500, 500)
leg_params = {'borders': True, 'header_text': 'Return is ...', 'resolution': 200, 'render_height': 300,
              'legend_dimensions': leg_dims, 'header_offset': 2, 'header_thickness': 1}

leg1 = Legend([hiRow, LegendRow(None), LegendRow(None)], **leg_params)
leg1Frm = leg1.render()
# leg1Frm.show()

leg2 = Legend([hiRow, loRow, LegendRow(None)], **leg_params)
leg2Frm = leg2.render()
# leg2Frm.show()

leg3 = Legend([hiRow, loRow, tpRow], **leg_params)
leg3Frm = leg3.render()
# leg3Frm.show()

leg_base = Legend([cpRow, LegendRow(None)], **leg_params)
legFrm_chip = leg_base.render()
# legFrm_chip.show()

leg_base = Legend([tpRow, cpRow], **leg_params)
legFrm = leg_base.render()
# legFrm.show()

leg_base = Legend([tpRow, cpRow_faint], **leg_params)
legFrm_tops = leg_base.render()
# legFrm_tops.show()

leg_base = Legend([tpRow_faint, cpRow], **leg_params)
legFrm_chip = leg_base.render()
# legFrm_chip.show()

### legends with circles
from ATMANpy.types.Overlays import make_circle_in
hiRow.shape_offset = 0.35
loRow.shape_offset = 0.35
tpRow.shape_offset = 0.35

leg1_crc_lo = Legend([LegendRow(None), loRow, LegendRow(None)], **leg_params, shape_func = make_circle_in)
leg1Frm_crc_lo = leg1_crc_lo.render()
# leg1Frm_crc_lo.show()

leg1_crc_hi = Legend([hiRow, LegendRow(None), LegendRow(None)], **leg_params, shape_func = make_circle_in)
leg1Frm_crc_hi = leg1_crc_hi.render()
# leg1Frm_crc_hi.show()

leg2_crc = Legend([hiRow, loRow, LegendRow(None)], **leg_params, shape_func = make_circle_in)
leg2Frm_crc = leg2_crc.render()
# leg2Frm_crc.show()

leg3_crc = Legend([hiRow, loRow, tpRow], **leg_params, shape_func = make_circle_in)
leg3Frm_crc = leg3_crc.render()
# leg3Frm_crc.show()

subReg = Region3D.from_points3D(Point3D((10, 10, 0)), Point3D((50, 10, 0)), Point3D((50, 50, 0)), Point3D((10, 50, 0)))

### anchor points for legend
anchorPt = Point2D((1920 // 24, 1080 // 24))
anchorPt_middle = Point2D((1200, 200))

### Flight Splines
serveSF = myShots[0].feat_spline_cumul(which_endpoint = 'penultimate')
returnSF = myShots[1].feat_spline_cumul(which_endpoint = 'penultimate')

serve_scenes = [Scene([spLn.recolor(BALL_COLOR).resize(1.5) for spLn in splines if spLn is not None]) for splines in serveSF.values()]
return_scenes = [Scene([spLn.recolor(BALL_COLOR).resize(1.5) for spLn in splines if spLn is not None]) for splines in returnSF.values()]
spline_scenes = [*serve_scenes, *return_scenes]
splnScn = spline_scenes[-1]

# ballSF = myShots.feat_ball(which_endpoint = 'penultimate')
# ballSF = myShots.feat_spline(which_endpoint = 'penultimate')
# topSF = myShots.feat_top_player()
# botSF = myShots.feat_bot_player()
# jntSF = ballSF.merge(topSF.rename('Top')).merge(botSF.rename('Bot'))

chipFrm = regFrm_in[regFrm_in.is_chipped == 1]
topspin_preds = chipFrm.pred_is_topspin.values


from ATMANpy.auxil.court import top_bsl_deu_2D, top_bsl_ad_2D
fhPt = Point3D.from_point2D(top_bsl_deu_2D, z = 0).front(10).annotate('Forehand')
bhPt = Point3D.from_point2D(top_bsl_ad_2D, z = 0).front(10).left(25).annotate('Backhand')

# make the lateral cam and intermediate cam over net
bPt = splnScn[0].leftPt ### origin of the shot
aPt = bPt.left(100)
sideretCam = Camera.from_to(aPt, bPt, f = 1000)

from ATMANpy.auxil.court import cnt_2D
cnt = Point3D.from_point2D(cnt_2D, z = 100)
intmCam = Camera.from_to(cnt, cnt.right(100).down(20))

def scene_to_graph_scene(splnScn, cutoff_frames = None):
    ### make it so that the scene looks good with the graph, and is easier to render
    if cutoff_frames is not None:
        splnShapes = splnScn.shapes[:cutoff_frames]
    else:
        splnShapes = splnScn.shapes

    splnShapes = [shape.resize(0.25) for shape in splnShapes]
    return Scene(splnShapes)

# _, topPly = topSF.last()
# _, botPly = botSF.last()
# plyScn = Scene([topPly, botPly])

DO_METHOD_SECTION = True
DO_HEATMAP_SECTION = True
DO_LAUNCH_ANGLE_INTRO = True
DO_LAUNCH_ANGLE_DISTRO = True
DO_LAUNCH_ANGLE_MODES = True
DO_LAUNCH_ANGLE_EXAMPLE = True

###############################
##### METHODOLOGY SECTION #####
###############################
if DO_METHOD_SECTION:
    all_frames = []
    script_lines = []

    script_line =   """
                        We build up our understanding of a good chip by looking at some characteristics of topspin and chip returns in general. 
                        First up: placement.
                    """ # 9 seconds
    # blankFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))
    # blankFrms = BGRFrames.from_frame(blankFrm, 25)

    ####### FLIGHT SUPERIMPOSED OVER BALL
    flightFrms, animDict = vs.animate_sequence(spline_scenes, start_cam = brestCam_resized, stop_cam = brestCam_resized, background_scene = None, 
                                        cumulate_argument = False, animate_over = scaledFrms, **prmDict)
    lastFrm = flightFrms[-1]

    flight_statFrms = BGRFrames.from_frame(lastFrm, 15)

    ####### MORPH TRAJECTORY TO VIRTUAL 
    staticScn = courtScn.combine(splnScn)
    virtualFrm = View.make(brestCam_resized, staticScn).snap()
    itplFrms = vs.interpolate_frames(lastFrm, virtualFrm, num_frames = 35)

    moveFrms, animDict = vs.animate_cam_motion(splnScn, start_cam = brestCam_resized, stop_cam = heatCam, background_scene = courtScn,
                                            num_frames = TRANSITION_FRAMES, interpolate_func = vs.interpolate_camera_slerp, **prmDict)

    move_statFrms = BGRFrames.from_frame(moveFrms[-1], 25)
        
    jointFrms = BGRFrames.from_frames([preFrms_dimmed, flightFrms, flight_statFrms, itplFrms, moveFrms, move_statFrms])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    ##### MAKE ALL BALL MARKS ####
    from ATMANpy.types.ShapesTennis3D import Mark

    # a) the visualized mark
    bouncePt = Point3D.from_point2D(Point2D(myShots[1].get_first_bounce_position()), z = 0)
    bounceMk = Mark.make(bouncePt, size = 3, color = const.BGR['ball_yellow'])

    # b) the undifferentiated, unrectified marks
    markScn_mono = vs.scatter_markmap(retShots, one_color = const.BGR['ball_yellow'])
    markScn_mono_trans = markScn_mono.realpha(INVISIBLE_ALPHA)

    # c) the undifferentiated, rectified marks
    markScn_mono_rect = vs.scatter_markmap(retShots_rectified, one_color = const.BGR['ball_yellow'])

    # d1) the differentiated, rectified marks
    markScn_poly_rect = vs.scatter_markmap(retShots_rectified, color_func_or_arr = 'is_slice_visual', 
                                            zero_color = const.BGR['light_brown'], one_color = const.BGR['marine_blue'])

    # d2) the differentiated, rectified marks, separated
    chip_markScn = vs.scatter_markmap(chip_retShots_rectified, one_color = const.BGR['marine_blue'])
    tops_markScn = vs.scatter_markmap(tops_retShots_rectified, one_color = const.BGR['light_brown'])

    ####### FADE OUT FLIGHT LINE
    script_line = "As in the case of this example return, we collect the bounce position of all returns in our sample."
    mark_scenes = [bounceMk for _ in range(TRANSITION_FRAMES)]
    fadeoutFrms, animDict = vs.animate_sequence(mark_scenes, start_cam = heatCam, stop_cam = heatCam, cumulative_foreground_transparence = FADE_OUT_FACTOR,
                                        background_scene = courtScn, **animDict, **prmDict)
    
    statFrms1 = BGRFrames.from_frame(fadeoutFrms[-1], TRANSITION_FRAMES)

    ####### FADE IN OTHER MARKS
    animDict['previous_foreground_scene'] = markScn_mono_trans
    fadeinFrms, animDict = vs.animate_sequence(mark_scenes, background_scene = courtScn, cumulate_argument = False,
                                        cumulative_foreground_transparence = FADE_IN_FACTOR, start_cam = heatCam, stop_cam = heatCam,
                                        **animDict, **prmDict)

    statFrms2 = BGRFrames.from_frame(fadeinFrms[-1], TRANSITION_FRAMES)

    jointFrms = BGRFrames.from_frames([fadeoutFrms, statFrms1, fadeinFrms, statFrms2])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    script_line =   """
                        We then flip returns hit back to lefty servers, so that the server's forehand side is always at deuce. 
                        This way, we can check whether returns are hit to the backhand side preferentially.
                    """
    #### Switch Over Returns to Lefties so we have forehand/backhand
    itpl_markScns = vs.interpolate_scenes(markScn_mono, markScn_mono_rect, num_frames = 25)
    flipFrms, animDict = vs.animate_sequence(itpl_markScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, **prmDict)
    
    statFrms, animDict = vs.animate_static(markScn_mono_rect, static_cam = heatCam, background_scene = courtScn, annotated_points = [fhPt, bhPt], **prmDict)

    jointFrms = BGRFrames.from_frames([flipFrms, statFrms])

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    script_line =   """
                        We mark topspin returns in orange, and chip returns in blue.
                    """
    itpl_markScns = vs.interpolate_scenes(markScn_mono_rect, markScn_poly_rect, num_frames = 25)
    markFrms, animDict = vs.animate_sequence(itpl_markScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, 
                                             annotated_points = [fhPt, bhPt], **prmDict)
    markFrms = overlay(markFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(markFrms)
    script_lines.append(script_line)

    script_line = "First, consider the empirical distribution of topspin returns."
    
    topsFrms, animDict = vs.animate_static(tops_markScn, background_scene = courtScn, static_cam = heatCam, 
                                    annotated_points = [fhPt, bhPt], **prmDict)
    topsFrms = overlay(topsFrms, legFrm_tops, top_left_point = anchorPt)
    # topsFrms[-1].show()

    itplFrms = vs.interpolate_frames(markFrms[-1], topsFrms[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    all_frames.append(itplFrms)
    script_lines.append(script_line)

    script_line = "Topspin returns are symmetrically clustered around the center of the court..."
    
    from ATMANpy.scripts.visualization_brest.atman_capabilities.utils import kde_func
    topsScn = kde_func(tops_retShots_rectified, bandwidth = 40, props = [0.2, 0.4, 0.6, 0.8],
                       resolution = RESOLUTION_KDE, heat_bgr = TOPS_COLOR, background_bgr = (0, 0, 0),
                       exp_bgr = 1, reg_scale_factor = 0.98)

    topsFrms, animDict = vs.animate_static(topsScn, background_scene = courtScn, static_cam = heatCam, **prmDict_nolum,
                                    previous_annotated_points = animDict['previous_annotated_points'])
    topsFrms = overlay(topsFrms, legFrm_tops, top_left_point = anchorPt)
    # topsFrms[-1].show()

    itplFrms = vs.interpolate_frames(itplFrms[-1], topsFrms[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    all_frames.append(itplFrms)
    script_lines.append(script_line)

    ### shares in sector
    share_color = const.BGR['leafy_green']

    a, b, c, d, dc, ad = vis_ut.get_court_sectors(side = 'Top')
    shrSer = vis_ut.shares_in_sector(tops_retShots_rectified, side = 'Top')
    shrStrs = [f'{share:>3.0%}' for share in shrSer.values]

    myRegs = [  Region3D.from_region2D(a, z = 0), Region3D.from_region2D(b, z = 0), 
                Region3D.from_region2D(c, z = 0), Region3D.from_region2D(d, z = 0), 
                Region3D.from_region2D(dc, z = 0), Region3D.from_region2D(ad, z = 0)]
    myLns = [regLn.recolor(share_color).resize(1) for reg in myRegs for regLn in reg.lineList]

    centerPts = [reg.center_point().annotate(shrStr).resize(2).recolor(share_color).left(10) for reg, shrStr in zip(myRegs, shrStrs)]

    script_line =   """
                        ... with a moderate preference for the server's backhand: 
                        Around 60% balls are hit to the backhand side of the court. 
                    """
    topsFrms_share, animDict = vs.animate_static(myLns, background_scene = courtScn, static_cam = heatCam, **prmDict_nolum, 
                                  **animDict, annotated_points = centerPts)
    topsFrms_share = overlay(topsFrms_share, legFrm_tops, top_left_point = anchorPt)
    
    itplFrms = vs.interpolate_frames(itplFrms[-1], topsFrms_share[0], 50)
    DecapStream(itplFrms, 0, 100).show()

    x, y, z = 1, 1, 3
    s = x + y + z
    shorten_factor = 0.125
    line_color = const.BGR['bright_grey']

    c0, c1, c2, c3, c4, c5 = centerPts

    c0 = c0.right(10).back(5)
    c1 = c1.right(7).back(5)
    c4 = c4.right(5).front(5)

    c2 = c2.right(10).back(5)
    c3 = c3.right(7).back(5)
    c5 = c5.right(5).front(5)

    midlfPt = (c0 * x + c1 * y + c4 * z) // s
    midrtPt = (c2 * x + c3 * y + c5 * z) // s

    l0 = Line3D(c0, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l1 = Line3D(c1, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l2 = Line3D(c2, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color) 
    l3 = Line3D(c3, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l4 = Line3D(c4, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l5 = Line3D(c5, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)

    midlfPt_annot = midlfPt.annotate('40%').recolor(share_color).resize(2).rethicken(2).left(5).back(2)
    midrtPt_annot = midrtPt.annotate('60%').recolor(share_color).resize(2).rethicken(2).left(5).back(2)

    annotPts_mid = [midlfPt_annot, midrtPt_annot]

    l0_null = Line3D(c0, c0).recolor(line_color)
    l1_null = Line3D(c1, c1).recolor(line_color)
    l2_null = Line3D(c2, c2).recolor(line_color)
    l3_null = Line3D(c3, c3).recolor(line_color)
    l4_null = Line3D(c4, c4).recolor(line_color)
    l5_null = Line3D(c5, c5).recolor(line_color)

    nullScn = Scene([l0_null, l1_null, l2_null, l3_null, l4_null, l5_null])
    lineScn = Scene([l0, l1, l2, l3, l4, l5])

    lineScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = 50)

    lineFrms, animDict = vs.animate_sequence(  lineScns, start_cam = heatCam, stop_cam = heatCam,
                                                background_scene = courtScn,
                                                **prmDict_nolum, **animDict)
    lineFrms = overlay(lineFrms, legFrm_tops, top_left_point = anchorPt)
    # DecapStream(lineFrms, 0, 100).show()

    statFrms, _ = vs.animate_static(Scene(  [l0, l1, l2, l3, l4, l5]), static_cam = heatCam,
                                            annotated_points = annotPts_mid, background_scene = courtScn,
                                            **prmDict_nolum, **animDict, num_frames = 25)
    statFrms = overlay(statFrms, legFrm_tops, top_left_point = anchorPt)
    # statFrms[-1].show()

    jointFrms = BGRFrames.from_frames([itplFrms, lineFrms, statFrms])

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    # Heatmap for Weighted, 'Choice' Chips 
    script_line = "On the other hand, for choice chipped returns, we see a more irregular heatmap; some land shallow, others deeper."
    
    chipScn = kde_func(chip_retShots_rectified, bandwidth = 40, props = [0.2, 0.4, 0.6, 0.8],
                       resolution = RESOLUTION_KDE, heat_bgr = CHIP_COLOR, background_bgr = (0, 0, 0),
                       exp_bgr = 1, reg_scale_factor = 0.98)

    chipFrms, animDict = vs.animate_static(chipScn, background_scene = courtScn, static_cam = heatCam, 
                                           annotated_points = [fhPt, bhPt], **prmDict_nolum)
    chipFrms = overlay(chipFrms, legFrm_chip, top_left_point = anchorPt)

    itplFrms = vs.interpolate_frames(statFrms[-1], chipFrms[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    all_frames.append(itplFrms)
    script_lines.append(script_line)

    ### shares in sector
    a, b, c, d, dc, ad = vis_ut.get_court_sectors(side = 'Top')
    shrSer = vis_ut.shares_in_sector(chip_retShots_rectified, side = 'Top')
    shrStrs = [f'{share:>3.0%}' for share in shrSer.values]

    myRegs = [  Region3D.from_region2D(a, z = 0), Region3D.from_region2D(b, z = 0), 
                Region3D.from_region2D(c, z = 0), Region3D.from_region2D(d, z = 0), 
                Region3D.from_region2D(dc, z = 0), Region3D.from_region2D(ad, z = 0)]
    myLns = [regLn.recolor(const.BGR['leafy_green']).resize(1) for reg in myRegs for regLn in reg.lineList]

    centerPts = [reg.center_point().annotate(shrStr).resize(2).recolor(const.BGR['leafy_green']).left(10) for reg, shrStr in zip(myRegs, shrStrs)]
    
    script_line = "As with topspin returns, chipped returns are hit predominantly to the backhand side; again, the split is around 60/40."
    
    chipFrms_share, animDict = vs.animate_static(myLns, background_scene = courtScn, static_cam = heatCam, **prmDict_nolum, 
                                                **animDict, annotated_points = centerPts)
    chipFrms_share = overlay(chipFrms_share, legFrm_chip, top_left_point = anchorPt)
    
    itplFrms = vs.interpolate_frames(itplFrms[-1], chipFrms_share[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    x, y, z = 1, 1, 3
    s = x + y + z
    shorten_factor = 0.125
    line_color = const.BGR['bright_grey']

    c0, c1, c2, c3, c4, c5 = centerPts

    c0 = c0.right(10).back(5)
    c1 = c1.right(7).back(5)
    c4 = c4.right(5).front(5)

    c2 = c2.right(10).back(5)
    c3 = c3.right(7).back(5)
    c5 = c5.right(5).front(5)

    midlfPt = (c0 * x + c1 * y + c4 * z) // s
    midrtPt = (c2 * x + c3 * y + c5 * z) // s

    l0 = Line3D(c0, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l1 = Line3D(c1, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l2 = Line3D(c2, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color) 
    l3 = Line3D(c3, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l4 = Line3D(c4, midlfPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)
    l5 = Line3D(c5, midrtPt).shorten_by(shorten_factor, which = 'right').recolor(line_color)

    midlfPt_annot = midlfPt.annotate('40%').recolor(share_color).resize(2).rethicken(2).left(5).back(2)
    midrtPt_annot = midrtPt.annotate('60%').recolor(share_color).resize(2).rethicken(2).left(5).back(2)

    annotPts_mid = [midlfPt_annot, midrtPt_annot]

    l0_null = Line3D(c0, c0).recolor(line_color)
    l1_null = Line3D(c1, c1).recolor(line_color)
    l2_null = Line3D(c2, c2).recolor(line_color)
    l3_null = Line3D(c3, c3).recolor(line_color)
    l4_null = Line3D(c4, c4).recolor(line_color)
    l5_null = Line3D(c5, c5).recolor(line_color)

    nullScn = Scene([l0_null, l1_null, l2_null, l3_null, l4_null, l5_null])
    lineScn = Scene([l0, l1, l2, l3, l4, l5])

    lineScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = 50)

    lineFrms, animDict = vs.animate_sequence(  lineScns, start_cam = heatCam, stop_cam = heatCam,
                                                background_scene = courtScn,
                                                **prmDict_nolum, **animDict)
    lineFrms = overlay(lineFrms, legFrm_chip, top_left_point = anchorPt)
    # DecapStream(lineFrms, 0, 100).show()

    statFrms, _ = vs.animate_static(Scene(  [l0, l1, l2, l3, l4, l5]), static_cam = heatCam,
                                            annotated_points = annotPts_mid, background_scene = courtScn,
                                            **prmDict_nolum, **animDict, num_frames = 25)
    statFrms = overlay(statFrms, legFrm_chip, top_left_point = anchorPt)
    # statFrms[-1].show()

    jointFrms = BGRFrames.from_frames([itplFrms, lineFrms, statFrms])

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    script_line =   """
                        Let us now look at a different facet of return characteristics: launch angle.
                    """
    # fade trajectory back in
    splnScn_thin = scene_to_graph_scene(splnScn)
    itplScns = vs.interpolate_scenes(splnScn_thin.realpha(0), splnScn_thin, num_frames = 25)

    splnFrms, _ = vs.animate_static(splnScn_thin, static_cam = heatCam, background_scene = courtScn, annotated_points = [fhPt, bhPt], **prmDict_nolum)

    itplFrms = vs.interpolate_frames(statFrms[-1], splnFrms[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    # cam motion -> made so that the motion shows also the entire trajectory!
    moveFrms1, animDict = vs.animate_cam_motion(splnScn_thin, num_frames = 50, annotated_points = [fhPt, bhPt],
                                            background_scene = courtScn, start_cam = heatCam, stop_cam = intmCam, **prmDict_nolum)

    moveFrms2, animDict =  vs.animate_cam_motion(splnScn_thin, num_frames = 50, annotated_points = [fhPt, bhPt],
                                            background_scene = courtScn, start_cam = intmCam, stop_cam = sideretCam, **prmDict_nolum)
    

    jointFrms = BGRFrames.from_frames([itplFrms, moveFrms1, moveFrms2])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    # show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)

####################################
######### HEATMAP SECTION ##########
####################################

if DO_HEATMAP_SECTION:
    all_frames = []
    script_lines = []

    # ### DIFFERENCE BETWEEN TOPSPIN AND CHIP PLACEMENT
    # topsScn_not_norm = vs.kde_heatmap(tops_retShots_rectified, 
    #                                     position_func = 'get_first_bounce_position', 
    #                                     normalize_density = False,
    #                                     zero_color = TOPS_COLOR, one_color = TOPS_COLOR, 
    #                                     bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

    # chipScn_not_norm = vs.kde_heatmap(chip_retShots_rectified, 
    #                                     position_func = 'get_first_bounce_position', 
    #                                     weight_func_or_arr = topspin_preds,
    #                                     normalize_density = False,
    #                                     zero_color = CHIP_COLOR, one_color = CHIP_COLOR,
    #                                     bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

    # tops_kde = topsScn_not_norm.alphaArrs[0]
    # chip_kde = chipScn_not_norm.alphaArrs[0]

    # diff_kde = tops_kde - chip_kde

    # rastArr = topsScn_not_norm.rastArrs[0]
    # bgrArr = np.where(diff_kde > 0, TOPS_COLOR, CHIP_COLOR)

    # diff_kde_plus, diff_kde_minus = diff_kde.clip(min = 0), -diff_kde.clip(max = 0)

    # diff_alpha_plus = diff_kde_plus / diff_kde_plus.max()
    # diff_alpha_minus = diff_kde_minus / diff_kde_minus.max()

    # alphaArr = np.where(diff_kde > 0, diff_alpha_plus, diff_alpha_minus)

    # diffScn = DummyScene([rastArr], [bgrArr], [alphaArr])

    # script_line =   """
    #                     Here we see a difference heatmap that highlights the areas 
    #                     in which choice chip and topspin returns are more prevalent in their respective colors.

    #                     Chipped returns are overrepresented at the corners of the court and the shallower areas along the center service line,
    #                     whereas topspin returns are relatively most frequent in the center of the court.
    #                 """
    # myFrms, animDict = vs.animate_static(diffScn, background_scene = courtScn, static_cam = heatCam, **prmDict, annotated_points = [fhPt, bhPt])
    # myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    # myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
    # # myFrms[-1].show() ### BEAUTIFUL!
    # all_frames.append(myFrms)
    # script_lines.append(script_line)

    # script_line =   """
    #                     Let us now look at a different facet of return characteristics: launch angle.
    #                 """
    # staticFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
    # all_frames.append(staticFrms)
    # script_lines.append(script_line)

    # ### Saving the separate MultiFrame objects and the Joint Script
    # assert len(script_lines) == len(all_frames)

    # # show(all_frames)

    # # b) Save segments
    # for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
    #     subFrms, script_line = frames_and_script
    #     running_idx = current_idx + idx
    #     subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
    #     with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
    #         f.write(script_line)
    
    # current_idx = current_idx + len(all_frames)


#########################
#### LAUNCH ANGLE #######
#########################

thickness = 0.25

# a useful null line for interpolation
from ATMANpy.types.Shapes3D import Line3D
nullLn = Line3D(bPt, bPt, color = BALL_COLOR, thickness = thickness)
nullScn = Scene([nullLn])

# player shape
_, player = myShots[1].feat_top_player().first()

# get topspin and chip dists, with axes
kwargDict_chip = {  'bandwidth': 2.4,
                    'resolution': 1000,
                    'axis_length': 45,
                    'axis_origin_point': bPt,
                    'x_unit_vector': Point3D((0, -1, 0)), 
                    'y_unit_vector': Point3D((0, 0, 1)),
                    'normalize_density': False, 
                    'stretch_density': 7}
kwargDict_tops = {  'bandwidth': 2.4,
                    'resolution': 1000,
                    'axis_length': 45,
                    'axis_origin_point': bPt,
                    'x_unit_vector': Point3D((0, -1, 0)), 
                    'y_unit_vector': Point3D((0, 0, 1)),
                    'normalize_density': False, 
                    'stretch_density': 6.5}


weightArr = regFrm_in.loc[regFrm_in.is_chipped == 1, 'weight_topspin'].values
weightArr_preds = 1 - regFrm_in.loc[regFrm_in.is_chipped == 1, 'pred_is_chipped'].values

# tops_retShots_rectified_filt = Shots([shot for shot in tops_retShots_rectified if 0 < shot.get_launch_angle() <= 75])
# chip_retShots_rectified_filt = Shots([shot for shot in chip_retShots_rectified if 0 < shot.get_launch_angle() <= 75])

tops_angleScn, tops_modes = vs.kde_angle_distribution(tops_retShots_rectified, **kwargDict_tops, color = TOPS_COLOR)
chip_angleScn, chip_modes = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict_chip, weight_func_or_arr = weightArr_preds, color = CHIP_COLOR)

### make axes
xLn, yLn = tops_angleScn.shapes[1], tops_angleScn.shapes[2]

x0, x1 = xLn.leftPt, xLn.rightPt.front(28) # lengthen xLn 
y0, y1 = yLn.leftPt, yLn.rightPt

from ATMANpy.types.ShapesTennis3D import Arrow
xArr = Arrow.make(x0, x1, hat_length = 3, color = const.BGR['bright_grey'], rot_angle = np.pi / 2, thickness = thickness)
yArr = Arrow.make(y0, y1, hat_length = 3, color = const.BGR['bright_grey'], rot_angle = np.pi / 2, thickness = thickness)

xArr_null = Arrow.make(x0, x0.front(0.5), hat_length = 0.1, color = const.BGR['bright_grey'], rot_angle = np.pi / 2, thickness = thickness)
yArr_null = Arrow.make(y0, y0.up(0.5), hat_length = 0.1, color = const.BGR['bright_grey'], rot_angle = np.pi / 2, thickness = thickness)

axScn = Scene([xArr, yArr])
axScn_null = Scene([xArr_null, yArr_null])
court_and_axScn = courtScn.combine(axScn)

axPnts = [xLn.center_point().down(2).left(2).front(15).annotate('Horizontal Speed'), 
          yLn.center_point().back(2).annotate('Vertical Speed')]
# axPnts = [] ### not needed because confusing when density is plotted ?!

### drop axes from scene -> we will render them in the background, with court_and_axScn
tops_angleScn = tops_angleScn.drop(2).drop(1)
chip_angleScn = chip_angleScn.drop(2).drop(1)

##### TEST IT ######
kdeScn = tops_angleScn.combine(chip_angleScn)
kdeView = View.make(sideretCam, kdeScn)
backView = View.make(sideretCam, court_and_axScn)
# backView.combine(kdeView).snap().show()

##### TEST LITTLE RACKET ######
from ATMANpy.types.Shapes3D import Ellipse3D
myEll = Ellipse3D.from_point3D(x0.back(2).down(2), radius_long = 10, radius_short = 5, fixed_dimension = 'y', rot_x = np.pi / 7, rot_y = 0)
ellView = View.make(sideretCam, Scene([myEll]))
# backView.combine(kdeView).combine(ellView).snap().show()

### get modes
# a) chip
chip_mode_lo, chip_mode_hi = 11.9, 18.1
mode_angles = [chip_mode_lo, chip_mode_hi]
modeLns = vis_ut.make_mode_lines(mode_angles, line_len = kwargDict_chip['axis_length'] * 2.1, 
                                 x_ax = kwargDict_chip['x_unit_vector'], y_ax = kwargDict_chip['y_unit_vector'],
                                 orig = kwargDict_chip['axis_origin_point'])
modeScn_lo = Scene([modeLns[0].recolor(CHIP_LO_COLOR).resize(thickness)]).dim(1.2)
modeScn_hi = Scene([modeLns[1].recolor(CHIP_HI_COLOR).resize(thickness)]).dim(1.2)

itpl_lns_lo = vs.interpolate_scenes(nullScn, modeScn_lo, num_frames = 25)
itpl_lns_hi = vs.interpolate_scenes(nullScn, modeScn_hi, num_frames = 25)

# b) topspin
tops_mode = 7.5 #tops_modes[1]
mode_angles = [tops_mode]
modeLns = vis_ut.make_mode_lines(mode_angles, line_len = kwargDict_tops['axis_length'] * 2.1,
                                 x_ax = kwargDict_tops['x_unit_vector'], y_ax = kwargDict_tops['y_unit_vector'],
                                 orig = kwargDict_tops['axis_origin_point'])
modeScn_tp = Scene([modeLns[0].recolor(TOPS_COLOR).resize(thickness)]).dim(1.2)

itpl_lns_tp = vs.interpolate_scenes(nullScn, modeScn_tp, num_frames = 25)

### check lines
lineView = View.make(sideretCam, modeScn_lo.combine(modeScn_hi).combine(modeScn_tp))

# backView.combine(kdeView).snap().show()
# lineView.combine(backView).combine(kdeView).snap().show()


# c) example line
mode_angles = [11.9]
exLns = vis_ut.make_mode_lines(mode_angles, line_len = kwargDict_chip['axis_length'] * 2.1, 
                                 x_ax = kwargDict_chip['x_unit_vector'], y_ax = kwargDict_chip['y_unit_vector'],
                                 orig = kwargDict_chip['axis_origin_point'])
modeScn_ex = Scene([exLns[0].recolor(BALL_COLOR).resize(thickness)])

itpl_lns_ex = vs.interpolate_scenes(nullScn, modeScn_ex, num_frames = 25)

### build a distribution of launch angles on choice chips.
sorted_chip_retShots_rectified = chip_retShots_rectified.sort(weightArr_preds, descending = True)

if DO_LAUNCH_ANGLE_INTRO:
    all_frames = []
    script_lines = []
    
    script_line =   """
                        The example return that we saw just now is chipped back at an angle of 11.9 degrees. 
                    """

    splnScn_thin_short = scene_to_graph_scene(splnScn, cutoff_frames = 5)

    prmDict_angle['adaptive_resolution_exponent'] = 0

    itplScns = vs.interpolate_scenes(axScn_null, axScn, TRANSITION_FRAMES)
    axFrms, animDict = vs.animate_sequence(itplScns, start_cam = sideretCam, stop_cam = sideretCam, 
                                           background_scene = courtScn, previous_foreground_scene = splnScn_thin_short,
                                           **prmDict_angle)
    # DecapStream(axFrms, 0, 100).show()

    statFrms, animDict = vs.animate_static(splnScn_thin_short, static_cam = sideretCam, num_frames = 25,
                                           background_scene = court_and_axScn, annotated_points = axPnts, 
                                           **prmDict_angle)
    # statFrms[-1].show()

    ang = myShots[1].get_launch_angle()
    limit_theta = ((180 - ang) * const.ANGLE_TO_RAD, 180 * const.ANGLE_TO_RAD)
    degCrc = Circle3D.from_point3D(bPt, fixed_dimension = 'x', radius = 25, limit_theta = limit_theta)

    degFrms0, animDict = vs.animate_static(splnScn_thin_short.put(degCrc), static_cam = sideretCam, num_frames = 25,
                                           background_scene = court_and_axScn, annotated_points = axPnts, 
                                           **prmDict_angle)
    # degFrms0[-1].show()

    anglePt = bPt.front(24).up(1).annotate(f'{ang:.1f} Degrees')

    degFrms1, animDict = vs.animate_static(splnScn_thin_short.put(degCrc), static_cam = sideretCam, num_frames = 25,
                                           background_scene = court_and_axScn, annotated_points = [anglePt, *axPnts], 
                                           **prmDict_angle)

    jointFrms = BGRFrames.from_frames([axFrms, statFrms, degFrms0, degFrms1])
    # DecapStream(jointFrms, 0, 100).show()
    # jointFrms[-1].show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    # show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)
    
    current_idx = current_idx + len(all_frames)


if DO_LAUNCH_ANGLE_DISTRO:
    all_frames = []
    script_lines = []

    stretch_default = kwargDict_chip['stretch_density']
    stretch_min = 2
    lmbd_exp = 0.5

    acceleration_rate = 0.2
    initial_frames = 20
    max_simul_lines = 5
    skipped_before = False
    how_many_shots = 1
    prev_kde = None
    shotList = []
    simul_shots = []
    weightList = []
    frms = []
    idx = 0
    for shot, weight in zip(sorted_chip_retShots_rectified[:], weightArr_preds[:]):
        if not 0 < shot.get_launch_angle() <= 75:
            continue
        
        ### build new kde
        shotList.append(shot)
        weightList.append(weight)
        simul_shots.append(shot)

        ### calculate num_frames
        float_frames = initial_frames * (1 - acceleration_rate) ** idx
        print(float_frames)
        num_frames = max(1, int(float_frames))

        if skipped_before:
            if how_many_shots == 1:
                skipped_before = False
                how_many_shots = max(1, min(np.log10(1 / float_frames), 5))
                # how_many_shots = min(1 / float_frames, 5)
            else:
                how_many_shots -= 1
                skipped_before = True
                idx += 1
                continue
        else:
            how_many_shots = max(1, min(int(np.log10(1 / float_frames)), 5))
            if how_many_shots > 1:
                how_many_shots -= 1
                skipped_before = True
                idx += 1
                continue

        ### stretch_density adjustment
        # -> when few shots are in the list, stretch down, when many, stretch up
        kwargDict_new_stretch = {**kwargDict_chip}

        lmbd = idx / len(sorted_chip_retShots_rectified) # 201
        lmbd_star = lmbd ** lmbd_exp
        stretch_density = lmbd_star * stretch_default + (1 - lmbd_star) * stretch_min
        kwargDict_new_stretch['stretch_density'] = stretch_density

        kde1, _ = vs.kde_angle_distribution(Shots(shotList), **kwargDict_new_stretch,
                                            weight_func_or_arr = np.array(weightList), color = CHIP_COLOR)
        kde = Scene([kde1[0]])

        ### if necessary: create previous KDE
        if prev_kde is None:
            els = len(kde[0].coords[0])
            x, y, z = bPt.coords
            prev_pt = Point3D((np.full(els, fill_value = x), np.full(els, fill_value = y), np.full(els, fill_value = z)), color = CHIP_COLOR)
            prev_kde = Scene([prev_pt])
        
        ### Animate the current trajectory
        splnScn_simul = Scene.make_empty()
        for shot in simul_shots:
            anchTraj = shot.shotTraj.reanchor(bPt)
            _, splines_current = anchTraj.feat_spline_cumul().last()
            splnScn_curr = Scene([spLn.recolor(BALL_COLOR) for spLn in splines_current if spLn is not None])

            splnScn_curr = scene_to_graph_scene(splnScn_curr, cutoff_frames = 5)
            splnScn_simul = splnScn_simul.combine(splnScn_curr)
        
        myFrms, _ = vs.animate_static(splnScn_simul, background_scene = court_and_axScn, previous_foreground_scene = prev_kde, **prmDict_angle,
                                    static_cam = sideretCam, num_frames = num_frames, annotated_points = axPnts)
        frms.append(myFrms)

        ### Animate the new KDE
        if num_frames > 1:
            itpl_kdes = vs.interpolate_scenes(prev_kde, kde, num_frames)
        else:
            itpl_kdes = Scene([kde])
        myFrms, _ = vs.animate_sequence(itpl_kdes, background_scene = court_and_axScn, previous_foreground_scene = splnScn_simul, **prmDict_angle,
                                        start_cam = sideretCam, stop_cam = sideretCam, cumulative_foreground_transparence = 1,
                                        annotated_points = axPnts)
        frms.append(myFrms)

        simul_shots = []
        prev_kde = kde
        idx += 1

    ### final index pass
    
    ### stretch_density adjustment
    # -> when few shots are in the list, stretch down, when many, stretch up
    kwargDict_new_stretch = {**kwargDict_chip}

    lmbd = idx / 201 #len(sorted_chip_retShots_rectified)
    lmbd_star = lmbd ** lmbd_exp
    stretch_density = lmbd_star * stretch_default + (1 - lmbd_star) * stretch_min
    kwargDict_new_stretch['stretch_density'] = stretch_density

    kde1, _ = vs.kde_angle_distribution(Shots(shotList), **kwargDict_new_stretch,
                                        weight_func_or_arr = np.array(weightList), color = CHIP_COLOR)
    kde = Scene([kde1[0]])

    ### if necessary: create previous KDE
    if prev_kde is None:
        els = len(kde[0].coords[0])
        x, y, z = bPt.coords
        prev_pt = Point3D((np.full(els, fill_value = x), np.full(els, fill_value = y), np.full(els, fill_value = z)), color = CHIP_COLOR)
        prev_kde = Scene([prev_pt])
    
    ### Animate the current trajectory
    splnScn_simul = Scene.make_empty()
    for shot in simul_shots:
        anchTraj = shot.shotTraj.reanchor(bPt)
        _, splines_current = anchTraj.feat_spline_cumul().last()
        splnScn_curr = Scene([spLn.recolor(BALL_COLOR) for spLn in splines_current if spLn is not None])

        splnScn_curr = scene_to_graph_scene(splnScn_curr, cutoff_frames = 5)
        splnScn_simul = splnScn_simul.combine(splnScn_curr)
    
    myFrms, _ = vs.animate_static(splnScn_simul, background_scene = court_and_axScn, previous_foreground_scene = prev_kde, **prmDict_angle,
                                static_cam = sideretCam, num_frames = num_frames, annotated_points = axPnts)
    frms.append(myFrms)

    ### Animate the new KDE
    if num_frames > 1:
        itpl_kdes = vs.interpolate_scenes(prev_kde, kde, num_frames)
    else:
        itpl_kdes = Scene([kde])
    myFrms, _ = vs.animate_sequence(itpl_kdes, background_scene = court_and_axScn, previous_foreground_scene = splnScn_simul, **prmDict_angle,
                                    start_cam = sideretCam, stop_cam = sideretCam, cumulative_foreground_transparence = 1,
                                    annotated_points = axPnts)
    frms.append(myFrms)

    simul_shots = []
    prev_kde = kde
    idx += 1

    ################
    jntFrms = BGRFrames.from_frames(frms)
    jntFrms = overlay(jntFrms, legFrm_chip, top_left_point = anchorPt_middle)
    # DecapStream(jntFrms, 0, 100).show()

    script_line = "Let's see how launch angles are distributed for choice chipped returns in general."
    itplFrms = vs.interpolate_frames(degFrms1[-1], jntFrms[0], 50)
    # DecapStream(itplFrms, 0, 100).show()
    
    ## fade out axes -> not needed! we have distro now
    statFrms, _ = vs.animate_static(kde, background_scene = court_and_axScn, previous_foreground_scene = splnScn_simul, **prmDict_angle,
                                    static_cam = sideretCam)
    statFrms = overlay(statFrms, legFrm_chip, top_left_point = anchorPt_middle)
    fadeFrms = vs.interpolate_frames(jntFrms[-1], statFrms[0], 50)

    jntFrms_all = BGRFrames.from_frames([itplFrms, jntFrms, fadeFrms])
    # DecapStream(jntFrms_all, 0, 100).show()

    all_frames.append(jntFrms_all)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    # show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)
    
    current_idx = current_idx + len(all_frames)


if DO_LAUNCH_ANGLE_MODES:
    ### Show entire distribution
    script_lines = []
    all_frames = []

    # a) Chips
    #    --> Two modes: low angle chip; most prevalent, higher angle chip, less prevalent but pronounced.
    script_line =   """
                        Within choice chips, we observe two modes at two different launch angles.
                        It does seem to be the case empirically that when a player opts for a chipped return, 
                        he really opts for either a high-arcing chip, or, most commonly, for a low, shallow-bouncing chip.
                    """
    kdeFrms, animDict = vs.animate_static(kde, background_scene = court_and_axScn, **prmDict_angle,
                                    static_cam = sideretCam, num_frames = STATIC_FRAMES)
    kdeFrms = overlay(kdeFrms, legFrm_chip, top_left_point = anchorPt_middle)
    itplFrms = vs.interpolate_frames(jntFrms_all[-1], kdeFrms[0], 25)
    # DecapStream(itplFrms, 0, 100).show()

    lineFrms0, animDict = vs.animate_sequence(itpl_lns_hi, background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    lineFrms0 = overlay(lineFrms0, leg1Frm, top_left_point = anchorPt_middle)

    lineFrms1, animDict = vs.animate_sequence(itpl_lns_lo, background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    lineFrms1 = overlay(lineFrms1, leg2Frm, top_left_point = anchorPt_middle)

    jointFrms = BGRFrames.from_frames([itplFrms, lineFrms0, lineFrms1])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    script_line =   """
                        On the other hand, for topspin returns, the distribution is more uniform: we observe only a single mode.
                    """
    
    # script_line =   """
    #                     The topspin mode angle is lower than both chip return modes, 
    #                     which is why we differentiate between a slice and a chip: 
    #                     a typical rally slice would depart at a launch angle lower than, or similar to, a typical topspin shot.
    #                 """
    statFrms0 = BGRFrames.from_frame(lineFrms1[-1], 25)

    # b) Topspin  
    #  --> One Mode: low angle topspin
    animDict['previous_foreground_scene'] = animDict['previous_foreground_scene'].drop(0)
    itpl_angleScns = vs.interpolate_scenes(kde, tops_angleScn, 25)
    itplFrms, animDict = vs.animate_sequence(itpl_angleScns, background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    itplFrms = overlay(itplFrms, leg3Frm, top_left_point = anchorPt_middle)
    # DecapStream(itplFrms, 0, 100).show()

    statFrms1, animDict = vs.animate_static(tops_angleScn, background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    static_cam = sideretCam, num_frames = 25)
    statFrms1 = overlay(statFrms1, leg3Frm, top_left_point = anchorPt_middle)

    lineFrms2, animDict = vs.animate_sequence(itpl_lns_tp, background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    lineFrms2 = overlay(lineFrms2, leg3Frm, top_left_point = anchorPt_middle)

    statFrms2 = BGRFrames.from_frame(lineFrms2[-1], 25)
    statFrms2 = overlay(statFrms2, leg3Frm, top_left_point = anchorPt_middle)

    jointFrms = BGRFrames.from_frames([statFrms0, itplFrms, statFrms1, lineFrms2, statFrms2])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    # c) show angles jointly
    limit_theta = ((180 - tops_mode) * const.ANGLE_TO_RAD, 180 * const.ANGLE_TO_RAD)
    degCrc0 = Circle3D.from_point3D(bPt, fixed_dimension = 'x', radius = 25, limit_theta = limit_theta)
    limit_theta = ((180 - chip_mode_lo) * const.ANGLE_TO_RAD, 180 * const.ANGLE_TO_RAD)
    degCrc1 = Circle3D.from_point3D(bPt, fixed_dimension = 'x', radius = 40, limit_theta = limit_theta)
    limit_theta = ((180 - chip_mode_hi) * const.ANGLE_TO_RAD, 180 * const.ANGLE_TO_RAD)
    degCrc2 = Circle3D.from_point3D(bPt, fixed_dimension = 'x', radius = 55, limit_theta = limit_theta)

    angPt0 = bPt.front(22).up(0.5).annotate(f'{tops_mode:.1f} Degrees')
    angPt1 = bPt.front(38).up(1.5).annotate(f'{chip_mode_lo:.1f} Degrees')
    angPt2 = bPt.front(53).up(2.5).annotate(f'{chip_mode_hi:.1f} Degrees')

    script_line = f"To put numbers to it: the typical topspin return is launched at an angle of {tops_mode:.1f} Degrees..."
    animDict['previous_foreground_scene'] = animDict['previous_foreground_scene'].drop(2)

    myFrms, animDict = vs.animate_static(Scene([degCrc0]), background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    static_cam = sideretCam, num_frames = 25, annotated_points = [angPt0])
    myFrms = overlay(myFrms, leg3Frm, top_left_point = anchorPt_middle)

    fadeFrms = vs.interpolate_frames(jointFrms[-1], myFrms[0], 50)

    all_frames.append(fadeFrms)
    script_lines.append(script_line)

    script_line = f"... the low-angle choice chip return at {chip_mode_lo:.1f} Degrees..."
    myFrms, animDict = vs.animate_static(Scene([degCrc0, degCrc1]), background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    static_cam = sideretCam, num_frames = 25, annotated_points = [angPt0, angPt1])
    myFrms = overlay(myFrms, leg3Frm, top_left_point = anchorPt_middle)

    fadeFrms = vs.interpolate_frames(fadeFrms[-1], myFrms[0], 50)
    
    all_frames.append(fadeFrms)
    script_lines.append(script_line)

    script_line = f"... and the high_angle choice chip return at {chip_mode_hi:.1f} Degrees."
    myFrms, animDict = vs.animate_static(Scene([degCrc0, degCrc1, degCrc2]), background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    static_cam = sideretCam, num_frames = 25, annotated_points = [angPt0, angPt1, angPt2])
    myFrms = overlay(myFrms, leg3Frm, top_left_point = anchorPt_middle)

    fadeFrms = vs.interpolate_frames(fadeFrms[-1], myFrms[0], 50)

    all_frames.append(fadeFrms)
    script_lines.append(script_line)

    ang = myShots[1].get_launch_angle()
    script_line =   f"""
                        Our example chip return left the racket at {ang:.1f} degrees, making it a good representative of the low angle mode. 
                    """
    lnFrms, animDict = vs.animate_sequence(itpl_lns_ex, background_scene = court_and_axScn, **prmDict_angle, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam, annotated_points = [angPt0, angPt1, angPt2])
    lnFrms = overlay(lnFrms, leg3Frm, top_left_point = anchorPt_middle)
    # DecapStream(lnFrms, 0, 100).show()

    # myFrms, animDict = vs.animate_static(Scene([degCrc0, degCrc1, degCrc2]), background_scene = court_and_axScn, **prmDict_angle, **animDict,
    #                                 static_cam = sideretCam, num_frames = 25, annotated_points = [angPt0, angPt1, angPt2])
    # myFrms = overlay(myFrms, leg3Frm, top_left_point = anchorPt_middle)

    all_frames.append(lnFrms)
    script_lines.append(script_line)

    script_line =   "Let's watch it again, in this light."

    scale_factor = 1.5
    brestCam_resized_half = brestCam.resize(scale_factor)

    animDict['previous_foreground_scene'] = animDict['previous_foreground_scene'].combine(axScn)

    moveFrms_with_text, _ = vs.animate_cam_motion(None, background_scene = courtScn, **prmDict_nolum, **animDict,
                                    start_cam = sideretCam, stop_cam = brestCam_resized_half,
                                    num_frames = 100, cumulative_foreground_transparence = 0.95)
    
    animDict['previous_annotated_points'] = []

    moveFrms_without_text, _ = vs.animate_cam_motion(None, background_scene = courtScn, **prmDict_nolum, **animDict,
                                    start_cam = sideretCam, stop_cam = brestCam_resized_half,
                                    num_frames = 100, cumulative_foreground_transparence = 0.95)
    
    moveFrms = moveFrms_with_text.dim_progressively(1, 0, exp = 0.15) + moveFrms_without_text.dim_progressively(0, 1, exp = 0.15)
    # DecapStream(moveFrms, 0, 100).show()

    all_frames.append(moveFrms)
    script_lines.append(script_line)

    print(ang)
    print(chip_mode_lo) ### original shot very close to lower mode!

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    # show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)

if DO_LAUNCH_ANGLE_EXAMPLE:
    all_frames = []
    script_lines = []

    ### find similar shot of original, with higher launch angle V2 -> trajectory metrics
    devArr_ang = chip_retShots_rectified.get_launch_angle() - chip_mode_hi
    devArr_spd = chip_retShots_rectified.get_initial_speed() - myShots[1].get_initial_speed()
    devArr_hgt = chip_retShots_rectified.get_ball_hit_height() - myShots[1].get_ball_hit_height()
    devArr_spn = chip_retShots_rectified.get_initial_spin() - myShots[1].get_initial_spin()

    std_ang = (devArr_ang ** 2).mean() ** 0.5
    std_spd = (devArr_spd ** 2).mean() ** 0.5
    std_hgt = (devArr_hgt ** 2).mean() ** 0.5
    std_spn = (devArr_spn ** 2).mean() ** 0.5

    dev_metric = np.abs(devArr_ang / std_ang) + np.abs(devArr_spd / std_spd) + np.abs(devArr_hgt / std_hgt) + np.abs(devArr_spn / std_spn)
    dev_metric = dev_metric / 4 ### average standard deviations away from original shot

    small_to_large = np.argsort(dev_metric)
    best_idx = int(small_to_large[3]) #int(np.argmin(dev_metric)) ### not necessarily the smallest since we want a top to bot point!
    dev_metric[best_idx]

    bestShot = chip_retShots_rectified[best_idx]
    mtc_idx, pnt_idx = bestShot.get_match_index(), bestShot.get_point_index()
    bestPnt = retPnts.get(mtc_idx, pnt_idx)

    print('Match, Point Indices:')
    print(f'Original Point: {myShots[1].get_match_index(), myShots[1].get_point_index()}')
    print(f'Best Matching Point: {mtc_idx, pnt_idx}')

    bestShots = bestPnt[:2] ### flip so that we get the original trajectory back
    # bestShots.stream(confs = confs).rescale(3).show()

    # prep scaled camera view and frames
    scale_factor = 1.5
    brestCam_resized_half = brestCam.resize(scale_factor)

    text_size = 3
    text_color = const.BGR['bright_grey']
    text_where = (390, 520)
    text_where = (30, 50)
    text_thickness = 2

    fade_len = 50
    dim_factor = 0.5

    origStm = myShots.stream(confs = confs).join().maybe_to_stream()
    origDtm = origStm.decap()
    orig_scaledFrms = myDtm.frames.rescale(scale_factor)

    orig_scaledFrms_dim = orig_scaledFrms.dim_progressively(1, dim_factor)
    orig_scaledFrms = orig_scaledFrms.put('Low Chip', size = text_size, color = text_color, where = text_where, thickness = text_thickness)

    bestStm = bestShots.stream(confs = confs).join().maybe_to_stream()
    bestDtm = bestStm.decap()
    best_scaledFrms = bestDtm.frames.rescale(scale_factor)
    
    best_scaledFrms_dim = best_scaledFrms.dim_progressively(1, dim_factor)
    best_scaledFrms = best_scaledFrms.put('High Chip', size = text_size, color = text_color, where = text_where, thickness = text_thickness)

    early_origFrms = origStm.widen(fade_len).decap().frames[:fade_len]
    early_orig_scaledFrms = early_origFrms.rescale(scale_factor)
    early_orig_scaledFrms = early_orig_scaledFrms.put('Low Chip', size = text_size, color = text_color, where = text_where, thickness = text_thickness)

    early_bestFrms = bestStm.widen(fade_len).decap().frames[:fade_len]
    early_best_scaledFrms = early_bestFrms.rescale(scale_factor)
    early_best_scaledFrms = early_best_scaledFrms.put('High Chip', size = text_size, color = text_color, where = text_where, thickness = text_thickness)

    ### experiment: fade-in with movement
    lsFrm_resc = View.make(brestCam_resized_half, courtScn).snap()
    blankFrm_resc = View.make(brestCam_resized_half, Scene([])).snap()

    def mean_two_frames(aFrm, bFrm, my_lambda):
        aArr, bArr = aFrm.array.astype(float), bFrm.array.astype(float)
        itplArr = aArr * (1 - my_lambda) + bArr * my_lambda
        return itplArr.astype(np.uint8)

    orig_itplArrs = np.stack([mean_two_frames(lsFrm_resc, f, idx / fade_len) for idx, f in enumerate(early_orig_scaledFrms)], axis = 0)
    orig_itplFrms = BGRFrames(orig_itplArrs)

    best_itplArrs = np.stack([mean_two_frames(blankFrm_resc, f, idx / fade_len) for idx, f in enumerate(early_best_scaledFrms)], axis = 0)
    best_itplFrms = BGRFrames(best_itplArrs)

    # create spline animation for our original point on new bestCam_resized_half
    splineFrms_orig, _ = vs.animate_sequence(spline_scenes, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                        cumulate_argument = False, animate_over = orig_scaledFrms, **prmDict)
    # DecapStream(splineFrms_orig, 0, 100).show()

    splineFrms_orig_dim, _ = vs.animate_sequence(spline_scenes, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                        cumulate_argument = False, animate_over = orig_scaledFrms_dim, **prmDict)
    # DecapStream(splineFrms_orig_dim, 0, 100).show()

    # create spline animation for our best point on new brestCam_resized_half
    serveSF = bestShots[0].feat_spline_cumul(which_endpoint = 'penultimate')
    returnSF = bestShots[1].feat_spline_cumul(which_endpoint = 'penultimate')

    serve_scenes = [Scene([spLn.recolor(BALL_COLOR).resize(1.5) for spLn in splines if spLn is not None]) for splines in serveSF.values()]
    return_scenes = [Scene([spLn.recolor(BALL_COLOR).resize(1.5) for spLn in splines if spLn is not None]) for splines in returnSF.values()]
    best_spline_scenes = [*serve_scenes, *return_scenes]

    splineFrms_best, _ = vs.animate_sequence(best_spline_scenes, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                        cumulate_argument = False, animate_over = best_scaledFrms, **prmDict)
    # DecapStream(splineFrms_best, 0, 100).show(ms_between_frames = 100)
    
    splineFrms_best_dim, _ = vs.animate_sequence(best_spline_scenes, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                        cumulate_argument = False, animate_over = best_scaledFrms_dim, **prmDict)
    # DecapStream(splineFrms_best_dim, 0, 100).show()

    # rasterize frames, and fade in the point being played
    script_line = ""
    fadeinRst = FramesRaster(2, 2, borders = None)
    fadeinRst.add(orig_itplFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    fadeinFrms = fadeinRst.render()
    all_frames.append(fadeinFrms)
    script_lines.append(script_line)
    
    # play out point with spline
    script_line = ""
    myRst = FramesRaster(2, 2, borders = None)
    myRst.add(splineFrms_orig.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    playFrms_orig = myRst.render()
    all_frames.append(playFrms_orig)
    script_lines.append(script_line)

    # freeze, and play best matching point in lower right
    script_line =   """
                        We now compare the low chip to a representative of high-angle chip returns, in the lower right.
                    """
    staticFrms = BGRFrames.from_frame(playFrms_orig[-1], 25)

    # fade in best comparison in lower right
    fadeinRst = FramesRaster(2, 2, borders = None)
    fadeinRst.add(splineFrms_orig[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    fadeinRst.add(best_itplFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    fadeinFrms = fadeinRst.render()

    # play out point with spline
    playRst = FramesRaster(2, 2, borders = None)
    playRst.add(splineFrms_orig[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    playRst.add(splineFrms_best.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    playFrms = playRst.render()

    jointFrms = BGRFrames.from_frames([staticFrms, fadeinFrms, playFrms])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    # Matching Table
    lo_name = 'Low Chip'
    hi_name = 'High Chip'

    lo_ang = myShots[1].get_launch_angle()
    lo_spd = myShots[1].get_initial_speed()
    lo_spn = myShots[1].get_initial_spin()
    lo_hgt = myShots[1].get_ball_hit_height() / const.SCALE * const.FOOT_TO_METER * 0.8 ### the hit point for the low shot is one frame off -> correct!
    # lo_dst = myShots[1].get_flight_line().length() / const.SCALE * const.FOOT_TO_METER

    hi_ang = bestShots[1].get_launch_angle()
    hi_spd = bestShots[1].get_initial_speed()
    hi_spn = bestShots[1].get_initial_spin()
    hi_hgt = bestShots[1].get_ball_hit_height() / const.SCALE * const.FOOT_TO_METER
    # hi_dst = bestShots[1].get_flight_line().length() / const.SCALE * const.FOOT_TO_METER

    table_data = {  'Launch Angle': {   lo_name: f'{lo_ang:.1f} deg',   hi_name: f'{hi_ang:.1f} deg'},
                    'Launch Velocity': {lo_name: f'{lo_spd:.0f} km/h',  hi_name: f'{hi_spd:.0f} km/h'},
                    'Spin': {           lo_name: f'{lo_spn:.0f} rpm',   hi_name: f'{hi_spn:.0f} rpm'},
                    'Hit Height': {     lo_name: f'{lo_hgt:.1f} m',     hi_name: f'{hi_hgt:.1f} m'}} ### adjust hit height? obviously not correct for lo_name?
    
    dims = (splineFrms_orig.cols(), splineFrms_orig.rows())
    myTab = Table(table_data, dimensions = dims, skeleton_shrinkage = 0.2, skeleton_vertical_offset = 1, 
                  title = 'Initial Conditions', title_horizontal_offset = 11, title_vertical_offset = 0.5,
                  header_text_color = const.BGR['leafy_green'], title_color = const.BGR['bright_grey'],
                  index_offset = 0.25, 
                  header_text_size = 1, index_text_size = 1, data_text_size = 1, 
                  skip_after_how_many_symbols = None,
                  automatic_vertical_offset_text_line_height = 0.25)
    tabFrm_match = myTab.render()
    # tabFrm_match.show()

    script_line =   """
                        Both returns are hit at almost the same speed, spin and height.
                        We therefore isolate the effect of launch angle on ball flight, which differs as follows: 
                    """

    # script_line =   """
    #                     The return-of-comparison is chosen to be maximally similar to the low-angle chip in all respects except the launch angle,
    #                     so that we can cleanly see the effects of a higher launch angle on ball trajectory and bounce. 
    #                     As the table shows, both returns are made at almost the same speed, spin, and height.
    #                     We therefore isolate the effect of launch angle on trajectory characteristics, which we will look at now.
    #                 """
    myRst = FrameRaster(2, 2)
    myRst.add(splineFrms_orig[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(splineFrms_best[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_match, 0, 1)
    tabFrms_match = BGRFrames.from_frame(myRst.render(), STATIC_FRAMES)
    all_frames.append(tabFrms_match)
    script_lines.append(script_line)

    lo_apx = myShots[1].get_apex()[2] / const.SCALE * const.FOOT_TO_METER
    lo_apx_pst = myShots[1].get_apex_post_bounce()[2] / const.SCALE * const.FOOT_TO_METER
    lo_secs = myShots[1].get_num_frames_until_bounce() / 25
    lo_hori = myShots[1].get_post_bounce_horizontal_speed(how = 'corrected_next')
    
    hi_apx = bestShots[1].get_apex()[2] / const.SCALE * const.FOOT_TO_METER
    hi_apx_pst = bestShots[1].get_apex_post_bounce()[2] / const.SCALE * const.FOOT_TO_METER
    hi_secs = bestShots[1].get_num_frames_until_bounce() / 25
    hi_hori = bestShots[1].get_post_bounce_horizontal_speed(how = 'corrected_next')
    
    dims = (splineFrms_orig.cols(), splineFrms_orig.rows())


    table_data = {  'Apex':                     {lo_name: f'{lo_apx:.1f} m',        hi_name: f'{hi_apx:.1f} m'},
                    'Apex post-bounce':         {lo_name: f'{lo_apx_pst:.1f} m',    hi_name: f'{hi_apx_pst:.1f} m'},
                    'Ground Speed post-bounce': {lo_name: f'{lo_hori:.1f} km/h',    hi_name: f'{hi_hori:.1f} km/h'},
                    'Time until bounce':        {lo_name: f'{lo_secs:.2f} sec',     hi_name: f'{hi_secs:.2f} sec'},}
    
    table_data_0 = {'Apex':                     {lo_name: f'{lo_apx:.1f} m',        hi_name: f'{hi_apx:.1f} m'}}

    table_data_1 = {'Apex':                     {lo_name: f'{lo_apx:.1f} m',        hi_name: f'{hi_apx:.1f} m'},
                    'Apex post-bounce':         {lo_name: f'{lo_apx_pst:.1f} m',    hi_name: f'{hi_apx_pst:.1f} m'}
                    }
    table_data_2 = {'Apex':                     {lo_name: f'{lo_apx:.1f} m',        hi_name: f'{hi_apx:.1f} m'},
                    'Apex post-bounce':         {lo_name: f'{lo_apx_pst:.1f} m',    hi_name: f'{hi_apx_pst:.1f} m'},
                    'Ground Speed post-bounce': {lo_name: f'{lo_hori:.1f} km/h',    hi_name: f'{hi_hori:.1f} km/h'}}
    
    # myTab = Table(table_data, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1, 
    #               title = 'Trajectory Comparison', title_horizontal_offset = 10, header_text_color = const.BGR['bright_grey'],
    #               header_text_size = 1, index_text_size = 1, data_text_size = 1, 
    #               skip_after_how_many_symbols = None)
    # tabFrm_comp = myTab.render()

    # myTab_0 = Table(table_data_0, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1, 
    #               title = 'Trajectory Comparison', title_horizontal_offset = 10, header_text_color = const.BGR['bright_grey'],
    #               header_text_size = 1, index_text_size = 1, data_text_size = 1, null_rows = 3,
    #               skip_after_how_many_symbols = None)
    # tabFrm_comp_0 = myTab_0.render()

    # myTab_1 = Table(table_data_1, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1, 
    #               title = 'Trajectory Comparison', title_horizontal_offset = 10, header_text_color = const.BGR['bright_grey'],
    #               header_text_size = 1, index_text_size = 1, data_text_size = 1, null_rows = 2,
    #               skip_after_how_many_symbols = None)
    # tabFrm_comp_1 = myTab_1.render()
    
    # myTab_2 = Table(table_data_2, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1, 
    #               title = 'Trajectory Comparison', title_horizontal_offset = 10, header_text_color = const.BGR['bright_grey'],
    #               header_text_size = 1, index_text_size = 1, data_text_size = 1, null_rows = 1, ###?
    #               skip_after_how_many_symbols = None)
    # tabFrm_comp_2 = myTab_2.render()
    
    myTab = Table(table_data, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1,
                  title_vertical_offset = 0.5, skeleton_vertical_offset = 1, 
                  title = 'Trajectory Comparison', title_horizontal_offset = 10, 
                  header_text_color = const.BGR['leafy_green'], title_color = const.BGR['bright_grey'],
                  header_text_size = 1, index_text_size = 1, data_text_size = 1, 
                  skip_after_how_many_symbols = None, resolution = 2000,
                  automatic_vertical_offset_text_line_height = 0.25)
    myTab.set_back_color(const.BGR['darker_grey'], 2, 0)
    myTab.set_back_color(const.BGR['darker_grey'], 2, 1)
    myTab.set_back_color(const.BGR['darker_grey'], 2, 2)
    tabFrm_comp_0 = myTab.render()
    # tabFrm_comp_0.show()

    myTab = Table(table_data, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1, 
                  title_vertical_offset = 0.5, skeleton_vertical_offset = 1, 
                  title = 'Trajectory Comparison', title_horizontal_offset = 10, 
                  header_text_color = const.BGR['leafy_green'], title_color = const.BGR['bright_grey'],
                  header_text_size = 1, index_text_size = 1, data_text_size = 1, 
                  skip_after_how_many_symbols = None, resolution = 2000,
                  automatic_vertical_offset_text_line_height = 0.25)
    myTab.set_back_color(const.BGR['darker_grey'], 3, 0)
    myTab.set_back_color(const.BGR['darker_grey'], 3, 1)
    myTab.set_back_color(const.BGR['darker_grey'], 3, 2)
    tabFrm_comp_1 = myTab.render()
    # tabFrm_comp_1.show()

    myTab = Table(table_data, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1, 
                  title_vertical_offset = 0.5, skeleton_vertical_offset = 1, 
                  title = 'Trajectory Comparison', title_horizontal_offset = 10, 
                  header_text_color = const.BGR['leafy_green'], title_color = const.BGR['bright_grey'],
                  header_text_size = 1, index_text_size = 1, data_text_size = 1, 
                  skip_after_how_many_symbols = None, resolution = 2000,
                  automatic_vertical_offset_text_line_height = 0.25)
    myTab.set_back_color(const.BGR['darker_grey'], 4, 0)
    myTab.set_back_color(const.BGR['darker_grey'], 4, 1)
    myTab.set_back_color(const.BGR['darker_grey'], 4, 2)
    tabFrm_comp_2 = myTab.render()
    # tabFrm_comp_2.show()

    # tabFrms_blink_0 = vs.blinking(tabFrm_comp_1, tabFrm_comp_0, blinks_per_second = 1.5, duration_in_frames = 300)
    # tabFrms_blink_1 = vs.blinking(tabFrm_comp_2, tabFrm_comp_1, blinks_per_second = 1.5, duration_in_frames = 300)
    # tabFrms_blink_2 = vs.blinking(tabFrm_comp, tabFrm_comp_2, blinks_per_second = 1.5, duration_in_frames = 300)

    # DecapStream(tabFrms_blink_0, 0, 100).show()
    # DecapStream(tabFrms_blink_1, 0, 100).show()
    # DecapStream(tabFrms_blink_2, 0, 100).show()
    # tabFrm_comp.show()

    # script_line =   """
    #                     The high-arcing chip bounces 70 cm higher than the low chip,
    #                     If the arcing chip does not bounce close to the server's baseline, its high bounce can allow the server to attack with a flat shot at aggressive angles,
    #                     whereas a low-flying chip that bounces shallow in the court is not so exploitable: it stays low and is awkward to attack because it has to be played with aggressive topspin
    #                     or lower margins over the net to dip back down into the returner's half.
    #                 """
    script_line =   "The high-arcing chip bounces 70 cm higher than the low chip, which makes it easier to attack if it lacks sufficient depth."
    myRst = FramesRaster(2, 2)
    myRst.add(splineFrms_orig[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(splineFrms_best[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_match, 0, 1)
    myRst.add(tabFrm_comp_0, 1, 0)
    tabFrms_match = BGRFrames.from_frame(myRst.render()[0], 25)
    all_frames.append(tabFrms_match)
    script_lines.append(script_line)

    # script_line =   """
    #                     On the other hand, the high chip loses more of its speed along the ground, which means that the server has to generate all the power if he wants to attack;
    #                     in this sense, a high-arcing chip can neutralize a serve if hit with sufficient depth.
    #                 """
    script_line =   "On the other hand, the high chip loses more of its speed; it is up to the server to inject pace."
    myRst = FramesRaster(2, 2)
    myRst.add(splineFrms_orig[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(splineFrms_best[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_match, 0, 1)
    myRst.add(tabFrm_comp_1, 1, 0)
    tabFrms_match = BGRFrames.from_frame(myRst.render()[0], 25)
    all_frames.append(tabFrms_match)
    script_lines.append(script_line)
    
    script_line =   """
                        The arcing chip also gives the returner more time to recuperate, which may be especially useful if the serve is hit out wide and carries the returner off the court.
                        With respect to this latter point, let's visualize the impact of a ball's time in flight on the receiving player's ability to reposition on the court.
                    """
    myRst = FramesRaster(2, 2)
    myRst.add(splineFrms_orig[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(splineFrms_best[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_match, 0, 1)
    myRst.add(tabFrm_comp_2, 1, 0)
    tabFrms_match = BGRFrames.from_frame(myRst.render()[0], 25)
    all_frames.append(tabFrms_match)
    script_lines.append(script_line)

    ### To say about comparison: The high chip...
        # loses more of its speed along the ground -> server has to generate all the power if he wants to attack
        # gives returner more time to recuperate, especially if the serve is hit out wide and carries the receiver off the court, this is useful.
        # However, if it does not bounce close to the server's baseline, its high bounce can allow the server to attack with a flat shot at aggressive angles.
        # On the other hand, a low chip that lands shallow is not terrible: it stays low and is awkward to attack because it has to be played with aggressive topspin
        # or lower margins over the net to dip back down into the returner's half in time for it not to sail long.

    ### with respect to time to recuperate: let's take a look at the impact of more time on ground covered.
    player_speed_kph = 15 ### this is taken from https://tennismash.com/2016/11/24/gig-djokovic-fastest-tennis-player-world/#:~:text=As%20many%20might%20expect%2C%20Novak,top%20speed%20of%2014.89kph.
    player_speed_mps = 15 * const.KPH_TO_MPS
    seconds_needed_to_recover = 1/3 ### can't immediately start running after contact
    
    myPnt = dataPnts[0]

    repPts = dataPnts.get_returns().get_optimal_repositioning_position(as_array = True)
    retPts = dataPnts.get_returns().get_player_hit_position()

    px_dist = ((repPts - retPts) ** 2).sum(axis = 1) ** 0.5
    met_dist = px_dist / const.SCALE * const.FOOT_TO_METER
    
    path_dist = 20
    min_dist = 8

    dist_filter = (min_dist < met_dist) & (met_dist < path_dist)
    dist_filter.sum() ### for these points, the player has to sprint to get in position

    def get_ret_move_speed(myPnt, cutoff = 25, ramp_up = 8, min_dist = 8, max_dist = 20):
        myRet = myPnt.get_return()
        if myRet is None:
            return np.nan
        
        diffArr = myRet.get_optimal_repositioning_position(as_array = True) - myRet.get_player_hit_position()
        if np.isnan(diffArr.all()):
            return np.nan

        px_dist = (diffArr ** 2).sum() ** 0.5 
        met_dist = px_dist / const.SCALE * const.FOOT_TO_METER

        if not min_dist < met_dist < max_dist:
            return np.nan

        myPM = myRet.hitPlayer
        xSer, ySer = myPM['x'].iloc[:-1], myPM['y'].iloc[:-1]
        xSer_use, ySer_use = xSer.loc[ramp_up:cutoff], ySer.loc[ramp_up:cutoff]
        step_speed = (xSer_use.diff() ** 2 + ySer_use.diff() ** 2) ** 0.5
        step_speed = step_speed.dropna()
        pix_per_frame_speed = step_speed.sum() / len(step_speed)
        met_per_second_speed = pix_per_frame_speed / const.SCALE * const.FOOT_TO_METER * 25
        km_per_hour_speed = met_per_second_speed * const.M_TO_KM / (const.SEC_TO_MIN * const.MIN_TO_HOUR)
        return km_per_hour_speed

    speeds = dataPnts.apply(get_ret_move_speed, cutoff = 25)
    speeds = speeds[~np.isnan(speeds)]
    q9, q8, q7, q6, q5 = np.quantile(speeds, [0.9, 0.8, 0.7, 0.6, 0.5])
    print(q9, q8, q7, q6, q5) ### 80% quantile looks alright.

    player_speed_kph = 15 ### this is validated also by https://tennismash.com/2016/11/24/gig-djokovic-fastest-tennis-player-world/#:~:text=As%20many%20might%20expect%2C%20Novak,top%20speed%20of%2014.89kph.
    player_speed_mps = 15 * const.KPH_TO_MPS

    name = 'Return_Analysis_g600_v300_s13_h3'
    from ATMANpy.models.BallTrajectory.flight_bounce_model import load_trajectories, initialize
    xArr, zArr, rpmArr, infoDict = load_trajectories(name)
        
    ######## THEORETICAL CALCULATION --> purely with trajectories ### 
    def find_time_to_bounce(xArr, zArr, infoDict, 
                            height = 1.5, rpm = -2500, launch_degree = 10, travel_meters = 8):
        
        ### initialise search spaces with proper arguments
        import inspect
        kwargList = inspect.getfullargspec(initialize).args
        kwargs = {k: v for k, v in infoDict.items() if k in kwargList}
        
        initial_v_x, initial_v_z, initial_spins, initial_heights = initialize(**kwargs, squeeze = True)
            
        # filter for correct height
        height_idx = np.argmax(initial_heights == height)
        assert height_idx != 0

        rpm_idx = np.argmax(initial_spins == rpm)
        assert rpm_idx != 0

        sub_xArr, sub_zArr = xArr[:, :, rpm_idx, height_idx], zArr[:, :, rpm_idx, height_idx]

        degrees = np.arctan2(initial_v_z[np.newaxis, :], initial_v_x[:, np.newaxis]) * const.RAD_TO_ANGLE
        deg_dev = (degrees - launch_degree) ** 2
        deg_var = deg_dev.mean()
        deg_std = deg_var ** 0.5
        deg_norm = deg_dev ** 0.5 / deg_std

        first_bounce_idxs = np.argmax(sub_zArr == 0, axis = 2) ### this also happens to be the time in flight!
        
        dists = np.full_like(deg_dev, fill_value = np.nan)
        X, Z, T = sub_xArr.shape
        for x in range(X):
            for z in range(Z):
                idx = first_bounce_idxs[x, z]
                dist = sub_xArr[x, z, idx]
                dists[x, z] = dist

        dst_dev = (dists - travel_meters) ** 2
        dst_var = dst_dev.mean()
        dst_std = dst_var ** 0.5
        dst_norm = dst_dev ** 0.5 / dst_std
        dst_norm = dst_dev / dst_dev.mean()

        from ATMANpy.funcs.util_funcs import argmin_multidim
        x_min, z_min = argmin_multidim(deg_norm + dst_norm)

        frames_until_bounce = first_bounce_idxs[x_min, z_min]
        return frames_until_bounce / 25
    
    x0, y0, _ = myShots[1].get_ball_hit_position()
    x1, y1 = myShots[1].get_first_bounce_position()
    orig_dist = Point2D((x0, y0)).distance_to(Point2D((x1, y1))) / const.SCALE * const.FOOT_TO_METER
    
    x0, y0, _ = bestShots[1].get_ball_hit_position()
    x1, y1 = bestShots[1].get_first_bounce_position()
    best_dist = Point2D((x0, y0)).distance_to(Point2D((x1, y1))) / const.SCALE * const.FOOT_TO_METER

    x0, y0, _ = bestShots[1].get_ball_hit_position()
    x1, y1 = myShots[1].get_first_bounce_position()
    cntr_dist = Point2D((x0, y0)).distance_to(Point2D((x1, y1))) / const.SCALE * const.FOOT_TO_METER

    # filter for correct spins
    chip_spin = -2500 ### corresponds to our examples.
    tops_spin = 1500  ### corresponds to sample average.
    height = 1.5 ### hit height

    timeDict = {'orig_lo': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = chip_spin, launch_degree = chip_mode_lo, travel_meters = orig_dist),
                'orig_hi': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = chip_spin, launch_degree = chip_mode_hi, travel_meters = orig_dist),
                'orig_tp': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = tops_spin, launch_degree = tops_mode, travel_meters = orig_dist), 
                'best_lo': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = chip_spin, launch_degree = chip_mode_lo, travel_meters = best_dist),
                'best_hi': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = chip_spin, launch_degree = chip_mode_hi, travel_meters = best_dist),
                'best_tp': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = tops_spin, launch_degree = tops_mode, travel_meters = best_dist), 
                'cntr_lo': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = chip_spin, launch_degree = chip_mode_lo, travel_meters = cntr_dist), 
                'cntr_hi': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = chip_spin, launch_degree = chip_mode_hi, travel_meters = cntr_dist),
                'cntr_tp': find_time_to_bounce(xArr, zArr, infoDict, height = height, rpm = tops_spin, launch_degree = tops_mode, travel_meters = cntr_dist)
                }

    time_dist_frame = pd.Series(timeDict, name = 'time').to_frame()
    time_dist_frame['time_adj'] = time_dist_frame['time'] - seconds_needed_to_recover
    time_dist_frame['dist_met'] = time_dist_frame['time_adj'] * player_speed_mps
    time_dist_frame['dist_pix'] = time_dist_frame['dist_met'] * const.METER_TO_FOOT * const.SCALE

    ### Tables
    def f(which):
        if which is None:
            return {'Time to Bounce\n(in seconds)\n\n': ''}
        else:   
            return {'Time to Bounce\n(in seconds)\n\n': f'{timeDict[which]}'}

    DARK_GREEN_COLOR = interpolate_tuples(const.BGR['leafy_green'], (0, 0, 0), 0.75)
    DARK_CHIP_LO_COLOR = interpolate_tuples(CHIP_LO_COLOR, (0, 0, 0), 0.95)
    DARK_CHIP_HI_COLOR = interpolate_tuples(CHIP_HI_COLOR, (0, 0, 0), 0.95)
    DARK_TOPS_COLOR = interpolate_tuples(TOPS_COLOR, (0, 0, 0), 0.95)
    table_text_thickness = 2

    kwargDict = {'dimensions': (960, 540), 'skeleton_shrinkage': 0.42, 'index_offset': 0.1,
                 'header_text_size': 2, 'index_text_size': 2, 'data_text_size': 2, 'resolution': 2000,
                 'automatic_vertical_offset_text_line_height': 0.2, 
                 'data_text_color': (255, 255, 255), #const.BGR['leafy_green'], 
                 'header_text_color': (255, 255, 255), #const.BGR['leafy_green'], 
                 'index_text_color': (255, 255, 255), #const.BGR['leafy_green'],
                 'data_text_thickness': table_text_thickness, 
                 'header_text_thickness': table_text_thickness, 
                 'index_text_thickness': table_text_thickness,
                 }

    tabDict = {'High Chip': f('orig_hi'),
               'Low Chip': f('orig_lo'),
               'Topspin': f('orig_tp')}
    myTab = Table(tabDict, **kwargDict)
    myTab.set_back_color(DARK_CHIP_HI_COLOR, 1, 0)
    myTab.set_back_color(DARK_CHIP_HI_COLOR, 1, 1)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 0)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 1)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 1)

    myTab.set_text_color(DARK_CHIP_HI_COLOR, 1, 0)
    myTab.set_text_color(DARK_CHIP_HI_COLOR, 1, 1)
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 1)
    tabFrm_orig1 = myTab.render()
    # tabFrm_orig1.show()

    tabDict = {'High Chip': f('orig_hi'),
               'Low Chip': f('orig_lo'),
               'Topspin': f('orig_tp')}
    myTab = Table(tabDict, **kwargDict)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 0)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 1)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 0)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 1)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 1)
    
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 1)

    tabFrm_orig2 = myTab.render()
    # tabFrm_orig2.show()

    tabDict = {'High Chip': f('orig_hi'),
               'Low Chip': f('orig_lo'),
               'Topspin': f('orig_tp')}
    myTab = Table(tabDict, **kwargDict)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 0)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 1)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 0)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 1)
    myTab.set_back_color(TOPS_COLOR, 3, 0)
    myTab.set_back_color(TOPS_COLOR, 3, 1)
    tabFrm_orig3 = myTab.render()
    # tabFrm_orig3.show()

    tabDict = {'High Chip': f('best_hi'),
               'Low Chip': f('best_lo'),
               'Topspin': f('best_tp')}
    myTab = Table(tabDict, **kwargDict)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 0)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 1)
    myTab.set_back_color(DARK_CHIP_LO_COLOR, 2, 0)
    myTab.set_back_color(DARK_CHIP_LO_COLOR, 2, 1)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 1)

    myTab.set_text_color(DARK_CHIP_LO_COLOR, 2, 0)
    myTab.set_text_color(DARK_CHIP_LO_COLOR, 2, 1)
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 1)
    tabFrm_best1 = myTab.render()
    # tabFrm_best1.show()

    tabDict = {'High Chip': f('best_hi'),
               'Low Chip': f('best_lo'),
               'Topspin': f('best_tp')}
    myTab = Table(tabDict, **kwargDict)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 0)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 1)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 0)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 1)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_back_color(DARK_TOPS_COLOR, 3, 1)
    
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 0)
    myTab.set_text_color(DARK_TOPS_COLOR, 3, 1)

    tabFrm_best2 = myTab.render()
    # tabFrm_best2.show()

    tabDict = {'High Chip': f('best_hi'),
               'Low Chip': f('best_lo'),
               'Topspin': f('best_tp')}
    myTab = Table(tabDict, **kwargDict)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 0)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 1)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 0)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 1)
    myTab.set_back_color(TOPS_COLOR, 3, 0)
    myTab.set_back_color(TOPS_COLOR, 3, 1)
    tabFrm_best3 = myTab.render()
    # tabFrm_best3.show()

    tabDict = {'High Chip': f('cntr_hi'),
               'Low Chip': f('cntr_lo'),
               'Topspin': f('cntr_tp')}
    myTab = Table(tabDict, **kwargDict)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 0)
    myTab.set_back_color(CHIP_HI_COLOR, 1, 1)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 0)
    myTab.set_back_color(CHIP_LO_COLOR, 2, 1)
    myTab.set_back_color(TOPS_COLOR, 3, 0)
    myTab.set_back_color(TOPS_COLOR, 3, 1)
    tabFrm_cntr3 = myTab.render()
    # tabFrm_cntr3.show()


    # tabDict = {'High Chip': f('orig_hi'),
    #            'Low Chip': f('orig_lo'),
    #            'Topspin': f('orig_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 1, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 1, 1)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_orig1 = myTab.render()
    # # tabFrm_orig1.show()

    # tabDict = {'High Chip': f('orig_hi'),
    #            'Low Chip': f('orig_lo'),
    #            'Topspin': f('orig_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_orig2 = myTab.render()
    # # tabFrm_orig2.show()

    # tabDict = {'High Chip': f('orig_hi'),
    #            'Low Chip': f('orig_lo'),
    #            'Topspin': f('orig_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # tabFrm_orig3 = myTab.render()
    # # tabFrm_orig3.show()

    # tabDict = {'High Chip': f('best_hi'),
    #            'Low Chip': f('best_lo'),
    #            'Topspin': f('best_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 2, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 2, 1)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_best1 = myTab.render()
    # # tabFrm_best1.show()

    # tabDict = {'High Chip': f('best_hi'),
    #            'Low Chip': f('best_lo'),
    #            'Topspin': f('best_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_best2 = myTab.render()
    # # tabFrm_best2.show()

    # tabDict = {'High Chip': f('best_hi'),
    #            'Low Chip': f('best_lo'),
    #            'Topspin': f('best_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # tabFrm_best3 = myTab.render()
    # # tabFrm_best3.show()

    # tabDict = {'High Chip': f('cntr_hi'),
    #            'Low Chip': f('cntr_lo'),
    #            'Topspin': f('cntr_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # tabFrm_cntr3 = myTab.render()
    # # tabFrm_cntr3.show()


    # tabDict = {'High Chip': f('orig_hi'),
    #            'Low Chip': f('orig_lo'),
    #            'Topspin': f('orig_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 1, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 1, 1)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_orig1 = myTab.render()
    # # tabFrm_orig1.show()

    # tabDict = {'High Chip': f('orig_hi'),
    #            'Low Chip': f('orig_lo'),
    #            'Topspin': f('orig_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 1, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 1, 1)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_orig1 = myTab.render()
    # # tabFrm_orig1.show()

    # tabDict = {'High Chip': f('orig_hi'),
    #            'Low Chip': f('orig_lo'),
    #            'Topspin': f('orig_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_orig2 = myTab.render()
    # # tabFrm_orig2.show()

    # tabDict = {'High Chip': f('orig_hi'),
    #            'Low Chip': f('orig_lo'),
    #            'Topspin': f('orig_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # tabFrm_orig3 = myTab.render()
    # # tabFrm_orig3.show()

    # tabDict = {'High Chip': f('best_hi'),
    #            'Low Chip': f('best_lo'),
    #            'Topspin': f('best_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 2, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 2, 1)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_best1 = myTab.render()
    # # tabFrm_best1.show()

    # tabDict = {'High Chip': f('best_hi'),
    #            'Low Chip': f('best_lo'),
    #            'Topspin': f('best_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 0)
    # myTab.set_text_color(DARK_GREEN_COLOR, 3, 1)
    # tabFrm_best2 = myTab.render()
    # # tabFrm_best2.show()

    # tabDict = {'High Chip': f('best_hi'),
    #            'Low Chip': f('best_lo'),
    #            'Topspin': f('best_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # tabFrm_best3 = myTab.render()
    # # tabFrm_best3.show()

    # tabDict = {'High Chip': f('cntr_hi'),
    #            'Low Chip': f('cntr_lo'),
    #            'Topspin': f('cntr_tp')}
    # myTab = Table(tabDict, **kwargDict)
    # tabFrm_cntr3 = myTab.render()
    # # tabFrm_cntr3.show()

    ### LINES FOR RASTER ###
    put_kwargs = {'thickness': 2, 'color': const.BGR['bright_grey']}
    y = 145

    from ATMANpy.types.Lines import Arrow
    aPt, bPt = Point2D((430, y)), Point2D((30, y))
    tabArr_orig = Arrow.make(aPt, bPt, hat_length = 20)

    aPt, bPt = Point2D((780, y)), Point2D((930, y))
    tabArr_best = Arrow.make(aPt, bPt, hat_length = 20)

    # aPt, bPt = Point2D((600, y + 55)), Point2D((600, y + 370))
    # tabArr_cntr = Arrow.make(aPt, bPt, hat_length = 20)

    # testRst = FrameRaster(2, 2)
    # testRst.add(tabFrm_orig3.put(tabArr_orig, **put_kwargs), 0, 1)
    # testRst.add(tabFrm_best3.put(tabArr_best, **put_kwargs), 1, 0)
    # testRst.render().show()

    # testRst = FrameRaster(2, 2)
    # testRst.add(tabFrm_orig3.put(tabArr_cntr, **put_kwargs), 0, 1)
    # testRst.add(tabFrm_best3.put(tabArr_best, **put_kwargs), 1, 0)
    # testRst.render().show()

    tabFrm_orig1 = tabFrm_orig1.put(tabArr_orig, **put_kwargs)
    tabFrm_orig2 = tabFrm_orig2.put(tabArr_orig, **put_kwargs)
    tabFrm_orig3 = tabFrm_orig3.put(tabArr_orig, **put_kwargs)
    tabFrm_best1 = tabFrm_best1.put(tabArr_best, **put_kwargs)
    tabFrm_best2 = tabFrm_best2.put(tabArr_best, **put_kwargs)
    tabFrm_best3 = tabFrm_best3.put(tabArr_best, **put_kwargs)
    tabFrm_cntr3 = tabFrm_cntr3.put(tabArr_best, **put_kwargs)

    ###########################
    from ATMANpy.types.Overlays import reverse_cascade
    revRst, revFrms = reverse_cascade(myRst, keep_which = [(0, 0), (1, 1)], frames_per_element = 10)
    # DecapStream(revFrms, 0, 100).show()

    ### Reverse return
    splineFrms_orig_rev = BGRFrames.from_frames([early_orig_scaledFrms[-10:], splineFrms_orig]).reverse()
    splineFrms_best_rev = BGRFrames.from_frames([early_best_scaledFrms[-10:], splineFrms_best]).reverse()
    # DecapStream(splineFrms_orig_rev, 0, 100).show()
    # DecapStream(splineFrms_best_rev, 0, 100).show()

    ### Play serve again
    splineFrms_orig_srv = BGRFrames.from_frames([early_orig_scaledFrms[-10:], splineFrms_orig_dim[:len(myShots[0])]])
    splineFrms_best_srv = BGRFrames.from_frames([early_best_scaledFrms[-10:], splineFrms_best_dim[:len(bestShots[0])]])
    # DecapStream(splineFrms_orig_srv, 0, 100).show()
    # DecapStream(splineFrms_best_srv, 0, 100).show()

    splineFrms_orig_ret = splineFrms_orig_dim[len(myShots[0]):]
    splineFrms_best_ret = splineFrms_best_dim[len(bestShots[0]):]
    # DecapStream(splineFrms_orig_ret, 0, 100).show()
    # DecapStream(splineFrms_best_ret, 0, 100).show()

    ###### prepare player masks #######
    ### a) low chip
    ### get features
    my_conf = confs[myShots[0].get_match_index()]
    plySF = StreamFeature.from_store(my_conf['video_id'], my_conf['store_name'], featname = 'botPlayerCnt')
    backFrm = ConstFeature.from_store(my_conf['video_id'], my_conf['store_name'], featname = 'median_backFrm').feature
    backFrm = backFrm.rescale(1.5)

    hit_frm = myShots[1].get_frames()[0]
    botCnt = plySF[hit_frm].rescale(1.5)

    shot_idx = len(myShots[0])
    # orig_scaledFrms[shot_idx].put(botCnt).show()

    ### some masks
    # player mask
    playerMask = MaskFrame.false(like = splineFrms_orig_ret[-1]).within_contour(botCnt)

    # keep the scoreboard the same
    scoreReg = Region.from_points(Point2D((0, 442)), Point2D((960, 540)))
    scoreMask = MaskFrame.false(like = splineFrms_orig_ret[-1]).within_region(scoreReg)

    # splineFrms_orig_ret[-1].put(scoreReg).show()

    keepMask_orig = playerMask | scoreMask
    shot_idx_orig = len(myShots[0])
    backFrm_orig = backFrm

    ### b) high chip
    ### get features
    my_conf = confs[bestShots[0].get_match_index()]
    plySF = StreamFeature.from_store(my_conf['video_id'], my_conf['store_name'], featname = 'botPlayerCnt')
    backFrm = ConstFeature.from_store(my_conf['video_id'], my_conf['store_name'], featname = 'median_backFrm').feature
    backFrm = backFrm.rescale(1.5)

    hit_frm = bestShots[1].get_frames()[0]
    botCnt = plySF[hit_frm].rescale(1.5)

    # orig_scaledFrms[shot_idx].put(botCnt).show()

    ### some masks
    # player mask
    playerMask = MaskFrame.false(like = splineFrms_best_ret[-1]).within_contour(botCnt)

    # keep the scoreboard the same
    scoreReg = Region.from_points(Point2D((0, 445)), Point2D((960, 540)))
    scoreMask = MaskFrame.false(like = splineFrms_best_ret[-1]).within_region(scoreReg)

    # splineFrms_best_ret[-1].put(scoreReg).show()

    keepMask_best = playerMask | scoreMask
    shot_idx_best = len(bestShots[0])
    backFrm_best = backFrm

    ######## prepare circles #########
    lo_dist, hi_dist, tp_dist = time_dist_frame.loc['orig_lo', 'dist_pix'], time_dist_frame.loc['orig_hi', 'dist_pix'], time_dist_frame.loc['orig_tp', 'dist_pix']
    
    def make_thick_circle(x, y, radius, theta_lo, theta_hi):
        cntPt = Point3D((x, y, 0))
        radii = [0.995 * radius, radius, 1.005 * radius]
        return Circles3D([Circle3D.from_point3D(cntPt, limit_theta = (theta_lo * np.pi, theta_hi * np.pi), radius = r) for r in radii])

    null_frames = round(seconds_needed_to_recover * 25)
    flightSer = time_dist_frame['time_adj'] * 25
    flightSer = flightSer.round().astype(int)
    min_m = 1
    min_px = min_m * const.METER_TO_FOOT * const.SCALE

    # a) for the original shot
    x, y = myShots[1].hitPlayer.get_initial_position()

    minCrc = make_thick_circle(x, y, min_px, 0, 2).recolor(CHIP_LO_COLOR)

    max_px = time_dist_frame.loc['orig_lo', 'dist_pix']
    maxCrc = make_thick_circle(x, y, max_px, 0, 2).recolor(CHIP_LO_COLOR)
    flight_frames = flightSer['orig_lo']

    step_dim = (1 - dim_factor) / (orig_scaledFrms.length() - 1) 
    step_dim = 0.011

    crc_frames = []
    for i, myFrm in enumerate(splineFrms_orig_ret):
        if i < null_frames:
            itplCrc = minCrc
        elif null_frames <= i < null_frames + flight_frames:
            lmbd = (i - null_frames) / flight_frames
            itplCrc = vs.interpolate_shape(minCrc, maxCrc, lmbd)
        else:
            itplCrc = maxCrc
        
        itplCrc = itplCrc.recolor(CHIP_LO_COLOR)

        myView = View.make(brestCam_resized_half, Scene([itplCrc]), resolution = prmDict_nolum['argument_resolution'])
        crcFrm = myView.snap(over_array = myFrm.array.copy())
        crcFrm_mixed = crcFrm.mask(~keepMask_orig) + orig_scaledFrms_dim[shot_idx_orig].dim((1 - step_dim) ** i).mask(keepMask_orig)
        crc_frames.append(crcFrm_mixed)

    maxCrc_orig = maxCrc
    crcFrms_orig = BGRFrames.from_single_frames(crc_frames)    
    # DecapStream(crcFrms_orig, 0, 100).show()

    # b) for the original shot
    x, y = bestShots[1].hitPlayer.get_initial_position()

    minCrc = make_thick_circle(x, y, min_px, 0, 2).recolor(CHIP_HI_COLOR)

    max_px = time_dist_frame.loc['best_hi', 'dist_pix']
    maxCrc = make_thick_circle(x, y, max_px, 0, 2).recolor(CHIP_HI_COLOR)
    flight_frames = flightSer['best_hi']

    step_dim = 0.01

    crc_frames = []
    for i, myFrm in enumerate(splineFrms_best_ret):
        if i < null_frames:
            itplCrc = minCrc
        elif null_frames <= i < null_frames + flight_frames:
            lmbd = (i - null_frames) / flight_frames
            itplCrc = vs.interpolate_shape(minCrc, maxCrc, lmbd)
        else:
            itplCrc = maxCrc
        
        itplCrc = itplCrc.recolor(CHIP_HI_COLOR)

        myView = View.make(brestCam_resized_half, Scene([itplCrc]), resolution = prmDict_nolum['argument_resolution'])
        crcFrm = myView.snap(over_array = myFrm.array.copy())
        crcFrm_mixed = crcFrm.mask(~keepMask_best) + best_scaledFrms_dim[shot_idx_best].dim((1 - step_dim) ** i).mask(keepMask_best)
        crc_frames.append(crcFrm_mixed)

    maxCrc_best = maxCrc
    crcFrms_best = BGRFrames.from_single_frames(crc_frames)    
    # DecapStream(crcFrms_best, 0, 100).show()

    #### fade out players, fade out trajectories, and fade in Xs
    origPt = Point3D.from_point2D(Point2D(myShots[1].get_first_bounce_position()), z = 0)
    bestPt = Point3D.from_point2D(Point2D(bestShots[1].get_first_bounce_position()), z = 0)

    MARK_SIZE = 10
    from ATMANpy.types.ShapesTennis3D import Mark
    origMark = Mark.make(origPt, size = MARK_SIZE)
    bestMark = Mark.make(bestPt, size = MARK_SIZE)

    origPt_repos = myShots[1].get_optimal_repositioning_position(reposition_to_meters_behind_baseline = 2, shot_is_met_at_meters_behind_baseline = 0)
    origPt_repos = Point3D.from_point2D(origPt_repos, z = 0)
    origMark_repos = Mark.make(origPt_repos, size = MARK_SIZE, color = const.BGR['intense_red'])
    
    bestPt_repos = bestShots[1].get_optimal_repositioning_position(reposition_to_meters_behind_baseline = 2, shot_is_met_at_meters_behind_baseline = 0)
    bestPt_repos = Point3D.from_point2D(bestPt_repos, z = 0)
    bestMark_repos = Mark.make(bestPt_repos, size = MARK_SIZE, color = const.BGR['intense_red'])

    ### a) low chip
    shapeFrms, _ = vs.animate_static(Scene([maxCrc_orig, origMark_repos, origMark]), static_cam = brestCam_resized_half, 
                                    argument_resolution = prmDict_nolum['argument_resolution'],
                                    animate_over = backFrm_orig.dim(dim_factor))
    shapeFrm = shapeFrms[0]

    endFrm_orig = crcFrms_orig[-1].mask(keepMask_orig) + shapeFrm.mask(~keepMask_orig)

    itplFrms_orig = vs.interpolate_frames(crcFrms_orig[-1], endFrm_orig, 50)
    # DecapStream(itplFrms_orig, 0, 100).show()

    ### b) hi chip
    shapeFrms, _ = vs.animate_static(Scene([maxCrc_best, bestMark_repos, bestMark]), static_cam = brestCam_resized_half, 
                                    argument_resolution = prmDict_nolum['argument_resolution'],
                                    animate_over = backFrm_best.dim(dim_factor))
    shapeFrm = shapeFrms[0]

    endFrm_best = crcFrms_best[-1].mask(keepMask_best) + shapeFrm.mask(~keepMask_best)

    itplFrms_best = vs.interpolate_frames(crcFrms_best[-1], endFrm_best, 50)
    # DecapStream(itplFrms_best, 0, 100).show()

    script_line =   """
                        First, consider the original example in the top left. 
                        The expanding range of motion depicted here represents the court positions
                        that the returner can comfortably reach, assuming a third of a second of recovery time 
                        and a maximum shuffle speed of 4.2 m/s, which is estimated from our data.
                    """
    
    myRst = FramesRaster(2, 2)
    myRst.add(splineFrms_orig_rev.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(splineFrms_best[-1].framed(FRAME_THICKNESS, FRAME_COLOR).dim(dim_factor), 1, 1)

    revFrms = myRst.render()
    # DecapStream(revFrms, 0, 100).show()

    fadeFrms = vs.interpolate_frames(tabFrms_match[-1], revFrms[0], 50)

    myRst.add(splineFrms_orig_srv.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    srvFrms = myRst.render()
    # DecapStream(srvFrms, 0, 100).show()
    
    myRst.add(crcFrms_orig.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    crcFrms = myRst.render()
    # crcFrms = overlay_center(crcFrms, leg1Frm_crc_lo)
    # DecapStream(crcFrms, 0, 100).show()

    myRst.add(crcFrms_orig[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(BGRFrames.from_frame(tabFrm_orig1, 50), 0, 1)
    statFrms = myRst.render()

    itplFrms = vs.interpolate_frames(crcFrms[-1], statFrms[0], 50)

    jointFrms = BGRFrames.from_frames([fadeFrms, revFrms, srvFrms, crcFrms, itplFrms, statFrms])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    script_line =   """
                        The low-angled chip shot gives the returner enough time to recover to the optimal spot, marked with a red X.
                    """

    myRst.add(itplFrms_orig.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    rastFrms0 = myRst.render()
    # rastFrms0 = overlay_center(rastFrms0, leg1Frm_crc_lo)
    DecapStream(rastFrms0, 0, 100).show()
    all_frames.append(rastFrms0)
    script_lines.append(script_line)

    script_line =   """
                        Had he played a high angle return, the additional .24 seconds of time-until-bounce would have given him more range of motion, 
                        which, in this case, would not have been necessary to reach the optimal point.
                    """
    x, y = myShots[1].hitPlayer.get_initial_position()

    px = time_dist_frame.loc['orig_lo', 'dist_pix']
    loCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_LO_COLOR)
    
    px = time_dist_frame.loc['orig_hi', 'dist_pix']
    hiCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_HI_COLOR)

    px = time_dist_frame.loc['orig_tp', 'dist_pix']
    tpCrcs = make_thick_circle(x, y, px, 0, 2).recolor(TOPS_COLOR)
    
    statFrms, _ = vs.animate_static(Scene([loCrcs, hiCrcs, origMark_repos, origMark]), static_cam = brestCam_resized_half, 
                                    argument_resolution = prmDict_nolum['argument_resolution'],
                                    animate_over = backFrm_orig.dim(dim_factor))
    
    endFrm = crcFrms_orig[-1].mask(keepMask_orig) + statFrms[-1].mask(~keepMask_orig)
    endFrms = BGRFrames.from_frame(endFrm, 50)

    myRst.add(endFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(tabFrm_orig2, 0, 1)
    rastFrms1 = myRst.render()
    # rastFrms1 = overlay_center(rastFrms1, leg2Frm_crc)
    # DecapStream(rastFrms1, 0, 100).show()

    itplFrms = vs.interpolate_frames(rastFrms0[-1], rastFrms1[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    all_frames.append(itplFrms)
    script_lines.append(script_line)

    script_line =   f"""
                        A comparable topspin shot down the line would travel {time_dist_frame.loc['orig_tp', 'time']:.2f} seconds, 
                        which does not afford the returning player with enough time to recover to the optimal spot. 
                    """
    
    statFrms, _ = vs.animate_static(Scene([loCrcs, hiCrcs, tpCrcs, origMark_repos, origMark]), static_cam = brestCam_resized_half, 
                                    argument_resolution = prmDict_nolum['argument_resolution'],
                                    animate_over = backFrm_orig.dim(dim_factor))
    
    endFrm = crcFrms_orig[-1].mask(keepMask_orig) + statFrms[-1].mask(~keepMask_orig)
    endFrms = BGRFrames.from_frame(endFrm, 50)

    myRst.add(endFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 0, 0)
    myRst.add(tabFrm_orig3, 0, 1)
    rastFrms2 = myRst.render()
    # rastFrms2 = overlay_center(rastFrms2, leg3Frm_crc)
    # DecapStream(rastFrms2, 0, 100).show()

    itplFrms = vs.interpolate_frames(rastFrms1[-1], rastFrms2[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    all_frames.append(itplFrms)
    script_lines.append(script_line)

    script_line =   """
                        Now, consider the situation in the lower right corner.
                    """

    myRst = FramesRaster(2, 2)
    myRst.add(endFrms[-1].framed(FRAME_THICKNESS, FRAME_COLOR).dim(dim_factor), 0, 0)
    myRst.add(tabFrm_orig3.dim(dim_factor), 0, 1)
    myRst.add(splineFrms_best_rev.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)

    revFrms = myRst.render()
    # revFrms = overlay_center(revFrms, leg3Frm_crc)
    # DecapStream(revFrms, 0, 100).show()

    myRst.add(splineFrms_best_srv.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    srvFrms = myRst.render()
    # srvFrms = overlay_center(srvFrms, leg3Frm_crc)
    # DecapStream(srvFrms, 0, 100).show()
    
    myRst.add(crcFrms_best.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    crcFrms = myRst.render()
    # crcFrms = overlay_center(crcFrms, leg3Frm_crc)
    # DecapStream(crcFrms, 0, 100).show()

    myRst.add(crcFrms_best[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(BGRFrames.from_frame(tabFrm_best1, 50), 1, 0)
    statFrms = myRst.render()

    itplFrms = vs.interpolate_frames(crcFrms[-1], statFrms[0], 50)

    jointFrms = BGRFrames.from_frames([revFrms, srvFrms, crcFrms, itplFrms, statFrms])
    # DecapStream(jointFrms, 0, 100).show()

    all_frames.append(jointFrms)
    script_lines.append(script_line)

    script_line =   """
                        Since the return goes cross-court, the returner is not required to recover to the other half of the baseline, 
                        so he has less distance to cover.
                    """

    myRst.add(itplFrms_best.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_best1, 1, 0)
    rastFrms0 = myRst.render()
    # rastFrms0 = overlay_center(rastFrms0, leg3Frm_crc)
    # DecapStream(rastFrms0, 0, 100).show()

    all_frames.append(rastFrms0)
    script_lines.append(script_line)

    script_line =   """
                        Again, the low-angled chip's flight time would have been more than enough to recover to the optimal spot.
                    """

    x, y = bestShots[1].hitPlayer.get_initial_position()

    px = time_dist_frame.loc['best_lo', 'dist_pix']
    loCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_LO_COLOR)
    
    px = time_dist_frame.loc['best_hi', 'dist_pix']
    hiCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_HI_COLOR)

    px = time_dist_frame.loc['best_tp', 'dist_pix']
    tpCrcs = make_thick_circle(x, y, px, 0, 2).recolor(TOPS_COLOR)
    
    statFrms, _ = vs.animate_static(Scene([loCrcs, hiCrcs, bestMark_repos, bestMark]), static_cam = brestCam_resized_half, 
                                    argument_resolution = prmDict_nolum['argument_resolution'],
                                    animate_over = backFrm_best.dim(dim_factor))
    
    endFrm = crcFrms_best[-1].mask(keepMask_best) + statFrms[-1].mask(~keepMask_best)
    endFrms = BGRFrames.from_frame(endFrm, 50)

    myRst.add(endFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_best2, 1, 0)
    rastFrms1 = myRst.render()
    # rastFrms1 = overlay_center(rastFrms1, leg3Frm_crc)
    # DecapStream(rastFrms1, 0, 100).show()

    itplFrms = vs.interpolate_frames(rastFrms0[-1], rastFrms1[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    all_frames.append(itplFrms)
    script_lines.append(script_line)

    script_line =   """
                        Even the topspin return cross-court would have afforded the returning player with enough time to make it to an adequate position on court; 
                        he can not quite reach the optimal position comfortably, but he is not far away.
                    """

    statFrms, _ = vs.animate_static(Scene([loCrcs, hiCrcs, tpCrcs, bestMark_repos, bestMark]), static_cam = brestCam_resized_half, 
                                    argument_resolution = prmDict_nolum['argument_resolution'],
                                    animate_over = backFrm_best.dim(dim_factor))
    
    endFrm = crcFrms_best[-1].mask(keepMask_best) + statFrms[-1].mask(~keepMask_best)
    endFrms = BGRFrames.from_frame(endFrm, 50)

    myRst.add(endFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_best3, 1, 0)
    rastFrms2 = myRst.render()
    # rastFrms2 = overlay_center(rastFrms2, leg3Frm_crc)
    # DecapStream(rastFrms2, 0, 100).show()

    itplFrms = vs.interpolate_frames(rastFrms1[-1], rastFrms2[0], 50)
    # DecapStream(itplFrms, 0, 100).show()

    all_frames.append(itplFrms)
    script_lines.append(script_line)

    script_line =   """
                        However, notice that the returner is pulled out wide for this serve. 
                        If, in this situation, he were to play a return down-the-line ...
                    """
    all_frames.append(rastFrms2)
    script_lines.append(script_line)

    script_line =   """
                        ... the returner would have more ground to cover to the optimal spot. 
                        At the same time, the reduced flight-time of a return down the line 
                        affords the returner with less time to make up that distance.
                    """
    
    ## interpolate marks and circles
    x, y = bestShots[1].hitPlayer.get_initial_position()

    px = time_dist_frame.loc['best_lo', 'dist_pix']
    loCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_LO_COLOR)
    
    px = time_dist_frame.loc['best_hi', 'dist_pix']
    hiCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_HI_COLOR)

    px = time_dist_frame.loc['best_tp', 'dist_pix']
    tpCrcs = make_thick_circle(x, y, px, 0, 2).recolor(TOPS_COLOR)

    crcScn_best = Scene([loCrcs, hiCrcs, tpCrcs])
    
    px = time_dist_frame.loc['cntr_lo', 'dist_pix']
    loCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_LO_COLOR)
    
    px = time_dist_frame.loc['cntr_hi', 'dist_pix']
    hiCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_HI_COLOR)

    px = time_dist_frame.loc['cntr_tp', 'dist_pix']
    tpCrcs = make_thick_circle(x, y, px, 0, 2).recolor(TOPS_COLOR)

    crcScn_cntr = Scene([loCrcs, hiCrcs, tpCrcs])

    x, y = myShots[1].hitPlayer.get_initial_position()

    px = time_dist_frame.loc['orig_lo', 'dist_pix']
    loCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_LO_COLOR)
    
    px = time_dist_frame.loc['orig_hi', 'dist_pix']
    hiCrcs = make_thick_circle(x, y, px, 0, 2).recolor(CHIP_HI_COLOR)

    px = time_dist_frame.loc['orig_tp', 'dist_pix']
    tpCrcs = make_thick_circle(x, y, px, 0, 2).recolor(TOPS_COLOR)

    crcScn_orig = Scene([loCrcs, hiCrcs, tpCrcs])

    best_markScn = Scene([bestMark_repos, bestMark])
    orig_markScn = Scene([origMark_repos, origMark])

    full_bestScn = best_markScn.combine(crcScn_best)
    full_cntrScn = orig_markScn.combine(crcScn_cntr)
    full_origScn = orig_markScn.combine(crcScn_orig)

    itplScns = vs.interpolate_scenes(full_bestScn, full_cntrScn, 50)

    crcFrms_cntr, _ = vs.animate_sequence(itplScns, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, 
                                            animate_over = backFrm_best.dim(dim_factor), **prmDict)
    # DecapStream(crcFrms_cntr, 0, 100).show()

    end_frames = []
    for frm in crcFrms_cntr:
        endFrm = crcFrms_best[-1].mask(keepMask_best) + frm.mask(~keepMask_best)
        end_frames.append(endFrm)

    endFrms = BGRFrames.from_single_frames(end_frames)
    # DecapStream(endFrms, 0, 100).show()

    tabFrms_itpl = vs.interpolate_frames(tabFrm_best3, tabFrm_cntr3, 50)

    myRst.add(endFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrms_itpl, 1, 0)
    rastFrms_cntr = myRst.render()
    # rastFrms_cntr = overlay_center(rastFrms_cntr, leg3Frm_crc)
    # DecapStream(rastFrms_cntr, 0, 100).show()

    all_frames.append(rastFrms_cntr)
    script_lines.append(script_line)

    script_line =   """
                        In this situation, where the returner is pulled somewhat out wide and attempts a return down the line, 
                        only the high-arcing chip buys him enough time to get into the optimal position at the time of bounce.
                    """
    staticFrms = BGRFrames.from_frame(rastFrms_cntr[-1], STATIC_FRAMES)
    all_frames.append(staticFrms)
    script_lines.append(script_line)

    ### SEGUE TO INFERENCE
    # given all of this -> where SHOULD a chipped ball go to improve the returners chances of winning?
    # -> what chip is more promising than an average topspin return?

    # script_line = ""
    # all_frames.append(staticFrms[:50])
    # script_lines.append(script_line)

    # de-render the shapes in the top left
    script_line =   """
                        Up until this point, we have tackled the chip return entirely descriptively.
                        We know, roughly, what an average challenger-level player does;
                        what we are lacking is an answer to the question: what should he do.

                        Given the 12% win-rate deficit of the chip return on aggregate: 
                        should a return be chipped by choice, at all? And if yes: where, and how?

                        To this end, we estimate a model on our data that calculates the returner's win probability 
                        by taking into account a variety of factors, 
                        such as return placement, launch angle, speed, spin and on-court positioning of the returner and server.
                    """

    all_frames.append(staticFrms)
    script_lines.append(script_line)
    
    script_line = "We can then use this model to find the most promising return in any specific situation."

    myRst = FramesRaster(2, 2, borders = None)
    myRst.add(endFrms[-1].framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    myRst.add(tabFrm_orig3.dim(0.5), 0, 1)
    myRst.add(tabFrm_cntr3, 1, 0)

    num_frames = 10
    blackFrm = BGRFrame.make_black(960, 540)
    blackFrms = BGRFrames.from_frame(blackFrm, num_frames)

    ### derender top left
    derender_list = []
    for i in range(len(full_origScn)):
        cutScn = full_origScn[:-(i+1)]

        statFrms, _ = vs.animate_static(cutScn, static_cam = brestCam_resized_half, 
                                        argument_resolution = prmDict_nolum['argument_resolution'],
                                        animate_over = backFrm_orig.dim(dim_factor), num_frames = num_frames)
        statFrm = statFrms[0]
        
        derFrm = crcFrms_orig[-1].mask(keepMask_orig) + statFrm.mask(~keepMask_orig)
        derFrm = derFrm.framed(FRAME_THICKNESS, FRAME_COLOR).dim(0.5)
        derFrms =  BGRFrames.from_frame(derFrm, num_frames)

        derender_list.append(derFrms)

    derender_list.append(blackFrms)

    derFrms = BGRFrames.from_frames(derender_list)
    # DecapStream(derFrms, 0, 100).show()

    myRst.add(derFrms, 0, 0)
    rstFrms_der0 = myRst.render()
    # DecapStream(rstFrms_der0, 0, 100).show()

    ### derender table top right
    myRst.add(derFrms[-1], 0, 0)
    myRst.add(blackFrms, 0, 1)

    rstFrms_der1 = myRst.render()
    # DecapStream(rstFrms_der1, 0, 100).show()

    ### derender table bottom left
    myRst.add(blackFrms, 1, 0)

    rstFrms_der2 = myRst.render()
    # DecapStream(rstFrms_der2, 0, 100).show()

    ### derender bottom right
    derender_list = []
    for i in range(len(full_cntrScn)):
        cutScn = full_cntrScn[:-(i+1)]

        statFrms, _ = vs.animate_static(cutScn, static_cam = brestCam_resized_half, 
                                        argument_resolution = prmDict_nolum['argument_resolution'],
                                        animate_over = backFrm_best.dim(dim_factor), num_frames = num_frames)
        statFrm = statFrms[0]
        
        derFrm = crcFrms_best[-1].mask(keepMask_best) + statFrm.mask(~keepMask_best)
        derFrms =  BGRFrames.from_frame(derFrm, num_frames)

        derender_list.append(derFrms)

    # derender_list.append(blackFrms) ### -> this stays!

    derFrms = BGRFrames.from_frames(derender_list)
    # DecapStream(derFrms, 0, 100).show()

    myRst.add(derFrms.framed(FRAME_THICKNESS, FRAME_COLOR), 1, 1)
    rstFrms_der3 = myRst.render()
    # DecapStream(rstFrms_der3, 0, 100).show()

    def get_spline_frames(myPnt):
        from ATMANpy.types.Visual3D import Camera
        myCam = Camera.calibrated_brest_behind().resize(3)
        serve_splines = myPnt[0].feat_spline_cumul(which_endpoint = 'penultimate').values()
        splnScns = [Scene([spln.recolor(BALL_COLOR).resize(1.5) for spln in splns if spln is not None]) for splns in serve_splines]
        realFrms = myPnt[0].stream(confs = confs).decap().frames
        myFrms, _ = vs.animate_sequence(splnScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = realFrms.rescale(3), **prmDict)
        return myFrms
    
    firstFrm = get_spline_frames(bestPnt)[-1].rescale(0.5) ### first inference frame

    itplFrms = vs.interpolate_frames(derFrms[-1], firstFrm, 40)
    itplFrms = itplFrms.framed(FRAME_THICKNESS, FRAME_COLOR)
    myRst.add(itplFrms, 1, 1)
    rstFrms_itpl = myRst.render()

    jointFrms_der = BGRFrames.from_frames([rstFrms_der0, rstFrms_der1, rstFrms_der2, rstFrms_der3, rstFrms_itpl])
    # DecapStream(jointFrms_der, 0, 100).show()

    all_frames.append(jointFrms_der)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)
    # show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)

    # script_line =   """
    #                     Up until this point, we have tackled the chip return entirely descriptively, 
    #                     having made note of the distributions of placement and launch angles, 
    #                     and their impact on court positioning. 
    #                     We know, roughly, what an average challenger-level player does 
    #                     with respect to the chipped return versus the topspin return,
    #                     what we are lacking is an answer to the question: what should he do.
    #                     Where should the chip return be placed to overcome the 12% win-rate deficit over the topspin return?
    #                     And what launch angle should the ball be played at?
    #                 """

    # script_line =   """
    #                     We will now attempt an answer to these questions, with the following caveat: To the extent that
    #                     these normative kinds of questions are the more interesting to answer, they are also more difficult to answer
    #                     statistically, for the lack of a counterfactual. 
    #                     In any specific situation, we only ever observe the return that was actually hit,
    #                     not the set of all other possible returns. Had we observed that set and had we recorded the outcome 
    #                     of the point for each of the possible returns, we could easily give an answer to what the returner should do,
    #                     especially if we had observed the whole set a bunch of times over to average out the randomness that 
    #                     a tennis point is subject to.
    #                     However, we do not inhabit the world of hypotheticals, and must make do with just the one outcome. 
                        
    #                     Though imperfect, an adequate attempt can still be made to answer the question: what should the returner do, here.
    #                 """
    
    # script_line =   """
    #                     To this end, we use the data collected by us to estimate a model that calculates the returner's win probability by taking into account a variety of factors, 
    #                     such as return placement, launch angle, speed, spin and on-court positioning of the returner and server.
    #                     We can then let the model evaluate not just the return that actually was observed, but also all kinds of hypothetical returns, 
    #                     and let it compare their win probability to find the most promising return in any specific situation.
    #                 """
