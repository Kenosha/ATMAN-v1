
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


import ATMANpy.auxil.constants as const
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.funcs.util_funcs import get_folder_struct, maybe_make_folder, make_timestamp
from ATMANpy.types.Abstract import Points, Point, Shots, read_in_all_tables
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.Visual3D import DummyScene
from ATMANpy.types.Shapes3D import Line3D, Circle3D, Circles3D, Region3D, Regions3D
from ATMANpy.types.Overlays import FrameRaster, FramesRaster, Table
from ATMANpy.types.Frames import BGRFrame

from statsmodels.discrete.discrete_model import Logit

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
allTable = read_in_all_tables(confs)
allPnts = Points.from_datatable(allTable, split_by_match = False)
firstPnts = allPnts.filter('is_first_serve')

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

RESOLUTION_MODIFIER = 0.5 # 2
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

prmDict_nolum = {**prmDict}
prmDict_nolum['argument_luminosity'] = 1

STATIC_FRAMES = 100 ### does not matter much once we record: the static frames will be fit to the narration
TRANSITION_FRAMES = 25 

INVISIBLE_ALPHA = 0.05 ### which alpha is accepted as invisible
FULL_ALPHA = 1 ### which alpha is accepted as fully visible --> FULL_ALPHA as parameter not needed except for FADE_IN, FADE_OUT_FACTOR calculation

FADE_IN_FACTOR = (FULL_ALPHA / INVISIBLE_ALPHA) ** (1 / TRANSITION_FRAMES)
FADE_OUT_FACTOR = 1 / FADE_IN_FACTOR ### easy to show that this must hold.

BALL_COLOR = const.BGR['ball_yellow']
CHIP_COLOR = const.BGR['marine_blue']
TOPS_COLOR = const.BGR['light_brown']

BANDWIDTH = 45

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
myDtm = myShots.stream(confs = confs).decap()
scaledFrms = myDtm.frames.rescale(3)

pre_frames = 150
preDtm = myShots[0].stream(confs = confs).widen(pre_frames).decap()
preFrms = preDtm.frames[:pre_frames].rescale(3)
# DecapStream(preFrms, 0, 100).show()

preArr_dimmed = np.stack([f.dim((idx / pre_frames) ** 2).array for idx, f in enumerate(preFrms)], axis = 0)
preFrms_dimmed = BGRFrames(preArr_dimmed)
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
from ATMANpy.types.Overlays import Legend, LegendRow, overlay
legRows = [LegendRow('Topspin', const.BGR['light_brown'], text_size = 2),
           LegendRow('Chip', const.BGR['marine_blue'], text_size = 2)]
myLeg_chip_topspin = Legend(legRows, header_text = '', header_offset = 2, header_thickness = 1,
                            legend_dimensions = (300, 200), resolution = 100)

legFrm = myLeg_chip_topspin.render()

anchorPt = Point2D((1920 // 12, 1080 // 12))

### Flight Splines
serveSF = myShots[0].feat_spline_cumul(which_endpoint = 'penultimate')
returnSF = myShots[1].feat_spline_cumul(which_endpoint = 'penultimate')

serve_scenes = [Scene([spLn.recolor(BALL_COLOR) for spLn in splines if spLn is not None]) for splines in serveSF.values()]
return_scenes = [Scene([spLn.recolor(BALL_COLOR) for spLn in splines if spLn is not None]) for splines in returnSF.values()]
spline_scenes = [*serve_scenes, *return_scenes]
splnScn = spline_scenes[-1]

# ballSF = myShots.feat_ball(which_endpoint = 'penultimate')
# ballSF = myShots.feat_spline(which_endpoint = 'penultimate')
# topSF = myShots.feat_top_player()
# botSF = myShots.feat_bot_player()
# jntSF = ballSF.merge(topSF.rename('Top')).merge(botSF.rename('Bot'))

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
                        We build up our understanding of a good chip by looking at some characteristics of chipped returns in general. 
                        First up: placement.
                    """
    blankFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))
    blankFrms = BGRFrames.from_frame(blankFrm, STATIC_FRAMES)
    all_frames.append(blankFrms)
    script_lines.append(script_line)

    script_line = ""
    all_frames.append(preFrms_dimmed)
    script_lines.append(script_line)

    ####### FLIGHT SUPERIMPOSED OVER BALL
    script_line = ""
    myFrms, animDict = vs.animate_sequence(spline_scenes, start_cam = brestCam_resized, stop_cam = brestCam_resized, background_scene = None, 
                                        cumulate_argument = False, animate_over = scaledFrms, **prmDict)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ####### MORPH TRAJECTORY TO VIRTUAL 
    script_line = ""
    lastFrm = myFrms[-1]
    staticScn = courtScn.combine(splnScn)
    virtualFrm = View.make(brestCam_resized, staticScn).snap()
    myFrms = vs.interpolate_frames(lastFrm, virtualFrm, num_frames = 100)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    myFrms, animDict = vs.animate_cam_motion(splnScn, start_cam = brestCam_resized, stop_cam = heatCam, background_scene = courtScn,
                                            num_frames = TRANSITION_FRAMES, interpolate_func = vs.interpolate_camera_slerp, **prmDict)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = "As in the case of this example return, we collect the bounce position of all returns in our sample."
    myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
    all_frames.append(myFrms)
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
    script_line = ""
    mark_scenes = [bounceMk for _ in range(TRANSITION_FRAMES)]
    myFrms, animDict = vs.animate_sequence(mark_scenes, start_cam = heatCam, stop_cam = heatCam, cumulative_foreground_transparence = FADE_OUT_FACTOR,
                                        background_scene = courtScn, **animDict, **prmDict)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    myFrms = BGRFrames.from_frame(myFrms[-1], TRANSITION_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ####### FADE IN OTHER MARKS
    script_line = ""
    animDict['previous_foreground_scene'] = markScn_mono_trans
    myFrms, animDict = vs.animate_sequence(mark_scenes, background_scene = courtScn, cumulate_argument = False,
                                        cumulative_foreground_transparence = FADE_IN_FACTOR, start_cam = heatCam, stop_cam = heatCam,
                                        **animDict, **prmDict)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line =   """
                        We then flip returns hit back to lefty servers, so that the server's forehand side is always at deuce. 
                        This way, we can check whether returns are hit to the backhand side preferentially.
                    """
    myFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    #### Switch Over Returns to Lefties so we have forehand/backhand
    script_line = ""
    itpl_markScns = vs.interpolate_scenes(markScn_mono, markScn_mono_rect, num_frames = 25)
    myFrms, animDict = vs.animate_sequence(itpl_markScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, **prmDict)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line =   """
                        Now, differentiate between chipped returns and topspin returns. 
                        Chipped returns are marked in blue, topspin returns in orange.
                    """

    from ATMANpy.auxil.court import top_bsl_deu_2D, top_bsl_ad_2D
    fhPt = Point3D.from_point2D(top_bsl_deu_2D, z = 0).front(10).annotate('Forehand')
    bhPt = Point3D.from_point2D(top_bsl_ad_2D, z = 0).front(10).left(25).annotate('Backhand')

    myFrms, animDict = vs.animate_static(markScn_mono_rect, static_cam = heatCam, background_scene = courtScn, annotated_points = [fhPt, bhPt], **prmDict)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    itpl_markScns = vs.interpolate_scenes(markScn_mono_rect, markScn_poly_rect, num_frames = 25)
    myFrms, animDict = vs.animate_sequence(itpl_markScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, annotated_points = [fhPt, bhPt], 
                                        **prmDict)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = "First, consider the empirical distribution of topspin returns."
    myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    # ### fade out chip marks
    # script_line = ""
    # mark_scenes = [tops_markScn for _ in range(TRANSITION_FRAMES)]
    # myFrms, animDict = vs.animate_sequence(mark_scenes, start_cam = heatCam, stop_cam = heatCam, background_scene = courtScn,
    #                                     **animDict, **prmDict, cumulative_foreground_transparence = FADE_OUT_FACTOR)
    # myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    # all_frames.append(myFrms)
    # script_lines.append(script_line)

    # script_line = "From these, we construct a heatmap."
    # myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    # myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
    # all_frames.append(myFrms)
    # script_lines.append(script_line)

    ### Topspin Heatmaps
    script_line = ""
    topsScn = vs.kde_heatmap(tops_retShots_rectified, 
                            position_func = 'get_first_bounce_position', 
                            normalize_density = True,
                            zero_color = TOPS_COLOR, one_color = TOPS_COLOR, 
                            bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

    topsScn_zero = topsScn.realpha(0)
    itplScns = vs.interpolate_dummy_scenes(topsScn_zero, topsScn, STATIC_FRAMES)

    myFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam,
                                        **animDict, **prmDict_nolum, cumulative_foreground_transparence = FADE_OUT_FACTOR)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = "Topspin returns are symmetrically clustered around the center of the court..."
    myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    myFrms = BGRFrames.from_frame(myFrm, TRANSITION_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ### shares in sector
    a, b, c, d, dc, ad = vis_ut.get_court_sectors(side = 'Top')
    shrSer = vis_ut.shares_in_sector(tops_retShots_rectified, side = 'Top')
    shrStrs = [f'{share:>3.0%}' for share in shrSer.values]

    myRegs = [  Region3D.from_region2D(a, z = 0), Region3D.from_region2D(b, z = 0), 
                Region3D.from_region2D(c, z = 0), Region3D.from_region2D(d, z = 0), 
                Region3D.from_region2D(dc, z = 0), Region3D.from_region2D(ad, z = 0)]
    myLns = [regLn.recolor(const.BGR['leafy_green']).resize(3) for reg in myRegs for regLn in reg.lineList]

    centerPts = [reg.center_point().annotate(shrStr).resize(2).recolor(const.BGR['leafy_green']).left(10) for reg, shrStr in zip(myRegs, shrStrs)]

    script_line = "... with a slight preference for the server's backhand."
    myFrms, _ = vs.animate_static(myLns, background_scene = courtScn, static_cam = heatCam, **prmDict_nolum, **animDict,
                                         annotated_points = centerPts)

    myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    # Heatmap for Weighted, 'Choice' Chips 
    script_line = ""
    chipFrm = regFrm_in[regFrm_in.is_chipped == 1]
    topspin_preds = chipFrm.pred_is_topspin.values
    chipScn = vs.kde_heatmap(chip_retShots_rectified, 
                            position_func = 'get_first_bounce_position', 
                            weight_func_or_arr = topspin_preds,
                            normalize_density = True,
                            zero_color = CHIP_COLOR, one_color = CHIP_COLOR,
                            bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

    itplScns = vs.interpolate_dummy_scenes(topsScn, chipScn, STATIC_FRAMES)

    myFrms, animDict = vs.animate_sequence(itplScns, background_scene = courtScn, start_cam = heatCam, stop_cam = heatCam, annotated_points = [fhPt, bhPt],
                                            **prmDict_nolum)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = "On the other hand, for choice chipped returns, we see a more irregular heatmap; some land shallow, others deeper."
    myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    myFrms = BGRFrames.from_frame(myFrm, TRANSITION_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ### shares in sector
    a, b, c, d, dc, ad = vis_ut.get_court_sectors(side = 'Top')
    shrSer = vis_ut.shares_in_sector(chip_retShots_rectified, side = 'Top')
    shrStrs = [f'{share:>3.0%}' for share in shrSer.values]

    myRegs = [  Region3D.from_region2D(a, z = 0), Region3D.from_region2D(b, z = 0), 
                Region3D.from_region2D(c, z = 0), Region3D.from_region2D(d, z = 0), 
                Region3D.from_region2D(dc, z = 0), Region3D.from_region2D(ad, z = 0)]
    myLns = [regLn.recolor(const.BGR['leafy_green']).resize(3) for reg in myRegs for regLn in reg.lineList]

    centerPts = [reg.center_point().annotate(shrStr).resize(2).recolor(const.BGR['leafy_green']).left(10) for reg, shrStr in zip(myRegs, shrStrs)]

    script_line = "As with topspin returns, chipped returns are hit predominantly to the backhand side."
    myFrms, _ = vs.animate_static(myLns, background_scene = courtScn, static_cam = heatCam, **prmDict_nolum, **animDict,
                                         annotated_points = centerPts)

    myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)



# ### TODO: try the contourmaps (50% of balls land within...) or even hillmaps. Contourmaps can be superimposed more easily with backhand and forehand.

# topsScn, _ = vs.kde_contourmap(tops_retShots_rectified, 
#                             position_func = 'get_first_bounce_position', 
#                             threshold_proportion = 0.9, display_at_height = 0,
#                             showReg = vis_ut.topReg,
#                             bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)
# myFrms, _ = vs.animate_static(topsScn, background_scene = courtScn, static_cam = heatCam, **prmDict)
# myFrms[-1].show()

# chipScn, _ = vs.kde_contourmap(chip_retShots_rectified, 
#                             position_func = 'get_first_bounce_position', 
#                             weight_func_or_arr = topspin_preds,
#                             threshold_proportion = 0.5, display_at_height = 0,
#                             showReg = vis_ut.topReg,
#                             bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE, 
#                             n_components = 3)
# myFrms, _ = vs.animate_static(chipScn, background_scene = courtScn, static_cam = heatCam, **prmDict)
# myFrms[-1].show()

# chipScn = vs.kde_hillmap(chip_retShots_rectified, showReg = vis_ut.topReg, resolution = RESOLUTION_KDE, position_func = 'get_first_bounce_position',
#                             weight_func_or_arr = topspin_preds, bandwidth = BANDWIDTH, max_height = 30, mix_colors = True, mix_alpha = True,
#                             one_color = CHIP_COLOR)
# myFrms, _ = vs.animate_static(chipScn, background_scene = courtScn, static_cam = heatCam, **prmDict)
# myFrms[-1].show()

# topsScn = vs.kde_hillmap(tops_retShots_rectified, showReg = vis_ut.topReg, resolution = RESOLUTION_KDE, position_func = 'get_first_bounce_position',
#                             bandwidth = BANDWIDTH, max_height = 30, mix_colors = False, mix_alpha = True,
#                             one_color = TOPS_COLOR)
# myFrms, _ = vs.animate_static(topsScn, background_scene = courtScn, static_cam = heatCam, **prmDict)
# myFrms[-1].show()

# topsScn = vs.kde_heatmap(tops_retShots_rectified, showReg = vis_ut.topReg, resolution = RESOLUTION_KDE, position_func = 'get_first_bounce_position',
#                             bandwidth = BANDWIDTH, mix_colors = False, mix_alpha = True,
#                             one_color = TOPS_COLOR)
# myFrms, _ = vs.animate_static(topsScn, background_scene = courtScn, static_cam = heatCam, **prmDict)
# myFrms[-1].show()


# chipScn = vs.kde_hillmap(chip_retShots_rectified, weight_func_or_arr = topspin_preds, 
#                          showReg = vis_ut.topReg, resolution = RESOLUTION_KDE, position_func = 'get_first_bounce_position',
#                             bandwidth = BANDWIDTH + 10, max_height = 30, mix_colors = False, mix_alpha = True,
#                             one_color = CHIP_COLOR)
# myFrms, _ = vs.animate_static(chipScn, background_scene = courtScn, static_cam = heatCam, **prmDict)
# myFrms[-1].show()

# chipScn = vs.kde_heatmap(chip_retShots_rectified, weight_func_or_arr = topspin_preds, 
#                          showReg = vis_ut.topReg, resolution = RESOLUTION_KDE, position_func = 'get_first_bounce_position',
#                             bandwidth = BANDWIDTH, mix_colors = False, mix_alpha = True,
#                             one_color = CHIP_COLOR)
# myFrms, _ = vs.animate_static(chipScn, background_scene = courtScn, static_cam = heatCam, **prmDict)
# myFrms[-1].show()


# vs.animate_cam_motion(topsScn)


####################################
######### HEATMAP SECTION ##########
####################################

if DO_HEATMAP_SECTION:
    all_frames = []
    script_lines = []

    ### DIFFERENCE BETWEEN TOPSPIN AND CHIP PLACEMENT
    topsScn_not_norm = vs.kde_heatmap(tops_retShots_rectified, 
                                        position_func = 'get_first_bounce_position', 
                                        normalize_density = False,
                                        zero_color = TOPS_COLOR, one_color = TOPS_COLOR, 
                                        bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

    chipScn_not_norm = vs.kde_heatmap(chip_retShots_rectified, 
                                        position_func = 'get_first_bounce_position', 
                                        weight_func_or_arr = topspin_preds,
                                        normalize_density = False,
                                        zero_color = CHIP_COLOR, one_color = CHIP_COLOR,
                                        bandwidth = BANDWIDTH, resolution = RESOLUTION_KDE)

    tops_kde = topsScn_not_norm.alphaArrs[0]
    chip_kde = chipScn_not_norm.alphaArrs[0]

    diff_kde = tops_kde - chip_kde

    rastArr = topsScn_not_norm.rastArrs[0]
    bgrArr = np.where(diff_kde > 0, TOPS_COLOR, CHIP_COLOR)

    diff_kde_plus, diff_kde_minus = diff_kde.clip(min = 0), -diff_kde.clip(max = 0)

    diff_alpha_plus = diff_kde_plus / diff_kde_plus.max()
    diff_alpha_minus = diff_kde_minus / diff_kde_minus.max()

    alphaArr = np.where(diff_kde > 0, diff_alpha_plus, diff_alpha_minus)

    diffScn = DummyScene([rastArr], [bgrArr], [alphaArr])

    script_line =   """
                        We construct a difference heatmap that highlights the areas 
                        in which choice chip and topspin returns are more prevalent in their respective colors.

                        Chipped returns are overrepresented at the corners of the court and the shallower areas along the center service line,
                        whereas topspin returns are relatively most frequent in the center of the court.
                    """
    myFrms, animDict = vs.animate_static(diffScn, background_scene = courtScn, static_cam = heatCam, **prmDict, annotated_points = [fhPt, bhPt])
    myFrm = overlay(myFrms[-1], legFrm, top_left_point = anchorPt)
    myFrms = BGRFrames.from_frame(myFrm, STATIC_FRAMES)
    # myFrms[-1].show() ### BEAUTIFUL!
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line =   """
                        Let us now look at a different facet of return characteristics: launch angle.
                    """
    staticFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
    all_frames.append(staticFrms)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)
    
    current_idx = current_idx + len(all_frames)


#########################
#### LAUNCH ANGLE #######
#########################

# make the lateral cam and intermediate cam over net
bPt = splnScn[0].leftPt ### origin of the shot
aPt = bPt.left(100)
sideretCam = Camera.from_to(aPt, bPt, f = 1000)

from ATMANpy.auxil.court import cnt_2D
cnt = Point3D.from_point2D(cnt_2D, z = 100)
intmCam = Camera.from_to(cnt, cnt.right(100).down(20))

# a useful null line for interpolation
from ATMANpy.types.Shapes3D import Line3D
nullLn = Line3D(bPt, bPt, color = BALL_COLOR)
nullScn = Scene([nullLn])

# player shape
_, player = myShots[1].feat_top_player().first()

# get topspin and chip dists, with axes
kwargDict = {'bandwidth': 2,
             'resolution': 100,
             'axis_length': 40,
             'axis_origin_point': bPt,
             'x_unit_vector': Point3D((0, -1, 0)), 
             'y_unit_vector': Point3D((0, 0, 1)),
             'normalize_density': False, 
             'stretch_density': 4.5}

weightArr = regFrm_in.loc[regFrm_in.is_chipped == 1, 'weight_topspin'].values
weightArr_preds = 1 - regFrm_in.loc[regFrm_in.is_chipped == 1, 'pred_is_chipped'].values

tops_angleScn, tops_modes = vs.kde_angle_distribution(tops_retShots_rectified, **kwargDict, color = TOPS_COLOR)
chip_angleScn, chip_modes = vs.kde_angle_distribution(chip_retShots_rectified, **kwargDict, weight_func_or_arr = weightArr_preds, color = CHIP_COLOR)

### make axes
xLn, yLn = tops_angleScn.shapes[1], tops_angleScn.shapes[2]
# lengthen xLn 
xLn = Line3D(xLn.leftPt, xLn.rightPt.front(22))
axScn = Scene([xLn, yLn])
court_and_axScn = courtScn.combine(axScn)

xLn_null = Line3D(xLn.leftPt, xLn.leftPt)
yLn_null = Line3D(yLn.leftPt, yLn.leftPt)
axScn_null = Scene([xLn_null, yLn_null])

axPnts = [xLn.center_point().down(2).left(12).annotate('Horizontal Launch Speed'), 
          yLn.center_point().back(2).annotate('Vertical Launch Speed')]
axPnts = [] ### not needed because confusing when density is plotted ?!

### drop axes from scene -> we will render them in the background, with court_and_axScn
tops_angleScn = tops_angleScn.drop(2).drop(1)
chip_angleScn = chip_angleScn.drop(2).drop(1)

### get modes
# a) chip
chip_mode_lo, chip_mode_hi = chip_modes[[2, 10]]
mode_angles = [chip_mode_lo, chip_mode_hi]
modeLns = vis_ut.make_mode_lines(mode_angles, line_len = kwargDict['axis_length'] * 1.5, 
                                 x_ax = kwargDict['x_unit_vector'], y_ax = kwargDict['y_unit_vector'],
                                 orig = kwargDict['axis_origin_point'])
modeScn0 = Scene([modeLns[0].recolor(CHIP_COLOR)]).dim(1.2)
modeScn1 = Scene([modeLns[1].recolor(CHIP_COLOR)]).dim(1.2)

itpl_lns0 = vs.interpolate_scenes(nullScn, modeScn0, num_frames = 25)
itpl_lns1 = vs.interpolate_scenes(nullScn, modeScn1, num_frames = 25)

# b) topspin
tops_mode = tops_modes[1]
mode_angles = [tops_mode]
modeLns = vis_ut.make_mode_lines(mode_angles, line_len = kwargDict['axis_length'] * 1.5,
                                 x_ax = kwargDict['x_unit_vector'], y_ax = kwargDict['y_unit_vector'],
                                 orig = kwargDict['axis_origin_point'])
modeScn2 = Scene([modeLns[0].recolor(TOPS_COLOR)]).dim(1.2)

itpl_lns2 = vs.interpolate_scenes(nullScn, modeScn2, num_frames = 25)

### build a distribution of launch angles on choice chips.
sorted_chip_retShots_rectified = chip_retShots_rectified.sort(weightArr_preds, descending = True)

if DO_LAUNCH_ANGLE_INTRO:
    all_frames = []
    script_lines = []

    # fade trajectory back in
    script_line = ""
    itplScns = vs.interpolate_scenes(splnScn.realpha(0), splnScn, num_frames = 25)
    myFrms, animDict = vs.animate_sequence(itplScns, start_cam = heatCam, stop_cam = heatCam, background_scene = courtScn, 
                                        **prmDict_nolum, **animDict, annotated_points = [fhPt, bhPt], cumulative_foreground_transparence = 0.9)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    # cam motion -> made so that the motion shows also the entire trajectory!
    script_line = ""
    myFrms1, animDict = vs.animate_cam_motion(splnScn, num_frames = 50,  annotated_points = [fhPt, bhPt],
                                            background_scene = courtScn, start_cam = heatCam, stop_cam = intmCam, **prmDict_nolum)

    myFrms2, animDict =  vs.animate_cam_motion(splnScn, num_frames = 50,  annotated_points = [fhPt, bhPt],
                                            background_scene = courtScn, start_cam = intmCam, stop_cam = sideretCam, **prmDict_nolum)
    myFrms = BGRFrames.from_frames([myFrms1, myFrms2])
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ### TODO: Use null line to animate axes appearing
    script_line = ""
    itplScns = vs.interpolate_scenes(axScn_null, axScn, TRANSITION_FRAMES)
    myFrms, animDict = vs.animate_sequence(itplScns, start_cam = sideretCam, stop_cam = sideretCam, background_scene = courtScn, **animDict, **prmDict_nolum)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    myFrms, animDict = vs.animate_static(splnScn, static_cam = sideretCam, background_scene = court_and_axScn, annotated_points = axPnts, **prmDict_nolum)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ang = myShots[1].get_launch_angle()
    limit_theta = ((180 - ang) * const.ANGLE_TO_RAD, 180 * const.ANGLE_TO_RAD)
    degCrc = Circle3D.from_point3D(bPt, fixed_dimension = 'x', radius = 25, limit_theta = limit_theta)

    myFrms, animDict = vs.animate_static(splnScn.put(degCrc), static_cam = sideretCam, background_scene = court_and_axScn, annotated_points = axPnts,
                                         **prmDict_nolum)
    # all_frames.append(myFrms)
    # script_lines.append(script_line)

    anglePt = bPt.front(24).up(1).annotate(f'{ang:.1f} Degrees')

    script_line =   """
                        The example return is chipped back at an angle of 11.9 degrees. 
                        Let's see how launch angles are distributed for choice chipped returns in general. 
                    """
    myFrms, animDict = vs.animate_static(splnScn.put(degCrc), static_cam = sideretCam, background_scene = court_and_axScn, annotated_points = [anglePt, *axPnts],
                                         **prmDict_nolum)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    show(all_frames)

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

    acceleration_rate = 0.2
    initial_frames = 20
    splnScn_curr = splnScn
    prev_kde = None
    shotList = []
    weightList = []
    frms = []
    idx = 0
    for shot, weight in zip(sorted_chip_retShots_rectified[:], weightArr_preds[:]):
        if not 0 < shot.get_launch_angle() <= 75:
            continue

        ### calculate num_frames
        num_frames = max(1, int(initial_frames * (1 - acceleration_rate) ** idx))

        ### build new kde
        shotList.append(shot)
        weightList.append(weight)
        kde, _ = vs.kde_angle_distribution(Shots(shotList), **kwargDict, weight_func_or_arr = np.array(weightList), 
                                        color = CHIP_COLOR)
        kde = Scene([kde[0]])

        ### if necessary: create previous KDE
        if prev_kde is None:
            els = len(kde[0].coords[0])
            x, y, z = bPt.coords
            prev_pt = Point3D((np.full(els, fill_value = x), np.full(els, fill_value = y), np.full(els, fill_value = z)), color = CHIP_COLOR)
            prev_kde = Scene([prev_pt])
        
        ### Animate the current trajectory
        anchTraj = shot.shotTraj.reanchor(bPt)
        _, splines_current = anchTraj.feat_spline_cumul().last()
        splnScn_curr = Scene([spLn.recolor(BALL_COLOR) for spLn in splines_current if spLn is not None])

        myFrms, _ = vs.animate_static(splnScn_curr, background_scene = court_and_axScn, previous_foreground_scene = prev_kde, **prmDict,
                                    static_cam = sideretCam, num_frames = num_frames)
        frms.append(myFrms)

        ### Animate the new KDE
        if num_frames > 1:
            itpl_kdes = vs.interpolate_scenes(prev_kde, kde, num_frames)
        else:
            itpl_kdes = Scene([kde])
        myFrms, _ = vs.animate_sequence(itpl_kdes, background_scene = court_and_axScn, previous_foreground_scene = splnScn_curr, **prmDict,
                                        start_cam = sideretCam, stop_cam = sideretCam, cumulative_foreground_transparence = 1)
        frms.append(myFrms)

        prev_kde = kde
        idx += 1

    script_line = ""
    jntFrms = BGRFrames.from_frames(frms)
    jntFrms = overlay(jntFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(jntFrms)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    show(all_frames)

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
                        he really opts for either a low, shallow-bouncing chip, or a high-arcing chip. 
                    """
    myFrms, animDict = vs.animate_static(chip_angleScn, background_scene = court_and_axScn, **prmDict_nolum,
                                    static_cam = sideretCam, num_frames = STATIC_FRAMES)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    myFrms, animDict = vs.animate_sequence(itpl_lns0, background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    myFrms, animDict = vs.animate_sequence(itpl_lns1, background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line =   """
                        On the other hand, for topspin returns, the distribution is more uniform: we observe only a single mode.
                    """
    staticFrms = BGRFrames.from_frame(myFrms[-1], STATIC_FRAMES)
    staticFrms = overlay(staticFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(staticFrms)
    script_lines.append(script_line)

    # b) Topspin  
    #  --> One Mode: low angle topspin
    script_line = ""
    animDict['previous_foreground_scene'] = animDict['previous_foreground_scene'].drop(0)
    itpl_angleScns = vs.interpolate_scenes(chip_angleScn, tops_angleScn, 25)
    myFrms, animDict = vs.animate_sequence(itpl_angleScns, background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line =   """
                        The topspin mode angle is lower than both chip return modes, 
                        which is why we differentiate between a slice and a chip: 
                        a typical rally slice would depart at a launch angle lower than, or similar to, a typical topspin shot.
                    """
    myFrms, animDict = vs.animate_static(tops_angleScn, background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    static_cam = sideretCam, num_frames = STATIC_FRAMES)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    myFrms, animDict = vs.animate_sequence(itpl_lns2, background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    start_cam = sideretCam, stop_cam = sideretCam)
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = ""
    statFrms = BGRFrames.from_frame(myFrms[-1], TRANSITION_FRAMES)
    statFrms = overlay(statFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(statFrms)
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
    myFrms, animDict = vs.animate_static(Scene([degCrc0]), background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    static_cam = sideretCam, num_frames = STATIC_FRAMES, annotated_points = [angPt0])
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = f"... the low-angle choice chip return at {chip_mode_lo:.1f} Degrees..."
    myFrms, animDict = vs.animate_static(Scene([degCrc0, degCrc1]), background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    static_cam = sideretCam, num_frames = STATIC_FRAMES, annotated_points = [angPt0, angPt1])
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    script_line = f"... and the high_angle choice chip return at {chip_mode_hi:.1f} Degrees."
    myFrms, animDict = vs.animate_static(Scene([degCrc0, degCrc1, degCrc2]), background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    static_cam = sideretCam, num_frames = STATIC_FRAMES, annotated_points = [angPt0, angPt1, angPt2])
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ang = myShots[1].get_launch_angle()
    script_line =   f"""
                        Note that our example chip return left the racket at a launch angle of {ang:.1f} degrees, 
                        making it a good representative of the low-angle chip mode. 
                        Let's watch it again, in this light.
                    """
    myFrms, animDict = vs.animate_static(Scene([degCrc0, degCrc1, degCrc2]), background_scene = court_and_axScn, **prmDict_nolum, **animDict,
                                    static_cam = sideretCam, num_frames = STATIC_FRAMES, annotated_points = [angPt0, angPt1, angPt2])
    myFrms = overlay(myFrms, legFrm, top_left_point = anchorPt)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    print(ang)
    print(chip_mode_lo) ### original shot very close to lower mode!

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)

    show(all_frames)

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

    ### Compare points in Raster, with Tables.
    # kill diagram
    script_line = ""
    myFrms, _ = vs.animate_static(None, background_scene = courtScn, **prmDict_nolum, static_cam = sideretCam, num_frames = 10)
    all_frames.append(myFrms)
    script_lines.append(script_line)

    # prep scaled camera view and frames
    scale_factor = 1.5
    brestCam_resized_half = brestCam.resize(scale_factor)

    origStm = myShots.stream(confs = confs)
    origDtm = origStm.decap()
    orig_scaledFrms = myDtm.frames.rescale(scale_factor)

    bestStm = bestShots.stream(confs = confs)
    bestDtm = bestStm.decap()
    best_scaledFrms = bestDtm.frames.rescale(scale_factor)

    frames = origStm.frmNums
    new_frames = list(range(min(frames) - STATIC_FRAMES, min(frames)))
    early_origStm = origStm.conform(new_frames)
    early_origDtm = early_origStm.decap()
    early_orig_scaledFrms = early_origDtm.frames.rescale(scale_factor)

    frames = bestStm.frmNums
    new_frames = list(range(min(frames) - STATIC_FRAMES, min(frames)))
    early_bestStm = bestStm.conform(new_frames)
    early_bestDtm = early_bestStm.decap()
    early_best_scaledFrms = early_bestDtm.frames.rescale(scale_factor)

    # zoom out of current view, back to the brestCam (already in rasterized view)
    script_line = ""
    myFrms, _ = vs.animate_cam_motion(None, background_scene = courtScn, **prmDict_nolum,
                                    start_cam = sideretCam, stop_cam = brestCam_resized_half,
                                    num_frames = 100)
    lsFrm = myFrms[-1]
    all_frames.append(myFrms)
    script_lines.append(script_line)

    ### experiment: fade-in with movement
    lsFrm_resc = View.make(brestCam_resized_half, courtScn).snap()
    blankFrm_resc = View.make(brestCam_resized_half, Scene([])).snap()

    def mean_two_frames(aFrm, bFrm, my_lambda):
        aArr, bArr = aFrm.array.astype(float), bFrm.array.astype(float)
        itplArr = aArr * (1 - my_lambda) + bArr * my_lambda
        return itplArr.astype(np.uint8)

    orig_itplArrs = np.stack([mean_two_frames(lsFrm_resc, f, idx / STATIC_FRAMES) for idx, f in enumerate(early_orig_scaledFrms)], axis = 0)
    orig_itplFrms = BGRFrames(orig_itplArrs)

    best_itplArrs = np.stack([mean_two_frames(blankFrm_resc, f, idx / STATIC_FRAMES) for idx, f in enumerate(early_best_scaledFrms)], axis = 0)
    best_itplFrms = BGRFrames(best_itplArrs)

    # create spline animation for our original point on new bestCam_resized_half
    splineFrms_orig, _ = vs.animate_sequence(spline_scenes, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                        cumulate_argument = False, animate_over = orig_scaledFrms, **prmDict)
    # DecapStream(myFrms, 0, 100).show()

    # create spline animation for our best point on new brestCam_resized_half
    serveSF = bestShots[0].feat_spline_cumul(which_endpoint = 'penultimate')
    returnSF = bestShots[1].feat_spline_cumul(which_endpoint = 'penultimate')

    serve_scenes = [Scene([spLn.recolor(BALL_COLOR) for spLn in splines if spLn is not None]) for splines in serveSF.values()]
    return_scenes = [Scene([spLn.recolor(BALL_COLOR) for spLn in splines if spLn is not None]) for splines in returnSF.values()]
    best_spline_scenes = [*serve_scenes, *return_scenes]

    splineFrms_best, _ = vs.animate_sequence(best_spline_scenes, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                        cumulate_argument = False, animate_over = best_scaledFrms, **prmDict)
    # DecapStream(splineFrms_best, 0, 100).show(ms_between_frames = 100)
    # myFrms[-1].show()

    # rasterize frames, and fade in the point being played
    script_line = ""
    fadeinRst = FramesRaster(2, 2)
    fadeinRst.add(orig_itplFrms, 0, 0)
    fadeinFrms = fadeinRst.render()
    all_frames.append(fadeinFrms)
    script_lines.append(script_line)
    
    # play out point with spline
    script_line = ""
    myRst = FramesRaster(2, 2)
    myRst.add(splineFrms_orig, 0, 0)
    playFrms_orig = myRst.render()
    all_frames.append(playFrms_orig)
    script_lines.append(script_line)

    # freeze, and play best matching point in lower right
    script_line =   """
                        We now compare the low chip to a representative of high-angle chip returns, in the lower right.
                    """
    staticFrms = BGRFrames.from_frame(playFrms_orig[-1], STATIC_FRAMES)
    all_frames.append(staticFrms)
    script_lines.append(script_line)

    # fade in best comparison in lower right
    script_line = ""
    fadeinRst = FramesRaster(2, 2)
    fadeinRst.add(splineFrms_orig[-1], 0, 0)
    fadeinRst.add(best_itplFrms, 1, 1)
    fadeinFrms = fadeinRst.render()
    all_frames.append(fadeinFrms)
    script_lines.append(script_line)

    # play out point with spline
    script_line = ""
    fadeinRst = FramesRaster(2, 2)
    fadeinRst.add(splineFrms_orig[-1], 0, 0)
    fadeinRst.add(splineFrms_best, 1, 1)
    fadeinFrms = fadeinRst.render()
    all_frames.append(fadeinFrms)
    script_lines.append(script_line)

    # Matching Table
    lo_name = 'Low Launch Angle'
    hi_name = 'High Launch Angle'

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
    myTab = Table(table_data, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.25, 
                  title = 'Initial Conditions', title_horizontal_offset = 11,
                  header_text_size = 1, index_text_size = 1, data_text_size = 1, 
                  skip_after_how_many_symbols = None)
    tabFrm_match = myTab.render()
    # tabFrm_match.show()

    script_line =   """
                        The return-of-comparison is chosen to be maximally similar to the low-angle chip in all respects except the launch angle,
                        so that we can cleanly see the effects of a higher launch angle on ball trajectory and bounce. 
                        As the table shows, both returns are made at almost the same speed, spin, and height.
                        We therefore isolate the effect of launch angle on trajectory characteristics, which we will look at now.
                    """
    myRst = FrameRaster(2, 2)
    myRst.add(splineFrms_orig[-1], 0, 0)
    myRst.add(splineFrms_best[-1], 1, 1)
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
    
    table_data = {  'Apex':                     {lo_name: f'{lo_apx:.1f} m',        hi_name: f'{hi_apx:.1f} m'},
                    'Apex post-bounce':         {lo_name: f'{lo_apx_pst:.1f} m',    hi_name: f'{hi_apx_pst:.1f} m'},
                    'Ground Speed post-bounce': {lo_name: f'{lo_hori:.1f} km/h',    hi_name: f'{hi_hori:.1f} km/h'},
                    'Time until bounce':        {lo_name: f'{lo_secs:.2f} sec',     hi_name: f'{hi_secs:.2f} sec'},}
    
    dims = (splineFrms_orig.cols(), splineFrms_orig.rows())
    myTab = Table(table_data, dimensions = dims, skeleton_shrinkage = 0.2, index_offset = 0.1, 
                  title = 'Trajectory Comparison', title_horizontal_offset = 10,
                  header_text_size = 1, index_text_size = 1, data_text_size = 1, 
                  skip_after_how_many_symbols = None)
    tabFrm_comp = myTab.render()
    # tabFrm_comp.show()

    script_line =   """
                        The high-arcing chip bounces 70 cm higher than the low chip. 
                        If the arcing chip does not bounce close to the server's baseline, its high bounce can allow the server to attack with a flat shot at aggressive angles,
                        whereas a low-angle chip that bounces shallow in the court is not so exploitable: it stays low and is awkward to attack because it has to be played with aggressive topspin
                        or lower margins over the net to dip back down into the returner's half.
                        On the other hand, the high chip loses more of its speed along the ground, which means that the server has to generate all the power if he wants to attack;
                        in this sense, a high-arcing chip can neutralize a serve if hit with sufficient depth.
                        The arcing chip also gives the returner more time to recuperate, which may be especially useful if the serve is hit out wide and carries the returner off the court.
                        With respect to this latter point, let's visualize the impact of a ball's time in flight on the receiving player's ability to reposition on the court.
                    """
    myRst = FrameRaster(2, 2)
    myRst.add(splineFrms_orig[-1], 0, 0)
    myRst.add(splineFrms_best[-1], 1, 1)
    myRst.add(tabFrm_match, 0, 1)
    myRst.add(tabFrm_comp, 1, 0)
    tabFrms_match = BGRFrames.from_frame(myRst.render(), STATIC_FRAMES)
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

    ### Kill Tables in FrameRaster
    script_line = ""
    myRst = FrameRaster(2, 2)
    myRst.add(splineFrms_orig[-1], 0, 0)
    myRst.add(splineFrms_best[-1], 1, 1)
    myRst.add(tabFrm_match, 0, 1)
    killFrms_comp = BGRFrames.from_frame(myRst.render(), 25)
    all_frames.append(killFrms_comp)
    script_lines.append(script_line)
    
    script_line =   """
                        First, consider the original example in the top left...
                    """
    myRst = FrameRaster(2, 2)
    myRst.add(splineFrms_orig[-1], 0, 0)
    myRst.add(splineFrms_best[-1], 1, 1)
    killFrms_match = BGRFrames.from_frame(myRst.render(), 25)
    all_frames.append(killFrms_match)
    script_lines.append(script_line)

    ### Reverse return, but keep the spline.
    revFrms_orig = myShots[1].stream(confs = confs).decap().frames.rescale(scale_factor).slow(2).reverse()
    revFrms_best = bestShots[1].stream(confs = confs).decap().frames.rescale(scale_factor).slow(2).reverse()

    len_orig = revFrms_orig.length()
    len_best = revFrms_best.length()
    # fade the trajectory out, and the mark in.
    last_spline_scenes_orig = [spline_scenes[-1].transparify(1 - ((i + 1) / len_orig)) for i in range(len_orig)]
    last_spline_scenes_best = [best_spline_scenes[-1].transparify(1 - ((i + 1) / len_best)) for i in range(len_best)]

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

    splinerevFrms_orig, _ = vs.animate_sequence(last_spline_scenes_orig, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                                cumulate_argument = False, animate_over = revFrms_orig, previous_foreground_scene = Scene([origMark]),
                                                **prmDict)
    splinerevFrms_best, _ = vs.animate_sequence(last_spline_scenes_best, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, background_scene = None, 
                                                cumulate_argument = False, animate_over = revFrms_best, previous_foreground_scene = Scene([bestMark]),
                                                **prmDict)

    script_line = ""
    myRst = FramesRaster(2, 2)
    myRst.add(splinerevFrms_orig, 0, 0)
    myRst.add(splineFrms_best[-1], 1, 1)
    myFrms_rev_orig = myRst.render()
    all_frames.append(myFrms_rev_orig)
    script_lines.append(script_line)

    ### add in the position mark
    script_line =   """
                        Notice first that the return down-the-line requires the returner to recover to the opposite side of the court, 
                        roughly to the red X.
                    """
    staticFrms_orig, _ = vs.animate_static(origMark_repos, static_cam = brestCam_resized_half, background_scene = None, animate_over = revFrms_orig[-1], 
                                            previous_foreground_scene = Scene([origMark]))
    
    myRst = FramesRaster(2, 2)
    myRst.add(staticFrms_orig, 0, 0)
    myRst.add(splineFrms_best[-1], 1, 1)
    staticFrms_orig = myRst.render()
    all_frames.append(staticFrms_orig)
    script_lines.append(script_line)

    ### Show distance circles
    last_origFrm = revFrms_orig[-1]
    last_bestFrm = revFrms_best[-1]

    lo_dist, hi_dist, tp_dist = time_dist_frame.loc['orig_lo', 'dist_pix'], time_dist_frame.loc['orig_hi', 'dist_pix'], time_dist_frame.loc['orig_tp', 'dist_pix']
    
    def make_thick_circle(x, y, radius, theta_lo, theta_hi):
        cntPt = Point3D((x, y, 0))
        radii = [0.995 * radius, radius, 1.005 * radius]
        return Circles3D([Circle3D.from_point3D(cntPt, limit_theta = (theta_lo * np.pi, theta_hi * np.pi), radius = r) for r in radii])

    # a) for the original shot
    x, y = myShots[1].hitPlayer.get_initial_position()

    loCrcs1 = make_thick_circle(x, y, lo_dist, 0, 0.2)
    loCrcs2 = make_thick_circle(x, y, lo_dist, 0.8, 2)
    loCrcs = loCrcs1.join(loCrcs2).recolor(CHIP_COLOR)

    hiCrcs1 = make_thick_circle(x, y, hi_dist, 0, 0.165)
    hiCrcs2 = make_thick_circle(x, y, hi_dist, 0.84, 2)
    hiCrcs = hiCrcs1.join(hiCrcs2).recolor(CHIP_COLOR)

    tpCrcs = make_thick_circle(x, y, tp_dist, 0, 2).recolor(TOPS_COLOR)

    script_line =   """
                        The low-angled chip shot gives him enough time to recover to the optimal spot. 
                        The range of motion implied by the return's time until bounce is indicated in blue.
                        It assumes a third of a second of recovery time after the return has been made, and a running speed of 15 km/h,
                        which is roughly in line with the ATP average for short distance sprints from a standing start.
                    """
    myFrms, _ = vs.animate_static(Scene([loCrcs]), static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1), 
                                  previous_foreground_scene = Scene([origMark, origMark_repos]),
                                  **prmDict)
    myRst = FramesRaster(2, 2)
    myRst.add(myFrms, 0, 0)
    myRst.add(splineFrms_best[-1], 1, 1)
    crcFrms_orig = myRst.render()
    all_frames.append(crcFrms_orig)
    script_lines.append(script_line)

    script_line =   """
                        Had he played a high angle return, the additional .25 seconds of time-until-bounce would have given him more range of motion, 
                        which, in this case, would not have been necessary to reach the optimal point.
                    """
    myFrms, _ = vs.animate_static(Scene([loCrcs, hiCrcs]), static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                  previous_foreground_scene = Scene([origMark, origMark_repos]),
                                  **prmDict)
    myRst.add(myFrms, 0, 0)
    crcFrms_orig = myRst.render()
    all_frames.append(crcFrms_orig)
    script_lines.append(script_line)

    script_line =   f"""
                        A comparable topspin shot down the line would travel {time_dist_frame.loc['orig_tp', 'time']:.1f} seconds, 
                        which does not afford the returning player with enough time to recover to the optimal spot. 
                        In this sense, the topspin return down-the-line is a forcing shot: it must be so good as to end the rally
                        on the spot, or at least force a weak ball or even an error from the serving player; 
                        otherwise, the returner will have opened himself up to a low-margin serve + 1 cross-court.

                        Now, consider the situation in the lower right corner. 
                    """
    staticFrms_orig, _ = vs.animate_static(Scene([loCrcs, hiCrcs, tpCrcs]), static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                  previous_foreground_scene = Scene([origMark, origMark_repos]),
                                  **prmDict)
    myRst.add(staticFrms_orig, 0, 0)
    crcFrms_orig = myRst.render()
    all_frames.append(crcFrms_orig)
    script_lines.append(script_line)

    orig_markScn = Scene([origMark, origMark_repos]) ### save for later
    orig_circScn = Scene([loCrcs, hiCrcs, tpCrcs]) ### save for later
    ls_staticFrm_orig = staticFrms_orig[-1]

    # b) for the best match shot
    x, y = bestShots[1].hitPlayer.get_initial_position()

    lo_dist, hi_dist, tp_dist = time_dist_frame.loc['best_lo', 'dist_pix'], time_dist_frame.loc['best_hi', 'dist_pix'], time_dist_frame.loc['best_tp', 'dist_pix']

    loCrcs1 = make_thick_circle(x, y, lo_dist, 0, 0.26)
    loCrcs2 = make_thick_circle(x, y, lo_dist, 0.74, 2)
    loCrcs = loCrcs1.join(loCrcs2).recolor(CHIP_COLOR)

    hiCrcs = make_thick_circle(x, y, hi_dist, 0.8, 1.815).recolor(CHIP_COLOR)
    tpCrcs = make_thick_circle(x, y, tp_dist, 0, 2).recolor(TOPS_COLOR)
    

    script_line = ""
    myRst = FramesRaster(2, 2)
    myRst.add(ls_staticFrm_orig, 0, 0)
    myRst.add(splinerevFrms_best, 1, 1)
    myFrms_rev_best = myRst.render()
    all_frames.append(myFrms_rev_best)
    script_lines.append(script_line)

    ### add in the position mark
    script_line =   """
                        Since the return goes cross-court, the returner is not required to recover to the other half of the baseline, 
                        so he has less distance to cover.
                    """
    staticFrms_best, _ = vs.animate_static(bestMark_repos, static_cam = brestCam_resized_half, background_scene = None, animate_over = last_bestFrm.dim(1), 
                                            previous_foreground_scene = Scene([bestMark]), **prmDict)

    myRst = FramesRaster(2, 2)
    myRst.add(ls_staticFrm_orig, 0, 0)
    myRst.add(staticFrms_best, 1, 1)
    staticFrms = myRst.render()
    all_frames.append(staticFrms)
    script_lines.append(script_line)

    orig_markScn = Scene([origMark, origMark_repos])
    best_markScn = Scene([bestMark, bestMark_repos])

    script_line =   f"""
                        Again, the low-angled chip's flight time would have been more than enough to recover to the optimal spot.
                        Even the topspin return cross-court would have afforded the returning player with enough time to make it to an adequate position on court; 
                        he can not quite reach the optimal position, but he is not far away.

                        However, notice that the returner is pulled out wide for this serve. 
                        If, in this situation, he were to play a return down-the-line, the returner would have more ground to cover to the optimal spot. 
                        At the same time, the reduced flight-time of a return down the line affords the returner with less time to make up that distance.
                    """
    staticFrms, _ = vs.animate_static(Scene([loCrcs, hiCrcs, tpCrcs]), static_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1),
                                  previous_foreground_scene = best_markScn,
                                  **prmDict)
    myRst.add(staticFrms, 1, 1)
    crcFrms_best = myRst.render()
    all_frames.append(crcFrms_best)
    script_lines.append(script_line)

    script_line = ""
    
    ## interpolate marks and circles
    lo_dist, hi_dist, tp_dist = time_dist_frame.loc['best_lo', 'dist_pix'], time_dist_frame.loc['best_hi', 'dist_pix'], time_dist_frame.loc['best_tp', 'dist_pix']
    loCrcs1 = make_thick_circle(x, y, lo_dist, 0, 0.26)
    loCrcs2 = make_thick_circle(x, y, lo_dist, 0.74, 2)
    loCrcs_best = loCrcs1.join(loCrcs2).recolor(CHIP_COLOR)

    hiCrcs_best = make_thick_circle(x, y, hi_dist, 0.8, 1.815).recolor(CHIP_COLOR)
    tpCrcs_best = make_thick_circle(x, y, tp_dist, 0, 2).recolor(TOPS_COLOR)
    
    lo_dist, hi_dist, tp_dist = time_dist_frame.loc['cntr_lo', 'dist_pix'], time_dist_frame.loc['cntr_hi', 'dist_pix'], time_dist_frame.loc['cntr_tp', 'dist_pix']
    
    loCrcs1 = make_thick_circle(x, y, lo_dist, 0, 0.24)
    loCrcs2 = make_thick_circle(x, y, lo_dist, 0.72, 2)
    loCrcs_cntr = loCrcs1.join(loCrcs2).recolor(CHIP_COLOR)
    
    hiCrcs_cntr = make_thick_circle(x, y, hi_dist, 0.78, 1.85).recolor(CHIP_COLOR)
    tpCrcs_cntr = make_thick_circle(x, y, tp_dist, 0, 2).recolor(TOPS_COLOR)

    crcScn0 = Scene([hiCrcs_best, loCrcs_best, tpCrcs_best])
    crcScn1 = Scene([hiCrcs_cntr, loCrcs_cntr, tpCrcs_cntr])

    startScn, endScn = best_markScn.combine(crcScn0), orig_markScn.combine(crcScn1)

    itplScns = vs.interpolate_scenes(startScn, endScn, 50)

    itplFrms, _ = vs.animate_sequence(itplScns, start_cam = brestCam_resized_half, stop_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1),
                                                **prmDict)
    myRst.add(itplFrms, 1, 1)
    crcFrms_cntr = myRst.render()
    all_frames.append(crcFrms_cntr)
    script_lines.append(script_line)

    script_line =   """
                        In this situation, where the returner is pulled somewhat out wide and attempts a return down the line, 
                        only the high-arcing chip buys him enough time to get into the optimal position at the time of bounce.
                    """
    staticFrms = BGRFrames.from_frame(crcFrms_cntr[-1], STATIC_FRAMES)
    all_frames.append(staticFrms)
    script_lines.append(script_line)

    ### SEGUE TO INFERENCE
    # given all of this -> where SHOULD a chipped ball go to improve the returners chances of winning?
    # -> what chip is more promising than an average topspin return?

    # de-render the shapes in the top left
    script_line =   """
                        Up until this point, we have tackled the chip return entirely descriptively, 
                        having made note of the distributions of placement and launch angles, 
                        and their impact on court positioning. 
                        We know, roughly, what an average challenger-level player does 
                        with respect to the chipped return versus the topspin return,
                        what we are lacking is an answer to the question: what should he do.
                        Where should the chip return be placed to overcome the 10% win-rate deficit over the topspin return?
                        At what angle should the ball be played? 

                        We will now attempt an answer to these questions, with the following caveat: To the extent that
                        these normative kinds of questions are the more interesting to answer, they are also more difficult to answer
                        statistically, for the lack of a counterfactual. 
                        In any specific situation, we only ever observe the return that was actually hit,
                        not the set of all other possible returns. Had we observed that set and had we recorded the outcome 
                        of the point for each of the possible returns, we could easily give an answer to what the returner should do,
                        especially if we had observed the whole set a bunch of times over to average out the randomness that 
                        a tennis point is heir to.
                        However, we do not inhabit the world of hypotheticals, and must make do with just the one outcome. 
                        
                        Though imperfect, an adequate attempt can still be made to answer the question: what should the returner do, here.

                        To this end, we estimate a model that takes into account a variety of factors, 
                        such as return placement, launch angle, speed, spin and on-court positioning of the returner and server
                        and calculates the returner's win probability. 
                        We can then let the model evaluate all kinds of hypothetical returns, and let it calculate their win probability.
                        For a few different example situations, the best chip and topspin returns emerge naturally as the best
                        hypothetical return among all.
                    """
    myRst = FramesRaster(2, 2)
    myRst.add(itplFrms[-1], 1, 1)

    num_frames = 10
    staticFrms_derender, _ = vs.animate_static(orig_circScn, static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                        previous_foreground_scene = orig_markScn, num_frames = num_frames,
                                        **prmDict)
    myRst.add(staticFrms_derender, 0, 0)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    script_line = ""
    staticFrms_derender, _ = vs.animate_static(orig_circScn, static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                        previous_foreground_scene = orig_markScn.drop(0), num_frames = num_frames,
                                        **prmDict)
    myRst.add(staticFrms_derender, 0, 0)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)
    
    script_line = ""
    staticFrms_derender, _ = vs.animate_static(orig_circScn, static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                        previous_foreground_scene = None, num_frames = num_frames,
                                        **prmDict)
    myRst.add(staticFrms_derender, 0, 0)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    script_line = ""
    staticFrms_derender, _ = vs.animate_static(orig_circScn.drop(1), static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                        previous_foreground_scene = None, num_frames = num_frames,
                                        **prmDict)
    myRst.add(staticFrms_derender, 0, 0)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)
    
    script_line = ""
    staticFrms_derender, _ = vs.animate_static(orig_circScn.drop(1).drop(0), static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                        previous_foreground_scene = None, num_frames = num_frames,
                                        **prmDict)
    myRst.add(staticFrms_derender, 0, 0)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    script_line = ""
    staticFrms_derender, _ = vs.animate_static(Scene([]), static_cam = brestCam_resized_half, animate_over = last_origFrm.dim(1),
                                        previous_foreground_scene = None, num_frames = num_frames,
                                        **prmDict)
    myRst.add(staticFrms_derender, 0, 0)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    ### derender also other frame
    myRst = FramesRaster(2, 2)
    
    script_line = ""
    staticFrms_derender, _ = vs.animate_static(endScn, static_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1), num_frames = num_frames, **prmDict)
    myRst.add(staticFrms_derender, 1, 1)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    script_line = ""
    staticFrms_derender, _ = vs.animate_static(endScn.drop(0), static_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1), num_frames = num_frames, **prmDict)
    myRst.add(staticFrms_derender, 1, 1)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)
    
    script_line = ""
    staticFrms_derender, _ = vs.animate_static(endScn.drop(0).drop(0), static_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1), num_frames = num_frames, **prmDict)
    myRst.add(staticFrms_derender, 1, 1)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    script_line = ""
    staticFrms_derender, _ = vs.animate_static(endScn.drop(0).drop(0).drop(0), static_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1), num_frames = num_frames, **prmDict)
    myRst.add(staticFrms_derender, 1, 1)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)
    
    script_line = ""
    staticFrms_derender, _ = vs.animate_static(endScn.drop(0).drop(0).drop(0).drop(0), static_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1), num_frames = num_frames, **prmDict)
    myRst.add(staticFrms_derender, 1, 1)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    script_line = ""
    staticFrms_derender, _ = vs.animate_static(Scene([]), static_cam = brestCam_resized_half, animate_over = last_bestFrm.dim(1), num_frames = num_frames, **prmDict)
    myRst.add(staticFrms_derender, 1, 1)
    crcFrms_derender = myRst.render()
    all_frames.append(crcFrms_derender)
    script_lines.append(script_line)

    ### Saving the separate MultiFrame objects and the Joint Script
    assert len(script_lines) == len(all_frames)
    show(all_frames)

    # b) Save segments
    for idx, frames_and_script in enumerate(zip(all_frames, script_lines)):
        subFrms, script_line = frames_and_script
        running_idx = current_idx + idx
        subFrms.save(fullPath, f'seg_{running_idx}', save_as = 'mp4', lossless = True)
        
        with open(os.path.join(fullPath, f'script_{running_idx}.txt'), mode = 'w') as f:
            f.write(script_line)

    current_idx = current_idx + len(all_frames)
