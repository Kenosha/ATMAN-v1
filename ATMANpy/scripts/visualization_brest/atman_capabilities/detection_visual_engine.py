"""
In this script, we want to animation possibilities within ATMAN

-> fully integrated 3D rendering engine to display results
-> fully integrated means: directly from the data that we gathered, within the same framework, we can
    -> display points as they happened, from different perspectives
    -> aggregate results with a variety of readily available visual aggregations, such as
        - heatmaps
        - graphs
        - tables
-> check out the two analyses to see it in action

technical things to show:
    -> camera can be turned, moved
    -> graphs can be blended in.
    -> point annotation (for scatter points? with line coming out of the ground.)
    
content to convey the technical things with:
    -> heatmaps are the entry point: show how they're maybe better to look at from right above.
    -> show arcs of shots from the side
    -> show a point from a different perspective

-> refer to visual analyses! show many more of the capabilities!

"""

import numpy as np
import pandas as pd
import sounddevice as sd
        
from ATMANpy.funcs.util_funcs import get_folder_struct, split
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames, MaskFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Visual3D import Camera, View, Scene

from ATMANpy.types.Abstract import Points, Shots, read_in_data_table, read_in_placeholder_table

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

from ATMANpy.models.ParticleFilter import ParticleFilter
from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points

from ATMANpy.types.Abstract import Points, Shots, read_in_data_tables, read_in_placeholder_tables

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
### CHOOSE VIDEO
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
for idx, vid in enumerate(avail_vids):
    print(idx, vid)

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
  
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

myCam = Camera.calibrated_brest_behind()

courtScn = Scene.make_court()

ball_color = const.BGR['ball_yellow']

### GET ABSTRACT POINT REPRESENTATION 
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)
allPnts = dataPnts.join(nullPnts).sort()

### FILTER
import ATMANpy.funcs.filter_shots_funcs as filt_s
import ATMANpy.funcs.filter_points_funcs as filt_p

filtPnts1 = dataPnts.filter(filt_p.is_returner, name = 'Brandon Nakashima').filter(filt_p.is_longer_than, threshold = 1)

filtPnts2 = filtPnts1.filter('is_first_serve')
decArr = vis_ut.get_winners_from_score(filtPnts2, allPnts) != 0
filtPnts2 = filtPnts2.filter(decArr)

filtPnts3 = filtPnts2.filter('is_serve_ad')

filtPnts4 = filtPnts3.filter(filt_p.is_serve_out_wide)

### GET RETURNS
retShots = filtPnts4.get_returns()
retShots_rect = retShots.enforce_direction('Top')

slc_retShots = retShots.filter('is_slice_visual')
top_retShots = retShots.filter('is_topspin_visual')

### fade in a point up until moment of return that has Brandon Nakashima on far side.
pt_idx = 9 ### this is from the detection_filtering script

retShots = filtPnts4.get_returns()
retShots_rect = retShots.enforce_direction('Top')

retShots_rect_out = retShots_rect.filter('is_fair', nextShot = None, negate = True)
retShots_rect_in = retShots_rect.filter('is_fair', nextShot = None)

### two regions -> opponents court, and behind the own baseline... a bit clumsy, but hey
from ATMANpy.types.Lines import CourtLines, Region
courtLines = CourtLines.make().add_singles_netline()
oppReg = courtLines.get_half_region(side = 'Top').scale(1.55)
ownReg_behind = Region.from_lines([courtLines['Bottom Baseline'], courtLines['Bottom Baseline'].down(300)])

in_res = 2
bh_res = 2.5
opp_res = 1.5

heatScn_in = vs.kde_heatmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'],
                            bandwidth = 60, resolution = in_res)
heatScn_out1 = vs.kde_heatmap(retShots_rect_out, one_color = const.BGR['intense_red'], zero_color = const.BGR['intense_red'],
                            bandwidth = 30, showReg = ownReg_behind, resolution = bh_res)
heatScn_out2 = vs.kde_heatmap(retShots_rect_out, one_color = const.BGR['intense_red'], zero_color = const.BGR['intense_red'],
                            bandwidth = 30, showReg = oppReg, resolution = opp_res)
heatScn = heatScn_in.combine(heatScn_out1).combine(heatScn_out2)

from ATMANpy.auxil.court import birds_eye_half_court_cam, baseline_side_cam, central_side_cam, back_view_asymmetric_cam
topCam = birds_eye_half_court_cam(side = 'Bot')
cnt_sideCam = central_side_cam(from_where = 'right', width = 400)
bsl_sideCam = baseline_side_cam(which = 'Ad', side = 'Top', width = 200)
behindCam = back_view_asymmetric_cam(which = 'Ad', side = 'Top')

### animation parameters
back_res = 1
shape_res = 1

######################
###### ANIMATION #####
######################
all_frames = []
script_lines = []

script_line =   """
                    In this example, it might be more instructive to look at the heatmap from above.
                """
myFrms, _ = vs.animate_cam_motion(heatScn, start_cam = myCam.resize(3), stop_cam = topCam, 
                                background_scene = courtScn, interpolate_func = vs.interpolate_camera_naive,
                                num_frames = 50, argument_resolution = shape_res, background_resolution = back_res)

# DecapStream(myFrms, 0, 100).show(ms_between_frames = 40)
# myFrms[-1].show()

all_frames.append(myFrms)
script_lines.append(script_line)

### fade in trajectories from above
script_line =   """
                    Alternatively, we can take a deeper look at the return trajectories and characteristics.
                """
from ATMANpy.types.Shapes3D import Lines3D
def full_splines_until_bounce(myShot, spline_thickness = 0.5):
    bnc_frame = myShot.get_bounce_frames()[0]
    splnSF = myShot.feat_spline_cumul()
    spline_list = [spln.resize(spline_thickness) for spln in splnSF[bnc_frame] if not vis_ut.has_nan_point_or_is_none(spln)]
    full_splines = Lines3D(spline_list, color = const.BGR['ball_yellow'])
    return full_splines
splnLns = [full_splines_until_bounce(myShot) for myShot in retShots_rect_in]

splnFrms, _ = vs.animate_static(splnLns, static_cam = topCam, num_frames = 1, background_scene = courtScn,
                                argument_resolution = shape_res, background_resolution = back_res)

interFrms = vs.interpolate_frames(myFrms[-1], splnFrms[0], 100) ### is this too lazy for fade-in fade-out? it looks okay. 

all_frames.append(interFrms)
script_lines.append(script_line)

DecapStream(interFrms, 0, 100).show()

animFrms1, _ = vs.animate_cam_motion(splnLns, start_cam = topCam, stop_cam = cnt_sideCam, background_scene = courtScn,
                                interpolate_func = vs.interpolate_camera_naive, 
                                num_frames = 50, argument_resolution = shape_res, background_resolution = back_res)

all_frames.append(animFrms1)
script_lines.append(script_line)

script_line = "For example, let's take a quick look at the typical point of impact."

animFrms2, animDict = vs.animate_cam_motion(splnLns, start_cam = cnt_sideCam, stop_cam = behindCam, background_scene = courtScn,
                                interpolate_func = vs.interpolate_camera_naive, 
                                num_frames = 50, argument_resolution = shape_res, background_resolution = back_res)
# animFrms = BGRFrames.from_frames([animFrms1, animFrms2])
# DecapStream(animFrms, 0, 100).show()

### make little lines indicating height of average hitting point and depth of average hitting point.
from ATMANpy.types.Shapes3D import Point3D, Line3D, Ball3D
from ATMANpy.types.ShapesTennis3D import Mark
x, y, z = np.mean(retShots_rect_in.get_ball_hit_position(), axis = 0)
y_bsl = courtLines['Top Baseline'].rightPt.y

meanPt = Point3D((x, y, z))
meanPt_on_floor = Point3D((x, y, 0))
meanPt_on_baseline = Point3D((x, y_bsl, 0))

floorMark = Mark.make(meanPt_on_floor, color = const.BGR['intense_red'], size = 4)
meanBall = Ball3D(meanPt.coords, color = const.BGR['intense_red'], radius = 1)

upLn = Line3D(meanPt_on_floor, meanPt)
inLn = Line3D(meanPt_on_floor, meanPt_on_baseline)

num_dashes = 20
frames_per_dash = 4 

upLn_dashed = upLn.dash(num_dashes).recolor(const.BGR['rose_orange'])
inLn_dashed = inLn.dash(num_dashes).recolor(const.BGR['rose_orange'])

dash_scenes = []
for dash_idx in range(1, num_dashes + 1):
    dashScn = Scene([floorMark, upLn_dashed[:dash_idx], inLn_dashed[:dash_idx]])
    for _ in range(frames_per_dash):
        dash_scenes.append(dashScn)

dashFrms, _ = vs.animate_sequence(dash_scenes, start_cam = behindCam, stop_cam = behindCam, background_scene = courtScn, **animDict,
                                    argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res)
# DecapStream(dashFrms, 0, 100).show()

dash_and_ball_scene = dash_scenes[-1].put(meanBall)

dnbFrms, _ = vs.animate_static(dash_and_ball_scene, static_cam = behindCam, background_scene = courtScn, 
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res, 
                                num_frames = 10,
                                **animDict)

anim_and_dashFrms = BGRFrames.from_frames([animFrms2, dashFrms, dnbFrms])

all_frames.append(anim_and_dashFrms)
script_lines.append(script_line)

### annotate height and depth
script_line = "We can extract average hit height and depth, for example."
h = upLn.length() / const.SCALE * const.FOOT_TO_METER
d = inLn.length() / const.SCALE * const.FOOT_TO_METER

annPnt_up = upLn.center_point().left(5).annotate(f'Average Return Hit Height: {h:.2f} m.').recolor(const.BGR['rose_orange']).rethicken(1)
annPnt_in = inLn.center_point().left(5).annotate(f'Average Return Hit Depth: {d:.2f} m.').recolor(const.BGR['rose_orange']).rethicken(1)

annFrms, _ = vs.animate_static(dash_and_ball_scene, static_cam = behindCam, background_scene = courtScn, 
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 25,
                                annotated_points = [annPnt_up, annPnt_in], **animDict)

all_frames.append(annFrms)
script_lines.append(script_line)

### splines rectified to average hitting point
script_line =   """
                    We can also calculate the return launch angle with ease. 
                    To visualize it, we make as if all shots had originated from the mean impact point...
                """

reanchored_trajs = [shot.shotTraj.reanchor(meanPt) for shot in retShots_rect_in]
reanchored_splnLns = [full_splines_until_bounce(myTraj) for myTraj in reanchored_trajs]

origScn = Scene(splnLns)
newScn = Scene(reanchored_splnLns)
itplScns = vs.interpolate_scenes(origScn, newScn, 50)

animFrms3, _ = vs.animate_sequence(itplScns, start_cam = behindCam, stop_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene, annotated_points = [annPnt_up, annPnt_in],
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res)
# DecapStream(animFrms3, 0, 100).show()

statFrms0, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene, annotated_points = [annPnt_up, annPnt_in],
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 50)

statFrms1, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene, annotated_points = [annPnt_up],
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 25)

statFrms2, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene,
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 20)

statFrms3, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene.drop(2),
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 17)

statFrms4, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene.drop(2).drop(0),
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 13)

statFrms5, animDict = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene.drop(2).drop(0).drop(0),
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 10)

remFrms = BGRFrames.from_frames([animFrms3, statFrms0, statFrms1, statFrms2, statFrms3, statFrms4, statFrms5])
DecapStream(remFrms, 0, 100).show()

all_frames.append(remFrms)
script_lines.append(script_line)

### move around to show axes
script_line = "... and seamlessly build axes and a mean launch angle vector into the scene..."
ax_sideCam = Camera.from_to(meanPt.right(50), meanPt)

moveFrms, animDict = vs.animate_cam_motion(Scene([meanBall]), start_cam = behindCam, stop_cam = ax_sideCam,
                                        background_scene = courtScn, interpolate_func = vs.interpolate_camera_naive,
                                        argument_resolution = shape_res, background_resolution = back_res,
                                        num_frames = 50,
                                        **animDict)
### shape_res has to equal ball_res! (which is shape_res!) 
# DecapStream(moveFrms, 0, 100).show(ms_between_frames = 100)

y_len = 35
z_len = 25
yLn = Line3D(meanPt, meanPt.back(y_len))
zLn = Line3D(meanPt, meanPt.up(z_len))
lineScn = Scene([yLn, zLn])

nullLn = Line3D(meanPt, meanPt)
nullScn = Scene([nullLn, nullLn])

axisScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = 25)
axisScns = [myScn.put(meanBall) for myScn in axisScns] ### avoid the transparification of the ball

axisFrms, animDict = vs.animate_sequence(axisScns, start_cam = ax_sideCam, stop_cam = ax_sideCam, background_scene = courtScn,
                                        cumulative_foreground_transparence = 0.95, 
                                        argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                        **animDict
                                        ) 

### show angle
mean_angle = np.mean(retShots_rect_in.get_launch_angle())
mean_rad = mean_angle * const.ANGLE_TO_RAD

y_unit, z_unit = np.cos(mean_rad), np.sin(mean_rad)
y, z = y_unit * z_len * 1.1, z_unit * z_len * 1.1
angPt = meanPt.back(y).up(z)

angLn = Line3D(meanPt, angPt)

nullScn = Scene([nullLn])
angScn = Scene([angLn])
angScns = vs.interpolate_scenes(nullScn, angScn, num_frames = 50)

angFrms, animDict = vs.animate_sequence(angScns, start_cam = ax_sideCam, stop_cam = ax_sideCam, background_scene = courtScn,
                                        argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                        **animDict)
# DecapStream(angFrms, 0, 100).show()

jntFrms = BGRFrames.from_frames([moveFrms, axisFrms, angFrms])
DecapStream(jntFrms, 0, 100).show()

all_frames.append(jntFrms)
script_lines.append(script_line)

script_line = ""

from ATMANpy.types.Shapes3D import Circle3D
degCrc = Circle3D.from_point3D(meanPt, radius = 0.85 * z_len, limit_theta = (0, mean_rad), fixed_dimension = 'x')
nullCrc = Circle3D.from_point3D(meanPt, radius = 0.85 * z_len, limit_theta = (0, 0), fixed_dimension = 'x')

degScn = Scene([degCrc])
nullScn = Scene([nullCrc])
degScns = vs.interpolate_scenes(nullScn, degScn, num_frames = 25)

angFrms_stat, animDict = vs.animate_sequence(degScns, start_cam = ax_sideCam, stop_cam = ax_sideCam, background_scene = courtScn,
                                            argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                            **animDict)
DecapStream(angFrms_stat, 0, 100).show()

### annotate angle
degPnt = meanPt.back(0.82 * z_len).up(0.75).annotate(f'Launch Angle: {mean_angle:.1f} deg.')

degFrms, _ = vs.animate_static(degScns[-1], static_cam = ax_sideCam, background_scene = courtScn, 
                                annotated_points = [degPnt],
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                **animDict)

jointFrms = BGRFrames.from_frames([angFrms_stat, degFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)


### TODO: need a fazit. for comparison, here's what we say in the beginning.
"""
    The Automated Tennis Match Annotator (ATMAN) automatically recognizes the lines, the players, the ball
    (how fast the ball is, how much spin it has, where it bounces) to be able to summarize a match set by set, game by game,
    point by point, shot by shot.
    Match data can be easily combined with other matches that have been visually recognized by ATMAN.
    The system then allows all matches at hand to be easily searched, filtered and aggregated 
    to reveal patterns in the style of play of a player, understand how he or she wins matches, or where he or she is vulnerable. 
    ATMAN also includes a 3D engine that is used to visualize insights from the matches at hand.
"""
### what more can we say in the end?

# -> ATMAN, as we have shown, is a start-to-end system, allowing to do everything within one framework.
# everything being: visual recognition; data search, filtering and aggregation; visualization/presentation of results and insights.  
# to see examples of the framework in action, to see what it can find out (with a limited set of data) -> check out analysis videos!

"""

"""


vis_ut.show(all_frames)