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

import os
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

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'visual_engine'
SAVE_FOLDER = 'atman_capabilities'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

#########################################

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
cap = confs[0]['cap']

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

myCam = Camera.calibrated_brest_behind()

courtScn = Scene.make_court(singles_net = False, net_alpha = 0.3)

from ATMANpy.types.Lines import CourtLines, Region
courtLines = CourtLines.make().add_singles_netline()

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

background_bgr = tuple(BGRFrame.from_video(cap, 1100).array[200, 350] * 0.25) ### the color of the court, dimmed
contScn = vis_ut.kde_func(retShots_rect, props = [0.3, 0.5, 0.75, 0.9], resolution = 5,
                            background_bgr = background_bgr,
                            heat_bgr = const.BGR['leafy_green'], exp_bgr = 2, bandwidth = 60)

### fade in a point up until moment of return that has Brandon Nakashima on far side.
pt_idx = 9 ### this is from the detection_filtering script

retShots = filtPnts4.get_returns()
retShots_rect = retShots.enforce_direction('Top')

retShots_rect_out = retShots_rect.filter('is_fair', nextShot = None, negate = True)
retShots_rect_in = retShots_rect.filter('is_fair', nextShot = None)

from ATMANpy.auxil.court import birds_eye_half_court_cam, baseline_side_cam, central_side_cam, back_view_asymmetric_cam
topCam = birds_eye_half_court_cam(side = 'Bot')
cnt_sideCam = central_side_cam(from_where = 'right', width = 400)
bsl_sideCam = baseline_side_cam(which = 'Ad', side = 'Top', width = 200)
behindCam = back_view_asymmetric_cam(which = 'Ad', side = 'Top')

### animation parameters
back_res = 12# 12 ##1 
shape_res = 20 # 7.5 ##1 ### -> double the shape resolution to at least 15, maybe even 20

######################
###### ANIMATION #####
######################
all_frames = []
script_lines = []

script_line =   """
                    In this example, it might be more instructive to look at the heatmap from above.
                """
myFrms, _ = vs.animate_cam_motion(contScn, start_cam = myCam.resize(3), stop_cam = topCam, num_frames = 50, 
                                background_scene = courtScn, interpolate_func = vs.interpolate_camera_naive,
                                argument_resolution = shape_res, background_resolution = back_res, foreground_resolution = shape_res)

stopFrms = BGRFrames.from_frame(myFrms[-1], 125)

joinFrms = BGRFrames.from_frames([myFrms, stopFrms])

# DecapStream(myFrms, 0, 100).show(ms_between_frames = 40)
# myFrms[-1].show()

all_frames.append(joinFrms)
script_lines.append(script_line)

### fade in trajectories from above
script_line =   """
                    Alternatively, we can take a deeper look at a certain aspect of the return trajectories.
                """
from ATMANpy.types.Shapes3D import Lines3D
def full_splines_until_bounce(myShot, spline_thickness = 0.25):
    bnc_frame = myShot.get_bounce_frames()[0]
    splnSF = myShot.feat_spline_cumul()
    spline_list = [spln.resize(spline_thickness) for spln in splnSF[bnc_frame] if not vis_ut.has_nan_point_or_is_none(spln)]
    full_splines = Lines3D(spline_list, color = const.BGR['ball_yellow'])
    return full_splines
splnLns = [full_splines_until_bounce(myShot) for myShot in retShots_rect_in]

splnFrms, _ = vs.animate_static(splnLns, static_cam = topCam, num_frames = 1, background_scene = courtScn,
                                argument_resolution = shape_res, background_resolution = back_res, foreground_resolution = shape_res)

interFrms = vs.interpolate_frames(myFrms[-1], splnFrms[0], 100) ### is this too lazy for fade-in fade-out? it looks okay. 
# DecapStream(interFrms, 0, 100).show()


animFrms1, _ = vs.animate_cam_motion(splnLns, start_cam = topCam, stop_cam = cnt_sideCam, background_scene = courtScn,
                                interpolate_func = vs.interpolate_camera_naive, num_frames = 50, 
                                argument_resolution = shape_res, background_resolution = back_res, foreground_resolution = shape_res)
# DecapStream(animFrms1, 0, 100).show()

jointFrms = BGRFrames.from_frames([interFrms, animFrms1])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line = "For example, let's find the typical point from where the return is hit."

animFrms2, animDict = vs.animate_cam_motion(splnLns, start_cam = cnt_sideCam, stop_cam = behindCam, background_scene = courtScn,
                                interpolate_func = vs.interpolate_camera_naive, num_frames = 50, 
                                argument_resolution = shape_res, background_resolution = back_res, foreground_resolution = shape_res)
# DecapStream(animFrms2, 0, 100).show()

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

num_dashes = 25
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

anim_and_dashFrms = BGRFrames.from_frames([animFrms2, dashFrms[:50]])

all_frames.append(anim_and_dashFrms)
script_lines.append(script_line)

### annotate height and depth
script_line = "We can extract average hit height and depth..."

dash_and_ball_scene = dash_scenes[-1].put(meanBall)

dnbFrms, _ = vs.animate_static(dash_and_ball_scene, static_cam = behindCam, background_scene = courtScn, 
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res, 
                                num_frames = 10,
                                **animDict)

h = upLn.length() / const.SCALE * const.FOOT_TO_METER
d = inLn.length() / const.SCALE * const.FOOT_TO_METER
annPnt_up = upLn.center_point().left(8).annotate(f'Average Return   Hit Height: {h:.2f} m.').recolor(const.BGR['rose_orange']).rethicken(1)
annPnt_in = inLn.center_point().left(5).annotate(f'Average Return Hit Depth: {d:.2f} m.').recolor(const.BGR['rose_orange']).rethicken(1)

annFrms, _ = vs.animate_static(dash_and_ball_scene, static_cam = behindCam, background_scene = courtScn, 
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 50,
                                annotated_points = [annPnt_up, annPnt_in], **animDict)

dash_and_annFrms = BGRFrames.from_frames([dashFrms[50:], dnbFrms, annFrms])

all_frames.append(dash_and_annFrms)
script_lines.append(script_line)

### splines rectified to average hitting point
script_line =   """
                    ... and we can visualize the average return launch angle with ease. 
                """

all_frames.append(annFrms)
script_lines.append(script_line)

script_line = "To this end, we make as if all shots had originated from the mean impact point..."

reanchored_trajs = [shot.shotTraj.reanchor(meanPt) for shot in retShots_rect_in]
reanchored_splnLns = [full_splines_until_bounce(myTraj) for myTraj in reanchored_trajs]

origScn = Scene(splnLns)
newScn = Scene(reanchored_splnLns)
itplScns = vs.interpolate_scenes(origScn, newScn, 50)

animFrms3, _ = vs.animate_sequence(itplScns, start_cam = behindCam, stop_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene, annotated_points = [annPnt_up, annPnt_in],
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res)
# DecapStream(animFrms3, 0, 100).show()

ann_and_animFrms = BGRFrames.from_frames([annFrms, animFrms3])
all_frames.append(ann_and_animFrms)
script_lines.append(script_line)

script_line = ""
# statFrms0, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
#                                 previous_foreground_scene = dash_and_ball_scene, annotated_points = [annPnt_up, annPnt_in],
#                                 argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
#                                 num_frames = 25)

statFrms1, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene, annotated_points = [annPnt_up],
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 20)

statFrms2, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene,
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 17)

statFrms3, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene.drop(2),
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 13)

statFrms4, _ = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene.drop(2).drop(0),
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 10)

statFrms5, animDict = vs.animate_static(itplScns[-1], static_cam = behindCam, background_scene = courtScn,
                                previous_foreground_scene = dash_and_ball_scene.drop(2).drop(0).drop(0),
                                argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                num_frames = 5)

remFrms = BGRFrames.from_frames([statFrms1, statFrms2, statFrms3, statFrms4, statFrms5])
# DecapStream(remFrms, 0, 100).show()

all_frames.append(remFrms)
script_lines.append(script_line)

### move around to show axes
script_line = "... and seamlessly build axes and a mean launch angle vector into the scene..."
ax_sideCam = Camera.from_to(meanPt.right(50), meanPt)

moveFrms, animDict = vs.animate_cam_motion(Scene([meanBall]), start_cam = behindCam, stop_cam = ax_sideCam,
                                        background_scene = courtScn, interpolate_func = vs.interpolate_camera_naive,
                                        argument_resolution = shape_res, background_resolution = back_res,
                                        foreground_resolution = shape_res,
                                        num_frames = 50,
                                        **animDict)
# DecapStream(moveFrms, 0, 100).show(ms_between_frames = 100)

from ATMANpy.types.ShapesTennis3D import Arrow

ax_color = const.BGR['marine_blue']

y_len = 35
z_len = 25
yLn = Arrow.make(meanPt, meanPt.back(y_len), rot_angle = np.pi / 2, color = ax_color)
zLn = Arrow.make(meanPt, meanPt.up(z_len), rot_angle = np.pi / 2, color = ax_color)
lineScn = Scene([yLn, zLn])

nullLn = Arrow.make(meanPt, meanPt.back(0.1), color = ax_color)
nullScn = Scene([nullLn, nullLn])

axisScns = vs.interpolate_scenes(nullScn, lineScn, num_frames = 25)
axisScns = [myScn.put(meanBall) for myScn in axisScns] ### avoid the transparification of the ball

axisFrms, animDict = vs.animate_sequence(axisScns, start_cam = ax_sideCam, stop_cam = ax_sideCam, background_scene = courtScn,
                                        cumulative_foreground_transparence = 0.925, 
                                        argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                        **animDict
                                        ) 

### annotate axes
yPt = (meanPt + meanPt.back(y_len)) / 2
zPt = (meanPt + meanPt.up(z_len)) / 2

yPt = yPt.annotate('Ground Speed').recolor(ax_color).down(1.5).back(1.5)
zPt = zPt.annotate('Vertical\nSpeed').recolor(ax_color).front(1)

annotFrms1, animDict = vs.animate_static(axisScns[-1], static_cam = ax_sideCam, background_scene = courtScn,
                                        argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                        annotated_points = [yPt], num_frames = 10,
                                        **animDict)

annotFrms2, animDict = vs.animate_static(axisScns[-1], static_cam = ax_sideCam, background_scene = courtScn,
                                        argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                        annotated_points = [yPt, zPt], num_frames = 10,
                                        **animDict)

### show angle
mean_angle = np.mean(retShots_rect_in.get_launch_angle())
mean_rad = mean_angle * const.ANGLE_TO_RAD

y_unit, z_unit = np.cos(mean_rad), np.sin(mean_rad)
y, z = y_unit * z_len * 1.1, z_unit * z_len * 1.1
angPt = meanPt.back(y).up(z)

ang_color = const.BGR['bright_blue']

angLn = Line3D(meanPt, angPt)
nullLn = Line3D(meanPt, meanPt)

nullScn = Scene([nullLn.recolor(ang_color).resize(0.15)])
angScn = Scene([angLn.recolor(ang_color).resize(0.15)])
angScns = vs.interpolate_scenes(nullScn, angScn, num_frames = 50)

angFrms, animDict = vs.animate_sequence(angScns, start_cam = ax_sideCam, stop_cam = ax_sideCam, background_scene = courtScn,
                                        argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                        annotated_points = [yPt, zPt],
                                        **animDict)
# DecapStream(angFrms, 0, 100).show()

jntFrms = BGRFrames.from_frames([moveFrms, axisFrms, annotFrms1, annotFrms2, angFrms])
# DecapStream(jntFrms, 0, 100).show()

all_frames.append(jntFrms)
script_lines.append(script_line)

script_line = f"... to find that the average launch angle of returns on these ad-side, out-wide serves is {mean_angle:.1f} degrees."

from ATMANpy.types.Shapes3D import Circle3D
degCrc = Circle3D.from_point3D(meanPt, radius = 0.85 * z_len, limit_theta = (0, mean_rad), fixed_dimension = 'x').recolor(ang_color)
nullCrc = Circle3D.from_point3D(meanPt, radius = 0.85 * z_len, limit_theta = (0, 0), fixed_dimension = 'x').recolor(ang_color)

degScn = Scene([degCrc])
nullScn = Scene([nullCrc])
degScns = vs.interpolate_scenes(nullScn.combine(angScn), degScn.combine(angScn), num_frames = 25)

angFrms_stat, animDict = vs.animate_sequence(degScns, start_cam = ax_sideCam, stop_cam = ax_sideCam, background_scene = courtScn,
                                            argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                            annotated_points = [yPt, zPt],
                                            **animDict)
# DecapStream(angFrms_stat, 0, 100).show()

### annotate angle
degPnt = meanPt.back(0.82 * z_len).up(0.75).annotate(f'Launch Angle: {mean_angle:.1f} deg.').recolor(ang_color)

degFrms, animDict = vs.animate_static(degScns[-1], static_cam = ax_sideCam, background_scene = courtScn, 
                                    annotated_points = [degPnt, yPt, zPt], num_frames = 200,
                                    argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                    **animDict)

jointFrms = BGRFrames.from_frames([angFrms_stat, degFrms])

all_frames.append(jointFrms)
script_lines.append(script_line)

script_line =   """
                    ATMAN, as we have shown, is a start-to-end-framework for the analysis of tennis matches, 
                    a system that, even when fed with comparatively low-quality match footage, 
                    is able to deal with visual recognition, data search, filtering and analysis, and the visualization of results and insights.
                    Every facet of a player's game can be analyzed to track progress, reveal vulnerabilities, assist in training design
                    and help work out a game plan tailored to specific opponents.
                    To see examples of the framework employed in actual case studies, 
                    check out the two analyses on serve and return behaviour supplied along with this video.  
                """


## for this: rotate camera to a bird's eye view of the court.

from ATMANpy.auxil.court import cnt_2D

dPt = Point3D.from_point2D(cnt_2D, z = 0)
oPt = Point3D.from_point2D(cnt_2D, z = 500)
cntCam = Camera.from_to(oPt, dPt).rotate(0.5)

# View.make(cntCam, courtScn, resolution = 5).snap().show()

finalScn = Scene([])
finalScn = itplScns[-1]
animDict_no_text = {**animDict}
animDict_no_text['previous_annotated_points'] = []

start_alpha = animDict['previous_foreground_scene'][0].alpha
rebrightenScns = vs.interpolate_scenes(finalScn.realpha(start_alpha), finalScn, num_frames = 100)

finalFrms, _ = vs.animate_sequence(rebrightenScns, start_cam = ax_sideCam, stop_cam = cntCam, background_scene = courtScn, 
                                    argument_resolution = shape_res, foreground_resolution = shape_res, background_resolution = back_res,
                                    cumulative_foreground_transparence = 0.95, interpolate_func = vs.interpolate_camera_naive,
                                    **animDict_no_text)
# DecapStream(finalFrms, 0, 100).show()

all_frames.append(finalFrms)
script_lines.append(script_line)

vis_ut.show(all_frames)

from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
fullPath = maybe_make_folder(savePath, saveFolder)

# b) Save segments
for idx, subFrms in enumerate(all_frames):
    subFrms.save(fullPath, f'seg_{idx}', save_as = 'mp4', lossless = True)

# c) Save script
for idx, script_line in enumerate(script_lines):
    with open(os.path.join(fullPath, f'script_{idx}.txt'), mode = 'w') as f:
        f.write(script_line)
