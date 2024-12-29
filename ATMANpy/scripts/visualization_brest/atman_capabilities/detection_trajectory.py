
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

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
### CHOOSE VIDEO
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
for idx, vid in enumerate(avail_vids):
    print(idx, vid)

idx = 12
conf = cnf_mod.get_conf(avail_vids[idx])

### GET ABSTRACT POINT REPRESENTATION 
dataTable = read_in_data_table(conf, conf_idx = idx)
nullTable = read_in_placeholder_table(conf, conf_idx = idx)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

################################
###### GET USEFUL STUFF ########
################################

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)

vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
fs, ls = conf['start_frame'], conf['stop_frame']
cap = conf['cap']

segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()


dim_factor = 0.5

myPnt = allPnts[0]
# myPnt.stream_ball(P, cap = cap).show()

sht0, sht1, sht2 = myPnt[0], myPnt[1], myPnt[2]
stm0, stm1, stm2 = sht0.stream(cap), sht1.stream(cap), sht2.stream(cap)

frms0, frms1, frms2 = stm0.decap().frames.rescale(1.5), stm1.decap().frames.rescale(1.5), stm2.decap().frames.rescale(1.5)

### get ball splines

ball_color = const.BGR['ball_yellow']

splns0 = sht0.feat_spline_cumul(which_endpoint = 'penultimate').values()
splns1 = myPnt[1].feat_spline_cumul(which_endpoint = 'penultimate').values()
splns2 = myPnt[2].feat_spline_cumul(which_endpoint = 'penultimate').values()

from ATMANpy.types.Visual3D import Camera, Scene
myCam_small = Camera.calibrated_brest_behind()
myCam = Camera.calibrated_brest_behind().resize(1.5)
myCam_full = Camera.calibrated_brest_behind().resize(3)

scns0 = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns if spln is not None]) for splns in splns0]
scns1 = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns if spln is not None]) for splns in splns1]
scns2 = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns if spln is not None]) for splns in splns2]

from ATMANpy.funcs.vis_funcs import animate_sequence

spln_frms0, _ = animate_sequence(scns0, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms0)
spln_frms1, _ = animate_sequence(scns1, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms1)
spln_frms2, _ = animate_sequence(scns2, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms2)

DecapStream(spln_frms0, 0, 100).show(ms_between_frames = 100)
DecapStream(spln_frms1, 0, 100).show(ms_between_frames = 100)
DecapStream(spln_frms2, 0, 100).show(ms_between_frames = 100)

ls_frm_dim0 = spln_frms0[-1].dim(dim_factor)
ls_frm_dim1 = spln_frms1[-1].dim(dim_factor)
ls_frm_dim2 = spln_frms2[-1].dim(dim_factor)
frms0_ls, frms1_ls, frms2_ls = BGRFrames.from_frame(ls_frm_dim0, 25), BGRFrames.from_frame(ls_frm_dim1, 25), BGRFrames.from_frame(ls_frm_dim2, 25)

myShot = sht0
def get_data_string(myShot):
    velo = myShot.get_initial_speed()
    spin = myShot.get_initial_spin()
    angl = myShot.get_launch_angle()

    veloStr = f'{int(round(velo, 0))} km/h'
    spinStr = f'{int(round(spin, 0))} RPM'
    anglStr = f'{round(angl, 1)} deg.'

    return f'Speed: {veloStr} \n\nSpin: {spinStr} \n\nLaunch Angle: {anglStr}'

### THE TABLE ###
str0, str1, str2 = vis_ut.get_shot_string(sht0), vis_ut.get_shot_string(sht1), vis_ut.get_shot_string(sht2)
dat0, dat1, dat2 = vis_ut.get_data_string(sht0), vis_ut.get_data_string(sht1), vis_ut.get_data_string(sht2)

tab_params = {  'index_offset': 1, 'index_col_width': 0.5, 'dimensions': (960, 540), 'skeleton_shrinkage': 0.2, 
                'automatic_vertical_offset_text_line_height': 0.4,
                'null_columns': 0
                }

tab_data0 = {'1st Shot': {'Description': str0, 'Data': dat0}}
tab_data1 = {'1st Shot': {'Description': str0, 'Data': dat0}, '2nd Shot': {'Description': str1, 'Data': dat1}}
tab_data2 = {'1st Shot': {'Description': str0, 'Data': dat0}, '2nd Shot': {'Description': str1, 'Data': dat1}, '3rd Shot': {'Description': str2, 'Data': dat2}}

tab0 = Table(tab_data0, **tab_params, null_rows = 2)
tab1 = Table(tab_data1, **tab_params, null_rows = 1)
tab2 = Table(tab_data2, **tab_params, null_rows = 0)

tab_frm0 = tab0.render()
tab_frm1 = tab1.render()
tab_frm2 = tab2.render()

tab_frms0 = BGRFrames.from_frame(tab_frm0, 25)
tab_frms1 = BGRFrames.from_frame(tab_frm1, 25)
tab_frms2 = BGRFrames.from_frame(tab_frm2, 25)

tab_frm0.show()
tab_frm1.show()
tab_frm2.show()

### THE RASTER ###
myRst = FramesRaster(2, 2)

myRst.add(spln_frms0, 0, 0)
myRst.add(tab_frms0, 1, 1)
rstFrms0 = myRst.render()
DecapStream(rstFrms0, 0, 100).show(ms_between_frames = 100)

myRst.add(frms0_ls, 0, 0)
myRst.add(spln_frms1, 0, 1)
myRst.add(tab_frms1, 1, 1)
rstFrms1 = myRst.render()
DecapStream(rstFrms1, 0, 100).show(ms_between_frames = 100)

myRst.add(frms1_ls, 0, 1)
myRst.add(spln_frms2, 1, 0)
myRst.add(tab_frms2, 1, 1)
rstFrms2 = myRst.render()
DecapStream(rstFrms2, 0, 100).show(ms_between_frames = 100)

rstFrms2[-1].show()

##### ANIMATION ######

all_frames = []
script_lines = []

script_line =   """
                    Trajectories are detected by combining all the available empirical information on ball flight shapes, sound of bounce and hit, player swing. 
                    This empirical information is joint in a theoretical framework that simulates, given an initial launch angle, speed and spin, how a ball should theoretically behave in flight, at any point in time.
                    We then deduce the actual trajectory by selecting the theoretical trajectory with which all empirical evidence agrees most.
                    In this way, not only do we get a trajectory that cleanly combines visual and aural information on ball flight, we also get a trajectory that is consistent with the laws of physics, 
                    and plausible given the limitations of player, racket and ball.
                    
                    As a nice by-product, the speed, spin and launch angle of the ball can be directly read out as the initial conditions that uniquely describe the chosen trajectory.

                    # This theoretical model takes into account all the forces acting on a ball in flight (drag, lift and gravity). 
                """
pre_frames = 150
preFrms = stm0.widen(pre_frames).decap().frames[:pre_frames]
preArr_dimmed = np.stack([f.dim(min((idx / pre_frames) ** 2, 1)).array for idx, f in enumerate(preFrms)], axis = 0)
preFrms_dimmed = BGRFrames(preArr_dimmed)
preFrms_dimmed = BGRFrames.from_frames([preFrms_dimmed[:50], preFrms_dimmed[50:100].slow(2), preFrms_dimmed[100:].slow(2).slow(2)])

dimRst = FramesRaster(2, 2)
dimRst.add(preFrms_dimmed, 0, 0)
rstFrms_init = dimRst.render()
rstFrms_init = rstFrms_init.rescale(1.5)

all_frames.append(rstFrms_init)
script_lines.append(script_line)

script_line =   """
                    Serve is recognized as more of a kick serve than a flat serve; landing location is very good however. 
                    The serve is generally more challenging to recognize since it flies faster and therefore there are fewer frames for the ball to be tracked in, and the balls are blurry within the frames.
                """
rstFrms_slow = rstFrms0.slow(2)
rstFrms_slow_add = BGRFrames.from_frame(rstFrms_slow[-1], 50)
rstFrms_slow_all = BGRFrames.from_frames([rstFrms_slow, rstFrms_slow_add])

all_frames.append(rstFrms_slow_all)
script_lines.append(script_line)


script_line =   """
                    The return is well recognized as a backhand slice at 77 km/h, 2500 RPM and a launch angle of 10.8 degrees.
                """
rstFrms_slow = rstFrms1.slow(2)
rstFrms_slow_add = BGRFrames.from_frame(rstFrms_slow[-1], 50)
rstFrms_slow_all = BGRFrames.from_frames([rstFrms_slow, rstFrms_slow_add])

all_frames.append(rstFrms_slow_all)
script_lines.append(script_line)

script_line =   """
                    The server's forehand is tracked perfectly, flying out of the court at 111 km/h at a considerable topspin rate of 3750 RPM.
                """
rstFrms_slow = rstFrms2.slow(2)
rstFrms_slow_add = BGRFrames.from_frame(rstFrms_slow[-1], 50)
rstFrms_slow_all = BGRFrames.from_frames([rstFrms_slow, rstFrms_slow_add])

all_frames.append(rstFrms_slow_all)
script_lines.append(script_line)

### remove raster elements, then remove trajectory, to prepare for zoom
from ATMANpy.types.Overlays import reverse_cascade, overlay

script_line = ""
finRst = myRst.to_frame_raster()
revRst, revFrms = reverse_cascade(finRst, [(1, 0)], frames_per_element = 15)
all_frames.append(revFrms)
script_lines.append(script_line)

script_line = ""
lsFrm = stm2.get(stm2.stop_frame).rescale(1.5)
y, x, _ = lsFrm.shape

blackFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))

tlPt = Point2D((0, 1080 - y))
overFrm = overlay(blackFrm, lsFrm, tlPt)
overFrms = BGRFrames.from_frame(overFrm, 10)

all_frames.append(overFrms)
script_lines.append(script_line)

### next few rallyes --> slowly zoom out at the end of the last shot.
script_line =   """
                    Let us now take a look at the next few rallyes to see more of the trajectory detection at work.
                """
fs_zoom, ls_zoom = stm2.stop_frame + 1, stm2.stop_frame + 101
zoomStm = Stream(cap, fs_zoom, ls_zoom - 1)
zoomStm.show()

zoom_frames = []
for frm_num, frm, _ in zoomStm.generator():
    lmbd = (ls_zoom - frm_num) / (ls_zoom - fs_zoom)
    lmbd_rescale_factor = 1.5 * lmbd + 3 * (1 - lmbd)

    frm_resc = frm.rescale(lmbd_rescale_factor)
    y, x, _ = frm_resc.shape
    tlPt = Point2D((0, 1080 - y))
    
    frm_resc_over = overlay(blackFrm, frm_resc, tlPt)
    zoom_frames.append(frm_resc_over)

zoomFrms = BGRFrames.from_single_frames(zoom_frames)

all_frames.append(zoomFrms)
script_lines.append(script_line)

### prep next points
nextPnt1 = allPnts[1]
nextPnt2 = allPnts[2]

stm_p1 = nextPnt1.stream(cap = cap)
stm_p2 = nextPnt2.stream(cap = cap)

### transition to beginning of trajectory
script_line = ""
start, stop = ls_zoom, stm_p1.start_frame

transStm1 = Stream(cap, start, stop - 1)
transFrms = transStm1.decap().frames.rescale(3)
transFrms = vis_ut.speed_up_frames_progressively(transFrms, ramp_up_frames = 25)

all_frames.append(transFrms)
script_lines.append(script_line)

### show next points

script_line = ""

point_splns = nextPnt1.feat_spline_cumul(which_endpoint = 'penultimate').values()
point_splns = [[spln for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)] for splns in point_splns ]
scns = [Scene([spln.recolor(ball_color).resize(1) for spln in splns]) for splns in point_splns]

frms = stm_p1.decap().frames[:-1]

splnFrms1, _ = animate_sequence(scns, start_cam = myCam_small, stop_cam = myCam_small, cumulate_argument = False, animate_over = frms)
splnFrms1 = splnFrms1.rescale(3)
# DecapStream(splnFrms1, 0, 100).show()

all_frames.append(splnFrms1)
script_lines.append(script_line)

### transition to the next trajectory
script_line = ""
start, stop = stm_p1.stop_frame, stm_p2.start_frame

transStm2 = Stream(cap, start, stop - 1)
transFrms = transStm2.decap().frames.rescale(3)
transFrms = vis_ut.speed_up_frames_progressively(transFrms, ramp_up_frames = 25)

all_frames.append(transFrms)
script_lines.append(script_line)

script_line = ""

point_splns = nextPnt2.feat_spline_cumul(which_endpoint = 'penultimate').values()
point_splns = [[spln for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)] for splns in point_splns ]
scns = [Scene([spln.recolor(ball_color).resize(1) for spln in splns]) for splns in point_splns]

frms = stm_p2.decap().frames[:-1]

splnFrms2, _ = animate_sequence(scns, start_cam = myCam_small, stop_cam = myCam_small, cumulate_argument = False, animate_over = frms)
splnFrms2 = splnFrms2.rescale(3)
# DecapStream(splnFrms2, 0, 100).show()

all_frames.append(splnFrms2)
script_lines.append(script_line)

### finish up this traj -> not quite done!
script_line = ""
start, stop = stm_p2.stop_frame, stm_p2.stop_frame + 100 

transStm3 = Stream(cap, start, stop - 1)
transFrms = transStm3.decap().frames.rescale(3)

all_frames.append(transFrms)
script_lines.append(script_line)


show(all_frames)

### Check out trajectories for the 9 other examples:
from ATMANpy.types.Abstract import NullPoint

raster_frames = []
for frm_idx, myPnt in enumerate(allPnts[:20]):
    print(frm_idx)

    if isinstance(myPnt, NullPoint):
        stm = Stream.from_SF(segBFs[frm_idx], cap)
        frms = stm.decap().frames
        raster_frames.append(frms)
        continue

    point_splns = myPnt.feat_spline_cumul(which_endpoint = 'penultimate').values()

    ### cull points in the net -> have a nan Point or two.
    ### also cull None

    point_splns = [[spln for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)] for splns in point_splns ]

    scns = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns]) for splns in point_splns]

    stm = myPnt.stream(cap = cap)
    frms = stm.decap().frames[:-1]

    myCam = Camera.calibrated_brest_behind()

    frms_spln, _ = animate_sequence(scns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms)

    raster_frames.append(frms_spln)

for frm_idx, frms in enumerate(raster_frames):
    print(frm_idx)
    DecapStream(frms, 0, 100).show()

DecapStream(raster_frames[2].rescale(3), 0, 100).show(ms_between_frames = 100)

