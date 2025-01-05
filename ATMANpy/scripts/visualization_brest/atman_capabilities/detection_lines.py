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

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'lines'
SAVE_FOLDER = 'atman_capabilities'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

#########################################

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

vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
fs, ls = conf['start_frame'], conf['stop_frame']
cap = conf['cap']

segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()

seg_idx = 0 ### lines -> index 0
my_segBF = segBFs[seg_idx]

fs, ls = 0, 750 ### this may be needed in the next video!
myStm = Stream(cap, fs, ls)
myDtm = myStm.decap()
myFrms = myDtm.frames
scaledFrms = myFrms.rescale(3)

### for the recreation of the line finding steps, we need the extracted steps!
step_frames = 50
from ATMANpy.scripts.pipeline_brest_tightseg.parameter_files.default import court_lines
prms = court_lines.courtline_params
lineSF = myStm.conform(fs, fs + step_frames * 3).recolor('grey').extract(ex.get_courtlines_substeps, **prms)

### pre-frames
dim_factor = 0.5
pre_frames = 300
preFrms = myFrms[:pre_frames]
preArr_dimmed = np.stack([f.dim(min((idx / pre_frames) ** 2, dim_factor)).array for idx, f in enumerate(preFrms)], axis = 0)
preFrms_dimmed = BGRFrames(preArr_dimmed)

### Projection Matrix
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

### Cams
brestCam = Camera.calibrated_brest_behind()
brestCam_resized = brestCam.resize(3)

################################
####### BEGIN ANIMATION ########
################################
STATIC_FRAMES = 25
TRANSITION_FRAMES = 100
all_frames = []
script_lines = []

blankFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))

script_line =   """
                    The Automated Tennis Match Annotator (or ATMAN, for short) automatically recognizes the lines, the players, the ball
                    - and how fast the ball is, how much spin it has, where it bounces - to be able to summarize a match set by set, game by game,
                    point by point, shot by shot.
                    Match data can be easily combined with other matches that have been visually recognized by ATMAN.
                    The system then allows all matches at hand to be easily searched, filtered and aggregated 
                    to reveal patterns in the style of play of a player, understand how he or she wins matches, or where he or she is vulnerable. 
                    ATMAN also includes a 3D engine that is used to visualize these patterns and insights.
                """

titleStr = 'The Automated Tennis Match Annotator (ATMAN)'
titleFrm = blankFrm.put(titleStr, size = 2, where = (600, 540))

titleFrms = BGRFrames.from_frame(titleFrm, STATIC_FRAMES)
all_frames.append(titleFrms)
script_lines.append(script_line)

script_line =   """
                    In this video, we break down how ATMAN does all of these things step-by-step. 
                    To this end, let's take as an example the first point of a challenger tournament match between Richard Gasquet and Thomas Fabbiano.
                """
all_frames.append(preFrms_dimmed)
script_lines.append(script_line)

script_line =   """
                    Lines are detected using the simple, but robust probabilistic hough line finding algorithm. 
                """

### recreate the line finding steps
zeroFrm = BGRFrame(np.zeros((360, 640, 3), dtype = 'uint8'))
line_color = const.BGR['rose_orange']

### first, show the white mask
start, stop = fs, fs+step_frames
frmSF = lineSF['allBFrm'][list(range(start, stop))]
maskFrms = MaskFrames.from_single_frames(list(frmSF.values()))
# DecapStream(maskFrms, 0, 100).show()

colFrm = zeroFrm + line_color
colFrms = BGRFrames.from_frame(colFrm, maskFrms.length())

currFrms = myDtm.conform(start, stop - 1).frames
mixFrms0 = colFrms.mask(maskFrms) + currFrms.mask(~maskFrms) * dim_factor
mixFrms0 = mixFrms0.rescale(3)
# DecapStream(mixFrms, 0, 100).show()

### then, show the hough lines
script_line = ""
start, stop = fs+step_frames, fs+2*step_frames
houghSF = lineSF['houghLines'][list(range(start, stop))]
# houghStm = myStm.conform(start, stop).attach(houghSF)
# houghDtm = houghStm.decap_with_features()
# houghFrms = houghDtm.frames
hough_frames = [zeroFrm.put(houghLns, color = line_color) for houghLns in houghSF.values()]
houghFrms = BGRFrames.from_single_frames(hough_frames)
houghMask = houghFrms.to_grey() > 0

currFrms = myDtm.conform(start, stop - 1).frames
mixFrms1 = houghFrms.mask(houghMask) + currFrms.mask(~houghMask) * dim_factor
mixFrms1 = mixFrms1.rescale(3)

### then show the vertical and horizontal filtered lines
script_line = ""
start, stop = fs+2*step_frames, fs+3*step_frames
filtSF = lineSF[['l_v_Lines', 'r_v_Lines', 'h_Lines']][list(range(start, stop))]
# lineStm = myStm.conform(start, stop).attach(lineSF)
# lineDtm = lineStm.decap_with_features()
# lineFrms = lineDtm.frames.rescale(3)
line_frames = [zeroFrm.put(featDict['l_v_Lines'], color = line_color).put(featDict['r_v_Lines'], color = line_color).put(featDict['h_Lines'], color = line_color) for featDict in filtSF.values()]
lineFrms = BGRFrames.from_single_frames(line_frames)
lineMask = lineFrms.to_grey() > 0

currFrms = myDtm.conform(start, stop - 1).frames
mixFrms2 = lineFrms.mask(lineMask) + currFrms.mask(~lineMask) * dim_factor
mixFrms2 = mixFrms2.rescale(3)

mixFrms = BGRFrames.from_frames([mixFrms0, mixFrms1, mixFrms2])

all_frames.append(mixFrms)
script_lines.append(script_line)

### finally, show the court lines detected 
script_line =   """
                    Having detected the lines, we can easily reconstruct the characteristics of the camera itself.
                    A camera is fully described by two matrices.
                """

start, stop = fs+3*step_frames, fs+6*step_frames
HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature
courtLines = sa.find_courtlines(H).drop('Net Line')
courtlineSF = ConstFeature('my_courtLines', courtLines)
# courtStm = myStm.conform(start, stop).attach(courtlineSF)
# courtDtm = courtStm.decap_with_features()
# courtFrms = courtDtm.frames.rescale(3)

court_frames = [zeroFrm.put(courtLines, color = line_color) for _ in range(start, stop)]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

currFrms = myDtm.conform(start, stop - 1).frames
mixFrms = courtFrms.mask(courtMask) + currFrms.mask(~courtMask) * dim_factor
mixFrms = mixFrms.rescale(3)

### internal, external, projection matrices
from ATMANpy.types.Overlays import Table

tab_x, tab_y = 600, 400
def make_matrix_table(matArr, title = None, dimensions = (tab_x, tab_y), horizontal_offset = 4.5):
    rs, cs = matArr.shape
    table_data = {i: {j: matArr[i, j] for j in range(cs)} for i in range(rs)}

    rename_null_dict = {0: '', 1: '', 2: '', 3: ''}

    myTab = Table(table_data, columns_rename_dict = rename_null_dict, index_rename_dict = rename_null_dict,
                  horizontal_lines = False, header_row_height = 0.5, index_col_width = 0, skeleton_shrinkage = 0.25,
                  number_format = '>10.2f', dimensions = dimensions,
                  title = title, title_horizontal_offset = horizontal_offset, title_vertical_offset = 0.5)
    return myTab

cam0 = Camera.calibrated_brest_behind().resize(3)

intTab = make_matrix_table(cam0.M_int, title = 'Internal Matrix', horizontal_offset = 5.5)
extTab = make_matrix_table(cam0.M_ext, title = 'External Matrix', horizontal_offset = 5.5)
prjTab = make_matrix_table(cam0.get_P(), title = 'Projection Matrix', horizontal_offset = 4.5)

intTab.render().show()
extTab.render().show()
prjTab.render().show()

int_coords, ext_coords = (0, 0), (0, tab_y)
intPt = Point2D(int_coords)
extPt = Point2D(ext_coords)

def render_over_frames(myTab, myFrms, tlPt = Point2D((0, 0))):
    from ATMANpy.types.Overlays import overlay

    testFrm = myTab.render()
    rs, cs = testFrm.rows, testFrm.cols
    cx, cy = tlPt.coords

    rendered_frames = []
    for frm in myFrms:
        overArr = frm.array[cy:cy+rs, cx:cx+cs]
        overFrm = myTab.render(over_array = overArr)
        rendered_frames.append(overFrm)
    
    renFrms = BGRFrames.from_single_frames(rendered_frames)
    outFrms = overlay(myFrms, renFrms, top_left_point = tlPt)
    return outFrms

# fade in int. matrix
script_line =   """
                    The internal camera matrix contains all camera intrinsic parameters, like focal length or sensor dimensions.
                """
start, stop = fs+6*step_frames, fs+9*step_frames
HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature
courtLines = sa.find_courtlines(H).drop('Net Line')
courtlineSF = ConstFeature('my_courtLines', courtLines)

court_frames = [zeroFrm.put(courtLines, color = line_color) for _ in range(start, stop)]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

currFrms = myDtm.conform(start, stop - 1).frames
mixFrms = courtFrms.mask(courtMask) + currFrms.mask(~courtMask) * dim_factor
mixFrms = mixFrms.rescale(3)

intFrms = render_over_frames(intTab, mixFrms, tlPt = intPt)
DecapStream(intFrms, 0, 100).show()

# fade in ext. matrix
script_line =   """
                    The external camera matrix contains all camera extrinsic parameters, like the placement and rotation of the camera in space.
                """
start, stop = fs+9*step_frames, fs+13*step_frames
HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature
courtLines = sa.find_courtlines(H).drop('Net Line')
courtlineSF = ConstFeature('my_courtLines', courtLines)

court_frames = [zeroFrm.put(courtLines, color = line_color) for _ in range(start, stop)]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

currFrms = myDtm.conform(start, stop - 1).frames
mixFrms = courtFrms.mask(courtMask) + currFrms.mask(~courtMask) * dim_factor
mixFrms = mixFrms.rescale(3)

intFrms = render_over_frames(intTab, mixFrms, tlPt = intPt)
ext_intFrms = render_over_frames(extTab, intFrms, tlPt = extPt)
DecapStream(ext_intFrms, 0, 100).show()

# fade out int. and ext. matrices, fade in proj matrix
script_line =   """
                    These matrices combine to form a projection matrix, which determines how the camera transforms points in 3-D space into 
                    a 2-D representation on a screen.
                """

prjFrm = render_over_frames(prjTab, mixFrms[[-1]], tlPt = extPt // 2)
transFrm = vs.interpolate_frames(ext_intFrms[-1], prjFrm, num_frames = 100)
DecapStream(transFrm, 0, 100).show()

# show net
script_line =   """
                    Understanding the broadcast camera tremendously eases the visual recognition of all subsequent features.
                    For example, we can immediately reconstruct the location of the net.
                """

from ATMANpy.types.ShapesTennis3D import Net
net_color = const.BGR['salmon_pink']
net_alpha = 0.75
myNet = Net.make_doubles(color = net_color, alpha = net_alpha)
myNet_null = Net.make_doubles(color = net_color, alpha = 0)

netScns = vs.interpolate_scenes(Scene([myNet_null]), Scene([myNet]), num_frames = 100)

netFrms, _ = vs.animate_sequence(netScns, start_cam = cam0, stop_cam = cam0, animate_over = transFrm[-1])
DecapStream(netFrms, 0, 100).show()

script_line =   """
                    Down the line, an understanding of camera determinants will also form the basis of the 3D engine, 
                    with which we can visually represent the court from all angles.
                """

###### -> idea: show the lines rotating in space with a small, arbitrary 2D representation with axes!
###### -> also: show the actual projection matrix on the screen!
from ATMANpy.types.Shapes3D import Point3D, Line3D, Lines3D
from ATMANpy.auxil.court import cnt_2D, courtLines
cnt_3D = Point3D.from_point2D(cnt_2D, z = 0)

# court
courtScn = Scene.make_court(make_net = False, line_color = line_color)
courtLines3D = Lines3D.from_lines2D(courtLines, z = 0, name = 'CourtLines3D')

# corner points
c1 = courtLines3D['Top Baseline'].leftPt
c2 = courtLines3D['Top Baseline'].rightPt
c3 = courtLines3D['Bottom Baseline'].rightPt
c4 = courtLines3D['Bottom Baseline'].leftPt
l, w = 2 * const.SIDE_LENGTH * const.SCALE, const.COURT_WIDTH * const.SCALE

# axes
ax_len = 100
ax_col = const.BGR['strong_blue']
ax_dist = 50

axPt = c1.left(ax_dist).front(ax_dist)

# TODO: maybe make these as arrows?
xLn = Line3D(axPt, axPt.right(ax_len), color = ax_col)
yLn = Line3D(axPt, axPt.back(ax_len), color = ax_col)
zLn = Line3D(axPt, axPt.up(ax_len), color = ax_col)

axScn = Scene([xLn, yLn, zLn])

# court and ax scene
courtaxScn = courtScn.combine(axScn)

# camera checkpoints to interpolate between
z = 100
f = 400
tilt = 0.1

oPt1 = c1.left(w/2).front(l/2).up(z)
cam1 = Camera.from_to(oPt1, cnt_3D, f = f).tilt(tilt)
# View.make(cam1, courtaxScn).snap().show()

oPt2 = c2.right(w/2).front(l/2).up(z)
cam2 = Camera.from_to(oPt2, cnt_3D, f = f).tilt(tilt)
# View.make(cam2, courtaxScn).snap().show()

oPt3 = c3.right(w/2).back(l/2).up(z)
cam3 = Camera.from_to(oPt3, cnt_3D, f = f).tilt(tilt)
# View.make(cam3, courtaxScn).snap().show()

oPt4 = c4.left(w/2).back(l/2).up(z)
cam4 = Camera.from_to(oPt4, cnt_3D, f = f).tilt(tilt)
# View.make(cam4, courtaxScn).snap().show()

oPt2a = cnt_3D.right(5*w/2).up(z)
cam2a = Camera.from_to(oPt2a, cnt_3D, f = f).tilt(tilt)
# View.make(cam2a, courtaxScn).snap().show()

oPt4a = cnt_3D.left(5*w/2).up(z)
cam4a = Camera.from_to(oPt4a, cnt_3D, f = f).tilt(tilt)
# View.make(cam4a, courtaxScn).snap().show()

## animate transition from court to abstract
num_frames_rot = 50
num_frames_trans = 100
arg_res = 0.1
itpl_func = vs.interpolate_camera_naive

## over frame(s)
overFrm = myStm.get(stop).rescale(3)
overFrm_dim = overFrm.dim(0.1)
overFrms_dim = vs.interpolate_frames(overFrm, overFrm_dim, num_frames = num_frames_trans // 2)
overFrms_still = BGRFrames.from_frame(overFrms_dim[-1], num_frames_trans - overFrms_dim.length())
overFrms_trans = BGRFrames.from_frames([overFrms_dim, overFrms_still])

transFrms, _ = vs.animate_sequence([courtScn for _ in range(num_frames_trans)], start_cam = cam0, stop_cam = cam3, interpolate_func = itpl_func,
                                    argument_resolution = arg_res, animate_over = overFrms_trans)
DecapStream(transFrms, 0, 100).show()

transFrms[-1].show()

# View.make(cam1, courtaxScn).snap().show()
# View.make(cam2, courtaxScn).snap().show()
# View.make(cam3, courtaxScn).snap().show()
# View.make(cam4, courtaxScn).snap().show()

## animate rotation in abstract space
def anim_rot(start, stop):
    outFrms, _ = vs.animate_cam_motion(courtaxScn, start_cam = start, stop_cam = stop, interpolate_func = itpl_func,
                                        num_frames = num_frames, argument_resolution = arg_res)

    return outFrms

rotFrms = BGRFrames.from_frames([anim_rot(cam3, cam2a), 
                                 anim_rot(cam2a, cam2),
                                 anim_rot(cam2, cam1),
                                 anim_rot(cam1, cam4a),
                                 anim_rot(cam4a, cam4),
                                 anim_rot(cam4, cam3),
                                 ])
DecapStream(rotFrms, 0, 100).show()

num_rots = 3
rotFrms_rep = BGRFrames.from_frames([rotFrms for _ in range(num_rots)])
DecapStream(rotFrms_rep, 0, 100).show()

animFrms = BGRFrames.from_frames([animFrms0, rotFrms])

DecapStream(animFrms, 0, 100).show(ms_between_frames = 100)


start, stop = fs+3*step_frames, fs+6*step_frames # -> until the end!
HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature
courtLines = sa.find_courtlines(H).drop('Net Line')
courtlineSF = ConstFeature('my_courtLines', courtLines)
# courtStm = myStm.conform(start, stop).attach(courtlineSF)
# courtDtm = courtStm.decap_with_features()
# courtFrms = courtDtm.frames.rescale(3)

court_frames = [zeroFrm.put(courtLines, color = line_color) for _ in range(start, stop)]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

currFrms = myDtm.conform(start, stop - 1).frames
mixFrms = courtFrms.mask(courtMask) + currFrms.mask(~courtMask) * dim_factor
mixFrms = mixFrms.rescale(3)

all_frames.append(mixFrms)
script_lines.append(script_line)

### the net -> a trivial matter once the lines are found.
# -> add it! 
# (maybe during the intermezzo between the two points, so that we can show how we find the players seamlessly, during the next point)

script_line =   """
                    Armed with the projection matrix, the location of the net can be easily reconstructed.
                """
net_color = const.BGR['salmon_pink']
net_alpha = 0.3

start = ls
stop, _ = segBFs[1].first()

currFrms = Stream(cap, start, stop - 1).decap().frames
currFrms = vis_ut.speed_up_frames_progressively(currFrms)

court_frames = [zeroFrm.put(courtLines, color = line_color) for _ in currFrms]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

mixFrms = courtFrms.mask(courtMask) + currFrms.mask(~courtMask) * dim_factor

from ATMANpy.types.ShapesTennis3D import Net
myNet = Net.make_doubles(color = net_color, alpha = net_alpha)

netFrms, _ = vs.animate_sequence([Scene([myNet]) for _ in mixFrms], start_cam = brestCam, stop_cam = brestCam, animate_over = mixFrms)
netFrms = netFrms.rescale(3)
all_frames.append(netFrms)
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
