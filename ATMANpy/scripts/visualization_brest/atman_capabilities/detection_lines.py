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

fs, ls = 200, 750 ### this may be needed in the next video!
myStm = Stream(cap, fs, ls)
myDtm = myStm.decap()
myFrms = myDtm.frames
scaledFrms = myFrms.rescale(3)

### pre-frames
dim_factor = 0.5
pre_frames = 300
preFrms = myFrms[:pre_frames]
preArr_dimmed = np.stack([f.dim(min((idx / pre_frames) ** 2, dim_factor)).array for idx, f in enumerate(preFrms)], axis = 0)
preFrms_dimmed = BGRFrames(preArr_dimmed).rescale(3)

fs_adj = fs + preFrms_dimmed.length()

### for the recreation of the line finding steps, we need the extracted steps!
step_frames = 25
from ATMANpy.scripts.pipeline_brest_tightseg.parameter_files.default import court_lines
prms = court_lines.courtline_params
lineSF = myStm.conform(fs_adj, fs_adj + step_frames * 3).recolor('grey').extract(ex.get_courtlines_substeps, **prms)

### Projection Matrix
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

### Cams
brestCam = Camera.calibrated_brest_behind()
brestCam_resized = brestCam.resize(3)


#####################################
#### ATMAN contraction animation ####
#####################################

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

pre_staticFrm = BGRFrames.from_frame(titleFrm, 25)

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

introFrms = BGRFrames.from_frames([pre_staticFrm, pre_dimFrms_lettered, post_dimFrms_lettered])

##### fade out letters, fade in play
fade_in_length = 200

post_dimFrms_fade_out = BGRFrames.from_frame(post_dimFrms_lettered[-1], fade_in_length).dim_progressively(1, 0)
preFrms_fade_in = preFrms_dimmed[:fade_in_length].dim_progressively(0, 1)

preFrms_fade = post_dimFrms_fade_out + preFrms_fade_in
# DecapStream(preFrms_fade, 0, 100).show()

preFrms_rest = preFrms_dimmed[fade_in_length:]

# DecapStream(introFrms, 0, 100).show()
#########################
####### BEGIN ANIMATION ########
################################
STATIC_FRAMES = 25
TRANSITION_FRAMES = 100
all_frames = []
script_lines = []

blankFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))

script_line =   """
                    The Automated Tennis Match Annotator, or ATMAN for short,
                    is a start-to-end framework for the generation of insights
                    from tennis matches. 

                    ATMAN generates detailed shot-by-shot data directly from match footage, 
                    allows all matches at hand to be easily searched, filtered and aggregated, 
                    and lets the user seamlessly visualize results in a 3D engine.
                """

# script_line =   """
#                     The Automated Tennis Match Annotator (or ATMAN, for short) automatically recognizes the lines, the players, the ball
#                     - and how fast the ball is, how much spin it has, where it bounces - to be able to summarize a match set by set, game by game,
#                     point by point, shot by shot.
#                     Match data can be easily combined with other matches that have been visually recognized by ATMAN.
#                     The system then allows all matches at hand to be easily searched, filtered and aggregated 
#                     to reveal patterns in the style of play of a player, understand how he or she wins matches, or where he or she is vulnerable. 
#                     ATMAN also includes a 3D engine that is used to visualize these patterns and insights.
#                 """

# titleStr = 'The Automated Tennis Match Annotator (ATMAN)'
# titleFrm = blankFrm.put(titleStr, size = 2, where = (600, 540))
# titleFrms = BGRFrames.from_frame(titleFrm, STATIC_FRAMES)

all_frames.append(introFrms)
script_lines.append(script_line)

script_line =   """
                    In this video, we break down how ATMAN does all of these things step-by-step. 
                """ # 5 seconds
all_frames.append(preFrms_fade)
script_lines.append(script_line)
#######

script_line =   """
                    Lines are detected using the probabilistic hough line finding algorithm. 
                """ # 4 seconds

### recreate the line finding steps
zeroFrm = BGRFrame(np.zeros((360, 640, 3), dtype = 'uint8'))
line_color = const.BGR['rose_orange']

### first, show the white mask
start, stop = fs_adj, fs_adj+step_frames
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
start, stop = fs_adj+step_frames, fs_adj+2*step_frames
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
start, stop = fs_adj+2*step_frames, fs_adj+3*step_frames
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

mixFrms = BGRFrames.from_frames([preFrms_rest, mixFrms0, mixFrms1, mixFrms2])

all_frames.append(mixFrms)
script_lines.append(script_line)

### finally, show the court lines detected 
script_line =   """
                    Having detected the lines, we can immediately reconstruct the location of the net.
                """ # 5 secs

start, stop = fs_adj+3*step_frames, fs_adj+3*step_frames+150
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

from ATMANpy.types.ShapesTennis3D import Net
net_color = const.BGR['salmon_pink']
net_alpha = 0.75
net_mesh_density = 0.65
myNet = Net.make_doubles(color = net_color, alpha = net_alpha, net_mesh_density = net_mesh_density)
myNet_null = Net.make_doubles(color = net_color, alpha = 0, net_mesh_density = net_mesh_density)

netScns = vs.interpolate_scenes(Scene([myNet_null]), Scene([myNet]), num_frames = 100)

netScns = [*[myNet_null] * 50, *netScns]

cam0 = Camera.calibrated_brest_behind().resize(3)
netFrms, _ = vs.animate_sequence(netScns, start_cam = cam0, stop_cam = cam0, animate_over = mixFrms)
# DecapStream(netFrms, 0, 100).show()

all_frames.append(netFrms)
script_lines.append(script_line)

### rotate back to starting position, in time for detection_player to start.
script_line = "Let's see how ATMAN detects the players on the field." # 4 secs

start, stop = stop, segBFs[1].first()[0]

courtLines = sa.find_courtlines(H).drop('Net Line')
court_frames = [zeroFrm.put(courtLines, color = line_color) for _ in range(start, stop)]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

currFrms = Stream(cap, start, stop - 1).decap().frames
mixFrms = courtFrms.mask(courtMask) + currFrms.mask(~courtMask) * dim_factor
mixFrms = mixFrms.rescale(3)

res = 1
mixFrms, _ = vs.animate_sequence([netScns[-1] for _ in range(currFrms.length())], start_cam = cam0, stop_cam = cam0, animate_over = mixFrms, 
                                 argument_resolution = res, background_resolution = res)

all_frames.append(mixFrms)
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
