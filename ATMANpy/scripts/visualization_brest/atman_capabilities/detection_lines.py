
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

seg_idx = 0 ### lines -> index 0
my_segBF = segBFs[seg_idx]

myStm = Stream.from_SF(my_segBF, cap)
myDtm = myStm.decap()
myFrms = myDtm.frames
scaledFrms = myFrms.rescale(3)

fs, ls = myStm.start_frame, myStm.stop_frame

### for the recreation of the line finding steps, we need the extracted steps!
from ATMANpy.scripts.pipeline_brest_tightseg.parameter_files.default import court_lines
prms = court_lines.courtline_params
step_frames = 25
lineSF = myStm.conform(fs, fs + step_frames * 3).recolor('grey').extract(ex.get_courtlines_substeps, **prms)

### pre-frames
dim_factor = 0.5
pre_frames = 180
preFrms = myStm.widen(pre_frames).decap().frames[:pre_frames].rescale(3)
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
                    The Automated Tennis Match Annotator (ATMAN) automatically recognizes the lines, the players, the ball
                    (how fast the ball is, how much spin it has, where it bounces) to be able to summarize a match set by set, game by game,
                    point by point, shot by shot.
                    Match data can be easily combined with other matches that have been visually recognized by ATMAN.
                    The system then allows all matches at hand to be easily searched, filtered and aggregated 
                    to reveal patterns in the style of play of a player, understand how he or she wins matches, or where he or she is vulnerable. 
                    ATMAN also includes a 3D engine that is used to visualize insights from the matches at hand.
                """

titleStr = 'The Automated Tennis Match Annotator (ATMAN)'
titleFrm = blankFrm.put(titleStr, size = 2, where = (600, 540))

titleFrms = BGRFrames.from_frame(titleFrm, STATIC_FRAMES)
all_frames.append(titleFrms)
script_lines.append(script_line)

script_line =   """
                    In this video, we break down how ATMAN does all of these things step-by-step. 
                    To this end, let's take as an example the first point of the match between Richard Gasquet and Thomas Fabbiano.
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
mixFrms = colFrms.mask(maskFrms) + currFrms.mask(~maskFrms) * dim_factor
mixFrms = mixFrms.rescale(3)
# DecapStream(mixFrms, 0, 100).show()

all_frames.append(mixFrms)
script_lines.append(script_line)

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
mixFrms = houghFrms.mask(houghMask) + currFrms.mask(~houghMask) * dim_factor
mixFrms = mixFrms.rescale(3)

all_frames.append(mixFrms)
script_lines.append(script_line)

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
mixFrms = lineFrms.mask(lineMask) + currFrms.mask(~lineMask) * dim_factor
mixFrms = mixFrms.rescale(3)

all_frames.append(mixFrms)
script_lines.append(script_line)

### finally, show the court lines detected 
script_line =   """
                    Having detected the lines, we can easily reconstruct the characteristics of the camera itself:
                    both internal (focal length, sensor dimensions) and external (placement of the camera in space).
                    These quantities coalesce into a projection matrix, which governs how the Camera transforms points in 3-D space into 
                    a 2-D representation on a screen.
                """

start, stop = fs+3*step_frames, ls # -> until the end!
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
