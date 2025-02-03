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

DESCRIPTOR = 'players'
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

seg_idx = 1 ### players -> index 1
my_segBF = segBFs[seg_idx]

dim_factor = 0.5

### recreate the line finding steps
zeroFrm = BGRFrame(np.zeros((360, 640, 3), dtype = 'uint8'))
line_color = const.BGR['rose_orange']

### still need court lines ...
HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature
courtLines = sa.find_courtlines(H).drop('Net Line')

### ... and the net
from ATMANpy.types.ShapesTennis3D import Net
net_color = const.BGR['salmon_pink']
net_alpha = 0.75
net_mesh_density = 0.65
myNet = Net.make_doubles(color = net_color, alpha = net_alpha, net_mesh_density = net_mesh_density)

### Cams
brestCam = Camera.calibrated_brest_behind()
brestCam_resized = brestCam.resize(3)

###################
##### PLAYERS #####
###################

all_frames = []
script_lines = []

### Read median and standard deviation background
medStms = StreamList.from_store('median_backFrm', vid, sto, target_color_space = 'hsv')
stdStms = StreamList.from_store('std_backFrm', vid, sto, target_color_space = 'hsv')

### burn-in mask, and shadow mask
burnStms = StreamList.from_store('foregroundMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')
shadStms = StreamList.from_store('shadowMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')

### foreground contours, and player contours (tracked)
foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])

my_segBF = segBFs[1] ### on to the next point!

### fade out the lines!
script_line =   """
                    ATMAN first synthesizes a background image; a still frame of the court only, 
                    against which we can easily detect foreground objects.
                """
duration = 50

### simple, no line frames
start, stop = my_segBF.first()[0], my_segBF.first()[0] + duration 
currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames

### frames with lines and net.
court_frames = [zeroFrm.put(courtLines, color = line_color) for _ in currFrms]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

mixFrms = courtFrms.mask(courtMask) + currFrms.mask(~courtMask) * dim_factor
currFrms = currFrms * dim_factor

netFrms, _ = vs.animate_sequence([Scene([myNet]) for _ in mixFrms], start_cam = brestCam_resized, stop_cam = brestCam_resized, animate_over = mixFrms.rescale(3))

lambdas = np.linspace(0, 1, duration)[:, np.newaxis, np.newaxis, np.newaxis]
backFrms0 = currFrms.rescale(3) * lambdas + netFrms * (1 - lambdas)
DecapStream(backFrms0, 0, 100).show()

### show the removal of foreground objects (by fading between the static background image and the action)

duration = 100
start, stop = stop, stop + duration 
currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames
currFrms = currFrms * dim_factor

medFrms = BGRFrames.from_frame(medStms[1].recolor('bgr').stillFrm, duration)
medFrms = medFrms * dim_factor

lambdas = np.linspace(0, 1, duration)[:, np.newaxis, np.newaxis, np.newaxis]
backFrms1 = medFrms * lambdas + currFrms * (1 - lambdas)
backFrms1 = backFrms1.rescale(3)

duration = 150
medFrms = BGRFrames.from_frame(medStms[1].recolor('bgr').stillFrm, duration)
backFrms2 = medFrms * dim_factor
backFrms2 = backFrms2.rescale(3)

backFrms = BGRFrames.from_frames([backFrms0, backFrms1, backFrms2])

all_frames.append(backFrms)
script_lines.append(script_line)

### fade players back in
script_line =   """
                    The players, the ball kids and the line judges are then clearly visibly against this background.
                """
duration = 100
start, stop = stop, stop + duration 
currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames

medFrms = BGRFrames.from_frame(medStms[1].recolor('bgr').stillFrm, duration)
medFrms = medFrms * dim_factor

burnMask = burnStms[1].conform(start, stop - 1).decap().frames
mixFrms = currFrms.mask(burnMask) + currFrms.mask(~burnMask) * dim_factor

lambdas = np.linspace(0, 1, duration)[:, np.newaxis, np.newaxis, np.newaxis]
plFrms0 = mixFrms * lambdas + medFrms * (1 - lambdas)
plFrms0 = plFrms0.rescale(3)

duration = 100
start, stop = stop, stop + duration 
currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames

burnMask = burnStms[1].conform(start, stop - 1).decap().frames
mixFrms = currFrms.mask(burnMask) + currFrms.mask(~burnMask) * dim_factor

plFrms1 = mixFrms.rescale(3)

plFrms = BGRFrames.from_frames([plFrms0, plFrms1])

all_frames.append(plFrms)
script_lines.append(script_line)

### now, show how we use this to find the players
script_line =   """
                    Notice how the player's shadows are still detected as foreground.
                """
duration = 2
start, stop = stop, stop + duration 
currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames

burnMask = burnStms[1].conform(start, stop - 1).decap().frames

shadFrms0 = currFrms.mask(burnMask) + currFrms.mask(~burnMask) * dim_factor
shadFrms0 = shadFrms0.rescale(3)

duration = 200

currFrm = currStm.get(stop - 1)
shadMask = shadStms[1].get(stop - 1)
burnMask = burnStms[1].get(stop - 1)

mixFrm = currFrm.mask(burnMask) + BGRFrame((currFrm.mask(~burnMask) * dim_factor).array.astype('uint8'))

remove_shadMask = ~shadMask & burnMask

mixFrm_noshad = currFrm.mask(remove_shadMask) + BGRFrame((currFrm.mask(~remove_shadMask) * dim_factor).array.astype('uint8'))

shadFrms1 = vs.blinking(mixFrm, mixFrm_noshad, duration_in_frames = duration, blinks_per_second = 2)
shadFrms1 = shadFrms1.rescale(3)

shadFrms = BGRFrames.from_frames([shadFrms0, shadFrms1])

all_frames.append(shadFrms)
script_lines.append(script_line)

script_line = "These shadows are automatically removed by ATMAN."

duration = 110
start, stop = stop, stop + duration 
currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames

shadMask = shadStms[1].conform(start, stop - 1).decap().frames
burnMask = burnStms[1].conform(start, stop - 1).decap().frames
joinMask = burnMask & ~shadMask

mixFrms = currFrms.mask(joinMask) + currFrms.mask(~joinMask) * dim_factor
mixFrms = mixFrms.rescale(3)

all_frames.append(mixFrms)
script_lines.append(script_line)

# DecapStream(mixFrms, 0, 100).show()

script_line = "The remaining contours are our player candidates..."

player_color = const.BGR['rose_orange']
duration = 110
start, stop = stop, stop + duration
my_foreSF = foreSF[list(range(start,stop))]

currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames

shadMask = shadStms[1].conform(start, stop - 1).decap().frames
burnMask = burnStms[1].conform(start, stop - 1).decap().frames
joinMask = burnMask & ~shadMask

# mixFrms = currFrms.mask(joinMask) + currFrms.mask(~joinMask) * dim_factor
mixFrms = currFrms * dim_factor

mixFrms_with_contours = BGRFrames.from_single_frames([mixFrm.put(myCnts, color = player_color, thickness = 2) for mixFrm, myCnts in zip(mixFrms, my_foreSF.values())])
mixFrms_with_contours = mixFrms_with_contours.rescale(3)

all_frames.append(mixFrms_with_contours)
script_lines.append(script_line)

# DecapStream(mixFrms_with_contours, 0, 100).show()

script_line = "... from which the players can easily be found and tracked."

player_color = const.BGR['strong_pink']
# duration = 100
my_plySF = plySF.split()[1]

start, stop = stop, my_plySF.last()[0] ### play until the end.
my_plySF = plySF[list(range(start,stop))]

currStm = Stream(cap, start, stop - 1)
currFrms = currStm.decap().frames

shadMask = shadStms[1].conform(start, stop - 1).decap().frames
burnMask = burnStms[1].conform(start, stop - 1).decap().frames
joinMask = burnMask & ~shadMask

# mixFrms = currFrms.mask(joinMask) + currFrms.mask(~joinMask) * dim_factor
mixFrms = currFrms * dim_factor

mixFrms_with_contours = BGRFrames.from_single_frames([mixFrm.put(featDict['topPlayerCnt'], color = player_color, thickness = 2).put(featDict['botPlayerCnt'], color = player_color, thickness = 2) for mixFrm, featDict in zip(mixFrms, my_plySF.values())])
mixFrms_with_contours = mixFrms_with_contours.rescale(3)

all_frames.append(mixFrms_with_contours)
script_lines.append(script_line)

### intermezzo -> introduce that we're about to look for the ball.
script_line = "Let's take a look at how ATMAN goes about finding the ball."

motiStms = StreamList.from_store('fine_motMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 
start, stop = stop, motiStms[2].start_frame

interFrms = Stream(cap, start, stop).decap().frames
interFrms = interFrms * dim_factor
interFrms = vis_ut.speed_up_frames_progressively(interFrms)
interFrms = interFrms.rescale(3)

all_frames.append(interFrms)
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
