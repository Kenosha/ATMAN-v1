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

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D

from ATMANpy.types.Abstract import NullPoint, Points, Shots, read_in_data_tables, read_in_placeholder_tables, read_in_colinear_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

from ATMANpy.types.ShapesTennis3D import Arrow
from ATMANpy.types.Shapes3D import Circle3D

from ATMANpy.models.ParticleFilter import ParticleFilter
from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points

from ATMANpy.types.Overlays import typewriter_effect
import cv2 as cv

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'recognition'
SAVE_FOLDER = 'website_snippets'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

from ATMANpy.funcs.util_funcs import make_timestamp, maybe_make_folder
saveFolder = DESCRIPTOR + '_' + str(make_timestamp())
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
cap = confs[0]['cap']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

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

#### features

cntSF = StreamFeature.from_store(vid, sto, featname = 'movCnts')
shadStms = StreamList.from_store('shadowMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')

HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature

plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])

cntSF = StreamFeature.from_store(vid, sto, featname = 'movCnts')
cndSF = StreamFeature.from_store(vid, sto, featname = 'cand2Cnts') 
pntSF = StreamFeature.from_store(vid, sto, featname = 'ball_choicePt') 
hitSF = StreamFeature.from_store(vid, sto, featname = 'hitCnt') 

hitcntSF = StreamFeature.from_store(vid, sto, featname = 'hitCnt')
hitflagSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')

segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()

##### IDEA
# court (pre serve)
# players (pre serve)
# serve recognized as stroke
# ball constituents: SVM -> Particles (Kitagawa) -> Ball Shape
# Trajectory (with frozen player?, some shot data?)

### serve recognition (and maybe return recognition) can be combined with SVM
# -> this way, we have a bunch of percentages on screen for two shots, one flight.
# -> then, on the return flight, we go to ball shape (recognized)
# -> then, as the serve + 1 is hit, we freeze on the player, 
#           display the shot info, and show the trajectory as it develops. 


# TODO:
# text closer to left edge
# collecting ball candidates could be a bit faster
# smoothing ball candidates could be a bit faster
# speed changes in the ball part a bit too hectic
# ball tracking a bit too fast, smoothing ball tracking a bit too slow
# transition is too fast at the end.

mtc_idx, pnt_idx = 0, 111 ### the first point of atman short intro video
myPnt = allPnts.get(mtc_idx, pnt_idx)
seg_idx = 112

all_frames = []

##### NOTES
# court, net and players have to be breathing in and out of the thing with a fade
initial_frames = 100
dim_factor = 0.25
myStm = myPnt.stream(confs = confs)
myDtm = myStm.widen(initial_frames).decap()
stream_start = myDtm.start_frame


blackFrm = BGRFrame.make_black(640, 360)

line_color = const.BGR['rose_orange']
net_color = const.BGR['salmon_pink']
player_color = const.BGR['strong_pink']

cand_color = const.BGR['fair_yellow']
kita_color = const.BGR['soft_yellow']
ball_color = const.BGR['ball_yellow']

type_kwargs = {'frames_per_char': 1, 'text_where': (15, 15), 'text_size': 1, 
               'text_font': cv.FONT_HERSHEY_PLAIN, 'text_color': const.BGR['leafy_green']}

first_trajectory_shot_idx = -2
fsShot = myPnt[first_trajectory_shot_idx]

def typewriter_with_ellipses(textStr, overFrms):
    ellipses_slower_frames = 3

    current_ellipsis = ' '
    textStr_with_ell = textStr + current_ellipsis
    frames_typed = len(textStr_with_ell) + 2
    
    overFrms_typed = overFrms[:frames_typed]
    overFrms_cycle = overFrms[frames_typed:]

    typedFrms = typewriter_effect(textStr_with_ell, overFrm = overFrms_typed, **type_kwargs)
    
    fs = 0
    cycle_frames = []
    while fs < overFrms_cycle.length():
        ls = fs + type_kwargs['frames_per_char'] + ellipses_slower_frames
        
        cycleFrms = overFrms_cycle[fs:ls].put('\n' + textStr_with_ell, 
                                            where = type_kwargs['text_where'],
                                            size = type_kwargs['text_size'], 
                                            color = type_kwargs['text_color'],
                                            font = type_kwargs['text_font'])
        cycle_frames.append(cycleFrms)

        if current_ellipsis.endswith('...'):
            current_ellipsis = ' '
        else:
            current_ellipsis += '.'
    
        textStr_with_ell = textStr + current_ellipsis

        fs = ls

    cycleFrms = BGRFrames.from_frames(cycle_frames)

    jointFrms = typedFrms.join(cycleFrms)
    return jointFrms

overlap_frames_curr = 0
overlap_frames_next = 10

### LINES ###
text = 'Detecting Court'

sec_frms = 75
brt_frms = sec_frms // 3
start, stop = stream_start, stream_start + sec_frms

courtLines = sa.find_courtlines(H).drop('Net Line')
courtlineSF = ConstFeature('my_courtLines', courtLines)
# courtStm = myStm.conform(start, stop).attach(courtlineSF)
# courtDtm = courtStm.decap_with_features()
# courtFrms = courtDtm.frames.rescale(3)

court_frames = [blackFrm.put(courtLines, color = line_color) for _ in range(start, stop)]
courtFrms = BGRFrames.from_single_frames(court_frames)
courtMask = courtFrms.to_grey() > 1

currFrms = myDtm.conform(start, stop - 1).frames
currFrms_in, currFrms_hold, currFrms_out = currFrms[:brt_frms], currFrms[brt_frms:2*brt_frms], currFrms[2*brt_frms:3*brt_frms]
courtFrms_in, courtFrms_hold, courtFrms_out = courtFrms[:brt_frms], courtFrms[brt_frms:2*brt_frms], courtFrms[2*brt_frms:3*brt_frms]
# courtMask_in, courtMask_hold, courtMask_out = courtMask[:brt_frms], courtMask[brt_frms:2*brt_frms], courtMask[2*brt_frms:3*brt_frms]

mixFrms_in = courtFrms_in.dim_progressively(0, 1) + currFrms_in.dim_progressively(1, dim_factor)
mixFrms_hold = courtFrms_hold + currFrms_hold.dim(dim_factor)
# mixFrms_out = courtFrms_out.dim_progressively(1, 0) + currFrms_out.dim_progressively(dim_factor, 1)
mixFrms_out = courtFrms_out + currFrms_out.dim(dim_factor)

mixFrms = BGRFrames.from_frames([mixFrms_in, mixFrms_hold, mixFrms_out])

# mixFrms = mixFrms.rescale(3)
# DecapStream(mixFrms, 0, 100).show()

### NET ####
# start, stop = stop, stop + sec_frms
from ATMANpy.types.ShapesTennis3D import Net

net_alpha = 0.75
net_mesh_density = 0.65
myNet = Net.make_doubles(color = net_color, alpha = net_alpha, net_mesh_density = net_mesh_density)
myNet_null = Net.make_doubles(color = net_color, alpha = 0, net_mesh_density = net_mesh_density)

netScns = vs.interpolate_scenes(Scene([myNet_null]), Scene([myNet]), num_frames = brt_frms)

netScns = [*netScns, *[Scene([myNet])] * brt_frms, *list(reversed(netScns))]

# currFrms = myDtm.conform(start, stop - 1).frames
# currFrms_in, currFrms_hold, currFrms_out = currFrms[:brt_frms], currFrms[brt_frms:2*brt_frms], currFrms[2*brt_frms:3*brt_frms]
# currFrms = BGRFrames.from_frames([currFrms_in.dim_progressively(1, dim_factor),
#                                   currFrms_hold.dim(dim_factor),
#                                   currFrms_out.dim(dim_factor)])

cam0 = Camera.calibrated_brest_behind()
netFrms, _ = vs.animate_sequence(netScns, start_cam = cam0, stop_cam = cam0, animate_over = mixFrms)
# DecapStream(netFrms, 0, 100).show()

netFrms_ovl1 = netFrms[:overlap_frames_curr]
netFrms_main = netFrms[overlap_frames_curr:-overlap_frames_next]
netFrms_ovl2 = netFrms[-overlap_frames_next:]

netFrms = typewriter_with_ellipses(text, BGRFrames.from_frames([netFrms_ovl1, netFrms_main]))
DecapStream(netFrms, 0, 100).show()

all_frames.append(netFrms)

### PLAYERS ###
text = 'Detecting Players'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 10

sec_frms = 60
in_cumul_share, hold_cumul_share, out_cumul_share = 0.1, 0.6, 1
in_frms = int(in_cumul_share * sec_frms)
hold_frms = int(hold_cumul_share * sec_frms) 
out_frms = int(out_cumul_share * sec_frms)

start, stop = stop - overlap_frames_curr, stop + sec_frms

currStm = Stream(cap, start, stop - 1)
blnkFrms = currStm.decap().frames.dim(dim_factor)
my_plySF = plySF.conform(list(range(start, stop)))

ply_frames = [blnkFrm.put(list(featDict.values()), color = player_color, thickness = 2) for blnkFrm, featDict in zip(blnkFrms, my_plySF.values())]
plyFrms = BGRFrames.from_single_frames(ply_frames)

plyFrms_ovl1 = plyFrms[:overlap_frames_curr]
plyFrms_main = plyFrms[overlap_frames_curr:-overlap_frames_next]
plyFrms_ovl2 = plyFrms[-overlap_frames_next:]

plyFrms_ovl = netFrms_ovl2.dim_progressively(1, 0) + plyFrms_ovl1.dim_progressively(0, 1) 

plyFrms = typewriter_with_ellipses(text, BGRFrames.from_frames([plyFrms_ovl, plyFrms_main]))
DecapStream(plyFrms, 0, 100).show()

all_frames.append(plyFrms)

vis_ut.show(all_frames)

#### BALL ####
text = 'Collecting Ball Candidates'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 5

### the candidate contours
sec_frms = 65 ### TODO: needs to be longer! 8 seconds (plus slight adjustments to make SVM part look sensible) --> done with slow!
start, stop = stop - overlap_frames_curr, stop + sec_frms

cnt_vals = cntSF.conform(list(range(start, stop))).values()

currFrms = Stream(cap, start, stop - 1).decap().frames
currFrms_with_contours = BGRFrames.from_single_frames([currFrm.put(candCnts, color = cand_color, thickness = 2) for currFrm, candCnts in zip(currFrms, cnt_vals)])

cntFrms = BGRFrames.from_single_frames([blackFrm.put(cnts, thickness = 2) for cnts in cnt_vals])
cntMask = cntFrms.to_grey() > 0
shadMask = shadStms[seg_idx].conform(start, stop - 1).decap().frames
joinMask = cntMask & ~shadMask

mixFrms = currFrms_with_contours.mask(joinMask) + currFrms.mask(~joinMask) * dim_factor
# DecapStream(mixFrms, 0, 100).show()

mixFrms_ovl1 = mixFrms[:overlap_frames_curr]
mixFrms_main = mixFrms[overlap_frames_curr:-overlap_frames_next]
mixFrms_ovl2 = mixFrms[-overlap_frames_next:]

mixFrms_ovl = plyFrms_ovl2.dim_progressively(1, 0) + mixFrms_ovl1.dim_progressively(0, 1) 

mixFrms = BGRFrames.from_frames([mixFrms_ovl, mixFrms_main])

mixFrms = typewriter_with_ellipses(text, mixFrms)
DecapStream(mixFrms, 0, 100).show()

all_frames.append(mixFrms)

text = 'Evaluating Ball Candidates'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 5

sec_frms = 15
start, stop = stop - overlap_frames_curr, stop + sec_frms

sub_cntSF = cntSF.conform(list(range(start, stop)))

hStm = Stream.from_SF(sub_cntSF, cap, target_color_space = 'hsv')
gStm = Stream.from_SF(sub_cntSF, cap, target_color_space = 'grey')

# create svm features
svm_params = prm_mod.ball_inference_svm.svm_params    
from ATMANpy.models.BallSVM.utils import load_svm
svmDict = load_svm(svm_params['svm_name'])

frmSF = MulStream([hStm, gStm]).attach(sub_cntSF).extract(ex.get_feature_frames, **svmDict['feature_params'])

# calculate ball probablitities
inference_params = prm_mod.ball_inference_svm.inference_params
ballTable = frmSF['ballTable'].to_frame()
svm = svmDict['svm']
features = svmDict['features']
mean = svmDict['mean']
std = svmDict['std']
normTable = (ballTable - mean) / std
normTable['isBall_proba'] = svm.predict_proba(normTable[features].values)[:,1]

from ATMANpy.funcs.util_funcs import interpolate_tuples

# annotate each frame
currStm = Stream(cap, start, stop)
sub_shadStm = shadStms[seg_idx].conform(start, stop) ### (so we can filter out the contours in the shadow...)

hold_for = 1 # 25
out_frames = []
for i in range(sec_frms + overlap_frames_curr):
    cnts = sub_cntSF[start + i]
    probs = normTable.loc[start + i, 'isBall_proba']

    currFrm_undimmed = currStm.get(start + i)
    currFrm = currFrm_undimmed.dim(dim_factor)
    shadFrm = sub_shadStm.get(start + i)
    regMask = shadFrm.zero()

    x_offset = 10
    y_offset = 5

    pStrs = [] ### this is so we can save the probability strings and their location to put them onto the resized frame
    for cnt, prb in zip(cnts, probs):
        cntPnt = cnt.center()
        
        ### check if the contour is in the shadow.
        if shadFrm.eval_at(cntPnt) == 255:
            continue
        
        pStr = f'{prb:.0%}'
        x, y = cntPnt.coords
        x, y = x + x_offset, y + y_offset
        reg = cnt.enclosing_region(margin = 3)

        regMask = regMask | regMask.within_region(reg)
        # currFrm = currFrm.put(reg, color = ball_color, thickness = 2).put(pStr, where = (x, y))
        # currFrm = currFrm.put(pStr, where = (x, y))

        one_color = const.BGR['grass_green']
        zero_color = const.BGR['intense_red']
        text_color = interpolate_tuples(zero_color, one_color, itpl_lambda = prb)

        pStrs.append((pStr, (x, y), text_color))

    mixFrm = currFrm_undimmed.mask(regMask) + currFrm.mask(~regMask)

    mixFrm = mixFrm

    for pStr, where, c in pStrs:
        x, y = where
        mixFrm = mixFrm.put(pStr, where = (x, y), size = 1, color = c)

    mixFrms = BGRFrames.from_frame(mixFrm, hold_for)

    out_frames.append(mixFrms)

probFrms = BGRFrames.from_frames(out_frames)

probFrms_ovl1 = probFrms[:overlap_frames_curr]
probFrms_main = probFrms[overlap_frames_curr:-overlap_frames_next]
probFrms_ovl2 = probFrms[-overlap_frames_next:]

probFrms_ovl = mixFrms_ovl2.slow(2).dim_progressively(1, 0) + probFrms_ovl1.slow(2).dim_progressively(0, 1) 

probFrms = BGRFrames.from_frames([probFrms_ovl, probFrms_main.slow(2).slow(2).slow(2)])

probFrms = typewriter_with_ellipses(text, probFrms)
DecapStream(probFrms, 0, 100).show()

all_frames.append(probFrms)
# DecapStream(probFrms, 0, 100).show()

vis_ut.show(all_frames)

### PARTICLE FILTER ###
text = 'Tracking Ball'

### Combine ball contours with hypothesized hit contours
cntplusSF = cndSF.rename('candCnts').merge(hitcntSF, how = 'outer').apply(sa.merge_contour_into_contours)

hitcntSFs = hitcntSF.conform(segBF.filter(segBF)).split()
hitflagSFs = hitflagSF.conform(segBF.filter(segBF)).split()
cntSFs_high = cndSF.rename('candCnts').conform(segBF.filter(segBF)).split()
cntplusSFs = cntplusSF.conform(segBF.filter(segBF)).split()

### Create model switching points at and around hit points
hitSFs = hitflagSFs.dropna().thin().merge(hitcntSFs) ### thin out hitflags so we have one flag for each hit exactly.

hitSF = hitSFs[seg_idx-1]

propagation_params = prm_mod.ball_inference_particles.propagation_params
filter_init_params = prm_mod.ball_inference_particles.filter_init_params
filter_run_params = prm_mod.ball_inference_particles.filter_run_params

bSF = sf.create_propagation_models(hitSF, courtLines = courtLines)
bSF = sf.widen_propagation_models(bSF, **propagation_params)

cntplusSF = cntplusSFs[seg_idx-1]
hitSF = hitSFs[seg_idx-1]

inSF = cntplusSF.rename('candCnts').merge(bSF).merge(hitSF)

myFilter = ParticleFilter(**filter_init_params)

base_length = courtLines['Bottom Baseline'].length() ### take the larger of the two (bot side) so that our maximum ratio is larger!
side_length = courtLines['Bottom Singles Ad Line'].length() + courtLines['Top Singles Deuce Line'].length()
    
width, height = 640, 360
filter_run_params_additional = {'frame_width': width,
                                    'frame_height': height,
                                    'pixel_per_meter_horizontal': base_length / const.COURT_WIDTH,
                                    'pixel_per_meter_vertical': side_length / const.SIDE_LENGTH}

partSF = myFilter.find_particles(inSF, **filter_run_params, **filter_run_params_additional)
kitaSF = sf.kitagawa_smooth(partSF)

### some parameters
N = 100 # how many particles to show
scale = (10, 10, 1, 1) ### standard deviation for kitagawa perturbation

# sec_frms = 25
# start, stop = stop - overlap_frames_curr, stop + sec_frms

# subStm = Stream(cap, start, stop - 1)
# subFrms = subStm.decap().frames * dim_factor
# # subFrms_slow = subFrms.slow(2).slow(2).slow(2)

# my_partSF = partSF['particles'].conform(list(range(start, stop)))
# partArrs = np.array(list(my_partSF.values()))
# # partArrs_slow = np.repeat(partArrs, repeats = 8, axis = 0)

# part_frames = [subFrm.put(particles_to_points(partArr, N), color = part_color, size = 0) for subFrm, partArr in zip(subFrms, partArrs)]
# partFrms = BGRFrames.from_single_frames(part_frames)

# partFrms_ovl1 = partFrms[:overlap_frames_curr]
# partFrms_main = partFrms[overlap_frames_curr:-overlap_frames_next]
# partFrms_ovl2 = partFrms[-overlap_frames_next:]

# partFrms_ovl = probFrms_ovl2.slow(2).slow(2).dim_progressively(1, 0) + partFrms_ovl1.slow(2).slow(2).dim_progressively(0, 1) 

# partFrms = BGRFrames.from_frames([partFrms_ovl, partFrms_main.slow(2)])

# partFrms = typewriter_with_ellipses(text, partFrms)
# DecapStream(partFrms, 0, 100).show()

# all_frames.append(partFrms)

### KITAGAWA SMOOTHING ###
# text = 'Smoothing Ball Tracking'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 10

sec_frms = 55 
start, stop = stop - overlap_frames_curr, stop + sec_frms

subStm = Stream(cap, start, stop - 1)
subFrms = subStm.decap().frames * dim_factor
# subFrms_slow = subFrms.slow(2).slow(2)

my_partSF = kitaSF.conform(list(range(start, stop)))
partArrs = np.array(list(my_partSF.values()))
# partArrs_slow = np.repeat(partArrs, repeats = 4, axis = 0)

### perturb the kitagawa particles a bit. 
# (even though this does not happen in the code, it would yield the same results and looks better on screen.)
partArr = partArrs[0]
def perturb_array(partArr, N = 100, scale = (5, 5, 1, 1)):
    uniques = np.unique(partArr, axis = 0)

    randArrs = []
    for _ in range(N):
        idx = np.random.choice(len(uniques))
        randArr = np.random.normal(uniques[idx], scale = scale)
        randArrs.append(randArr)

    return np.array(randArrs)

# scale = (10, 10, 1, 1)
loc_scale = partArrs[:, :, :2].std(axis = (0, 1)).mean() ** 0.2
vel_scale = partArrs[:, :, 2:].std(axis = (0, 1)).mean() ** 0.2
scale = (loc_scale, loc_scale, vel_scale, vel_scale)
partArrs = np.array([perturb_array(partArr, N = N, scale = scale) for partArr in partArrs])

part_frames = [subFrm.put(particles_to_points(partArr, N), color = kita_color, size = 0) for subFrm, partArr in zip(subFrms, partArrs)]
smoothFrms = BGRFrames.from_single_frames(part_frames)
# DecapStream(smoothFrms, 0, 100).show()

smoothFrms_ovl1 = smoothFrms[:overlap_frames_curr]
smoothFrms_main = smoothFrms[overlap_frames_curr:-overlap_frames_next]
smoothFrms_ovl2 = smoothFrms[-overlap_frames_next:]

smoothFrms_ovl = probFrms_ovl2.slow(2).dim_progressively(1, 0) + smoothFrms_ovl1.slow(2).dim_progressively(0, 1) 

smoothFrms = BGRFrames.from_frames([smoothFrms_ovl, smoothFrms_main.slow(2)])

smoothFrms = typewriter_with_ellipses(text, smoothFrms)
DecapStream(smoothFrms, 0, 100).show()

all_frames.append(smoothFrms)

### BALL
text = 'Ball detected'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 10

sec_frms = 35
start, stop = stop - overlap_frames_curr, stop + sec_frms

sub_pntSF = pntSF.conform(list(range(start, stop)))

subStm = Stream(cap, start, stop - 1)
subFrms = subStm.decap().frames * dim_factor

from ATMANpy.types.Lines import Region

out_frames = []
for pnt, frm in zip(sub_pntSF.values(), subFrms):
    # if pnt is None:
    #     reg = None
    # else:
    #     reg = Region.from_center(pnt, radius = 5)
    ball_frm = frm.put(pnt, color = ball_color, thickness = 2)

    out_frames.append(ball_frm)
    
ballFrms = BGRFrames.from_single_frames(out_frames)

ballFrms_ovl1 = ballFrms[:overlap_frames_curr]
ballFrms_main = ballFrms[overlap_frames_curr:-overlap_frames_next]
ballFrms_ovl2 = ballFrms[-overlap_frames_next:]

ballFrms_ovl = smoothFrms_ovl2.slow(2).dim_progressively(1, 0) + ballFrms_ovl1.slow(2).dim_progressively(0, 1) 

ballFrms = BGRFrames.from_frames([ballFrms_ovl, ballFrms_main.slow(2)])

ballFrms = typewriter_with_ellipses(text, ballFrms)
DecapStream(ballFrms, 0, 100).show()

all_frames.append(ballFrms)

vis_ut.show(all_frames)


##############
### VISUAL ###
##############
text = 'Analysing Swing Motion'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 10

sec_frms = 85
start, stop = stop - overlap_frames_curr, stop + sec_frms

topFrm = ff.read_feature_frame('top_hitTable', vid, sto)
botFrm = ff.read_feature_frame('bot_hitTable', vid, sto)

sub_topFrm = topFrm.loc[start:stop]
sub_botFrm = botFrm.loc[start:stop]

sub_topFrm = vis_ut.adjust_probs_for_handedness(sub_topFrm, hand = 'right')
sub_botFrm = vis_ut.adjust_probs_for_handedness(sub_botFrm, hand = 'right')

my_plySF = plySF.split()[seg_idx]
my_plySF = plySF[list(range(start,stop))]


### prep
from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
from ATMANpy.models.TemporalCNN.utils import load_cnn
model_params = prm_mod.hits_classify_video.model_params
top_mod = model_params['top']
model, params = load_cnn(top_mod, SmallFullConvModel, device = 'cpu')

T = params['T_CROP_SIZE'] + 1 ### add one so there is a clear "central" frame.
D = params['D_CROP_SIZE']


### region params
scale_factor = 1
# dim_factor = 0.5
p_threshold = 0.5

############################################
######## REGION STREAMS (FIRST SEG.) #######
############################################

from ATMANpy.types.Contours import Contours

##### PLAYER CONTOUR REGIONS #####
left_frames = my_plySF.get_frames()[:-T]
allDict = {}
for side in ['top', 'bot']:
    key = f'{side}PlayerCnt'
    plySF_one_player = my_plySF[key]

    plregDict = {}
    for left_frame in left_frames:
        right_frame = left_frame + T 
        choice_frames = list(range(left_frame, right_frame))
        central_frame = choice_frames[T//2] ### T is odd, so this is the central frame! (in case of T = 17, we want the 9th element, so need index 8!) 

        plCnts = []
        for frame in choice_frames:
            plCnt = plySF_one_player[frame]

            if len(plCnts) > 0:
                if plCnts[-1].overlap_factor(plCnt, relative_to = 'self') > 0:  
                    ### only append a contour if it is overlapping the previous
                    ### -> avoid drastic jumps between contours if a player contour is filtered out by moving over the baseline.                                    
                    plCnts.append(plCnt)
            else:
                plCnts.append(plCnt)

        plCnts = Contours(plCnts)

        plReg = plCnts.enclosing_regions().cover()
        plReg_shrunk = plReg.resize(to_height = D, to_width = D, how = 'any') ### shrink excess height or width --> the rest we do with padding! 

        plregDict[central_frame] = plReg_shrunk

    allDict[side] = plregDict      

#### SHOW REGIONS ###
frameList = []
for frame in my_plySF:
    currFrm = myStm.get(frame)

    try:
        topReg = allDict['top'][frame]
        botReg = allDict['bot'][frame]
    except KeyError:
        frameList.append(currFrm)
    else:
        frameList.append(currFrm.put(topReg).put(botReg))

regFrms = BGRFrames.from_single_frames(frameList)
# DecapStream(regFrms, 0, 100).show(ms_between_frames = 100)

### SHOW DIMMED FRAME WITH BRIGHT REGIONS
frameList = []
for frame in my_plySF:
    currFrm = myStm.get(frame).rescale(scale_factor)
    topSer = sub_topFrm.loc[frame]
    botSer = sub_botFrm.loc[frame]
    
    try:
        ball = pntSF[frame]
    except KeyError:
        ball = None

    # dimmed frame
    currFrm_dimmed = currFrm.dim(dim_factor).put(ball, color = const.BGR['ball_yellow'], thickness = 2)

    # reactive dimming factor
    top_p_event = 1 - topSer['top_oth']
    bot_p_event = 1 - botSer['bot_oth']

    top_dim_factor = top_p_event + (1 - top_p_event) * dim_factor
    bot_dim_factor = bot_p_event + (1 - bot_p_event) * dim_factor

    # probability strings
    topSer_event = topSer[topSer.index != 'top_oth']
    top_best_idx = topSer_event.argmax()
    top_best_evt = topSer_event.index[top_best_idx]
    top_best_prb = topSer_event[top_best_evt]
    top_best_str = vis_ut.map_label_to_explicit(top_best_evt)
    top_best_str = top_best_str.replace('(', '\n').strip(')')
    if top_best_prb > p_threshold:
        topStr = f'{top_best_str}\nConfidence: {top_best_prb:.0%}'
    else:
        topStr = ''

    botSer_event = botSer[botSer.index != 'bot_oth']
    bot_best_idx = botSer_event.argmax()
    bot_best_evt = botSer_event.index[bot_best_idx]
    bot_best_prb = botSer_event[bot_best_evt]
    bot_best_str = vis_ut.map_label_to_explicit(bot_best_evt)
    bot_best_str = bot_best_str.replace('(', '\n').strip(')')
    if bot_best_prb > p_threshold:
        botStr = f'{bot_best_str}\nConfidence: {bot_best_prb:.0%}'
    else:
        botStr = ''

    # overlay the dimmed crop and point onto frame
    try:
        topReg = allDict['top'][frame]
        botReg = allDict['bot'][frame]
    except KeyError:
        frameList.append(currFrm_dimmed)
    else:
        top_crop, bot_crop = currFrm.crop(topReg * scale_factor), currFrm.crop(botReg * scale_factor)
        top_tlPt, bot_tlPt = topReg.p1 * scale_factor, botReg.p1 * scale_factor

        top_crop = top_crop.dim(top_dim_factor)
        bot_crop = bot_crop.dim(bot_dim_factor)
        
        currFrm_over = overlay(currFrm_dimmed, top_crop, top_tlPt)  
        currFrm_over = overlay(currFrm_over, bot_crop, bot_tlPt)  

        top_text_coords = top_tlPt.right(D + 5).down(15).coords
        bot_text_coords = bot_tlPt.right(D + 5).down(15).coords

        currFrm_over = currFrm_over.put(topStr, where = top_text_coords).put(botStr, where = bot_text_coords)

        frameList.append(currFrm_over)

regFrms = BGRFrames.from_single_frames(frameList)
# DecapStream(regFrms, 0, 100).show(ms_between_frames = 25)

regFrms_ovl1 = regFrms[:overlap_frames_curr]
regFrms_main = regFrms[overlap_frames_curr:-overlap_frames_next]
regFrms_ovl2 = regFrms[-overlap_frames_next:]

regFrms_ovl = ballFrms_ovl2.slow(2).slow(2).dim_progressively(1, 0) + regFrms_ovl1.slow(2).slow(2).dim_progressively(0, 1) 

regFrms = BGRFrames.from_frames([regFrms_ovl, regFrms_main.slow(2).slow(2)])

regFrms = typewriter_with_ellipses(text, regFrms)
DecapStream(regFrms, 0, 100).show()

all_frames.append(regFrms)

### AUDIO ###
text = 'Listening for Hits and Bounces'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 10

# sec_frms = 75
start, stop = stop - overlap_frames_curr, fsShot.get_hit_frame()

audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)

from ATMANpy.models.SoundCNN.data import waveform_generator, point_sound_generator
from ATMANpy.models.SoundCNN.nets import SoundModel_MN
from ATMANpy.models.SoundCNN.utils import load_cnn, first_window_with_sample, last_window_with_sample

from ATMANpy.types.Overlays import ScatterChart, Graph

model_params = prm_mod.hits_classify_sound.model_params
model_name = model_params['model_name']

model, params = load_cnn(model_name, SoundModel_MN, device = 'cpu')
    
wave_gen = waveform_generator(vid, sto, yield_frame_indices = True)

for _ in range(seg_idx + 1):
    my_wave, fs_wave, ls_wave = next(wave_gen)

### renormalize
abs_wave = np.abs(my_wave)
sign_wave = np.sign(my_wave)
my_wave = sign_wave * (abs_wave / abs_wave.max()) ** 0.5

window_width_in_seconds = 0.2
window_width_in_frames = window_width_in_seconds * 25

sr = 44100
wav_color = const.BGR['bluey_green']
hit_color = const.BGR['rose_orange']
bnc_color = const.BGR['soft_yellow']
now_color = const.BGR['bright_blue']
now_line_points = 400
chart_dims = (1920 // 2, 1080 // 2)

box2D = CourtLines.make().add_singles_netline().get_service_box(side = 'Bottom', which = 'Ad')
box3D = Region3D.from_region2D(box2D, z = 0).right(5)

start_seconds = - window_width_in_seconds / 2
wave_graphs = []
prob_graphs = []
spec_graphs = []
for frame in range(start, stop): 
    stop_seconds = start_seconds + window_width_in_seconds
    
    start_sample = int(max(start_seconds, 0) * sr) 
    stop_sample = int(stop_seconds * sr)

    ### ball    
    try:
        ball = pntSF[frame]
    except KeyError:
        ball = None

    ### waveform
    wave_snippet = my_wave[start_sample:stop_sample]

    pad_left = -min(start_seconds, 0) ### to pad with zeros, if there is no waveform (at the very beginning)
    pad_samples = int(pad_left * sr)
    pad_snippet = np.zeros(pad_samples)

    snippet = np.concatenate([pad_snippet, wave_snippet])
    xs = np.linspace(0, 1, num = len(snippet))
    waveChart = ScatterChart(xs, snippet, color = wav_color)

    nowChart = ScatterChart(np.ones(now_line_points) * 0.5, np.linspace(-1, 1, now_line_points), color = now_color)

    waveGraph = Graph([waveChart, nowChart], make_default_x_labels = False, as_density = False, resolution = 20,
                        tick_label_size = 0, color = const.BGR['bright_grey'], dimensions = chart_dims, offset = 0.25,
                        header_text = ' ', header_height = 1, header_size = 1, header_horizontal_offset = 10,
                        cam_reg_override = box3D,
                    )
    waveScn, annPnts = waveGraph.preview()
    waveScn = waveScn.drop(1) ### drop the y axis

    annPnts[2] = annPnts[2].right(115).back(55)

    hit, bnc = audFrm.loc[frame]
    if hit > 0.5:
        currStr = f'Hit Heard\n(Certainty: {hit:.0%})'
    elif bnc > 0.5:
        currStr = f'Bounce Heard\n(Certainty: {bnc:.0%})'
    else:
        currStr = ''

    annPnts.append(annPnts[2].left(115).back(175).retext(currStr))

    # waveView = View.make(waveGraph.cam, waveScn, annotatedPoints = annPnts, resolution = 20)
    waveView = View.make(Camera.calibrated_brest_behind(), waveScn, annotatedPoints = annPnts, resolution = 20)
    
    # courtView = View.make(Camera.calibrated_brest_behind(), Scene.make_court(), resolution = 10)
    currFrm = myStm.get(frame).dim(dim_factor).put(ball, color = const.BGR['ball_yellow'], thickness = 2)

    waveFrm = waveView.snap(over_array = currFrm.array)
    wave_graphs.append(waveFrm)

    start_seconds += (1 / 25)

waveFrms = BGRFrames.from_single_frames(wave_graphs)

waveFrms_ovl1 = waveFrms[:overlap_frames_curr]
waveFrms_main = waveFrms[overlap_frames_curr:-overlap_frames_next]
waveFrms_ovl2 = waveFrms[-overlap_frames_next:]

waveFrms_ovl = regFrms_ovl2.slow(2).slow(2).dim_progressively(1, 0) + waveFrms_ovl1.slow(2).slow(2).dim_progressively(0, 1) 

waveFrms = BGRFrames.from_frames([waveFrms_ovl, waveFrms_main.slow(2).slow(2)])

waveFrms = typewriter_with_ellipses(text, waveFrms)
DecapStream(waveFrms, 0, 100).show()

all_frames.append(waveFrms)

######### Trajectory #########

text = 'Trajectory Detected'

overlap_frames_curr = overlap_frames_next
overlap_frames_next = 10

mark_color = const.BGR['bright_blue']
mark_size = 6

shtScns, bncScns = vis_ut.get_shot_scenes(myPnt[first_trajectory_shot_idx:],
                                          mark_size = mark_size, mark_color = mark_color,
                                          ball_color = ball_color, net_color = net_color)

lstFrms = myPnt[first_trajectory_shot_idx:].stream(cap).join().maybe_to_stream().decap().frames
lstFrms = lstFrms.dim(dim_factor)

assert lstFrms.length() == len(shtScns)

res = 1
prmDict = {'foreground_resolution': res, 'background_resolution': res, 'argument_resolution': res,
           'cumulative_foreground_transparence': 0.9, 'cumulate_argument': True,}

trajFrms, animDict = vs.animate_sequence(shtScns, 
                                        start_cam = Camera.calibrated_brest_behind(), 
                                        stop_cam = Camera.calibrated_brest_behind(),
                                        animate_over = lstFrms,
                                        **prmDict)

trajFrms_ovl1 = trajFrms[:overlap_frames_curr]
trajFrms_main = trajFrms[overlap_frames_curr:-overlap_frames_next]
trajFrms_ovl2 = trajFrms[-overlap_frames_next:]

trajFrms_ovl = waveFrms_ovl2.slow(2).slow(2).dim_progressively(1, 0) + trajFrms_ovl1.slow(2).slow(2).dim_progressively(0, 1) 

trajFrms = BGRFrames.from_frames([trajFrms_ovl, trajFrms_main.slow(2), trajFrms_ovl2])

trajFrms = typewriter_with_ellipses(text, trajFrms)
DecapStream(trajFrms, 0, 100).show()

all_frames.append(trajFrms)

### transition to beginning
hold_frames = 25
fade_frames = 50
transition_frames = hold_frames + fade_frames
lsFrms = myStm.widen(transition_frames).decap().frames[-transition_frames:]
fsFrms = myStm.widen(fade_frames + initial_frames).decap().frames[:fade_frames]

lsFrms_with_fading, _ = vs.animate_sequence([Scene.make_empty() for _ in range(transition_frames)],
                                            start_cam = Camera.calibrated_brest_behind(), 
                                            stop_cam = Camera.calibrated_brest_behind(),
                                            animate_over = lsFrms.dim(dim_factor),
                                            **animDict,
                                            **prmDict)

lsFrms_hold, lsFrms_fade = lsFrms_with_fading[:hold_frames], lsFrms_with_fading[hold_frames:]

holdFrms = lsFrms_hold
fadeFrms = lsFrms_fade.dim_progressively(1, 0) + fsFrms.dim_progressively(0, 1)
transFrms = BGRFrames.from_frames([holdFrms, fadeFrms])
DecapStream(transFrms, 0, 100).show()

all_frames.append(transFrms)

vis_ut.show(all_frames)

allFrms = BGRFrames.from_frames(all_frames)

allFrms.save(fullPath, f'recognition', lossless = True, save_as = 'mp4')

# TODO:
# text closer to left edge
# collecting ball candidates could be a bit faster
# smoothing ball candidates could be a bit faster
# speed changes in the ball part a bit too hectic
# ball tracking a bit too fast, smoothing ball tracking a bit too slow
# transition is too fast at the end.




