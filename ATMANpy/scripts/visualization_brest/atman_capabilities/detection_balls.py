
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

seg_idx = 2 ### balls -> idx = 2
my_segBF = segBFs[seg_idx]

dim_factor = 0.5

shadStms = StreamList.from_store('shadowMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')

zeroFrm = BGRFrame(np.zeros((360, 640, 3), dtype = 'uint8'))

### still need court lines ...
HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature
courtLines = sa.find_courtlines(H).drop('Net Line')

#######################
####### BALL ##########
#######################

all_frames = []
script_lines = []

motiStms = StreamList.from_store('fine_motMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 
motiStms_filt = StreamList.from_store('fine_motMask_filt', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 

my_motiStms = motiStms[seg_idx]
my_motiStms_filt = motiStms_filt[seg_idx]

cntSF = StreamFeature.from_store(vid, sto, featname = 'movCnts')
cndSF = StreamFeature.from_store(vid, sto, featname = 'cand2Cnts') 
pntSF = StreamFeature.from_store(vid, sto, featname = 'ball_choicePt') 
hitSF = StreamFeature.from_store(vid, sto, featname = 'hitCnt') 

### now, show the fine motion mask.
script_line = "To identify the ball, we look for objects in motion."

duration = 100
start = motiStms[2].start_frame
stop = start + duration

motiMask = my_motiStms.conform(start, stop - 1).decap().frames
currFrms = Stream(cap, start, stop - 1).decap().frames

mixFrms = currFrms.mask(motiMask) + currFrms.mask(~motiMask) * dim_factor
mixFrms = mixFrms.rescale(3)

all_frames.append(mixFrms)
script_lines.append(script_line)

DecapStream(mixFrms, 0, 100).show()

### the filtered motion mask
script_line = "Naturally, the players and their shadows are in motion but have already been detected: We filter them out."

duration = 100
start, stop = stop, stop + duration

motiMask = my_motiStms_filt.conform(start, stop - 1).decap().frames
shadMask = shadStms[2].conform(start, stop - 1).decap().frames
joinMask = motiMask & ~shadMask

currFrms = Stream(cap, start, stop - 1).decap().frames

mixFrms = currFrms.mask(joinMask) + currFrms.mask(~joinMask) * dim_factor
mixFrms = mixFrms.rescale(3)

all_frames.append(mixFrms)
script_lines.append(script_line)

DecapStream(mixFrms, 0, 100).show()

### the candidate contours
script_line = "What remains are ball candidates; among these, we see the ball, but also some visual clutter which we have to filter out."

ball_color = const.BGR['ball_yellow']
duration = 90
start, stop = stop, stop + duration

cnt_vals = cntSF.conform(list(range(start, stop))).values()

currFrms = Stream(cap, start, stop - 1).decap().frames
currFrms_with_contours = BGRFrames.from_single_frames([currFrm.put(candCnts, color = ball_color, thickness = 2) for currFrm, candCnts in zip(currFrms, cnt_vals)])

cntFrms = BGRFrames.from_single_frames([zeroFrm.put(cnts, thickness = 2) for cnts in cnt_vals])
cntMask = cntFrms.to_grey() > 0
shadMask = shadStms[2].conform(start, stop - 1).decap().frames
joinMask = cntMask & ~shadMask

mixFrms = currFrms_with_contours.mask(joinMask) + currFrms.mask(~joinMask) * dim_factor
mixFrms = mixFrms.rescale(3)

all_frames.append(mixFrms)
script_lines.append(script_line)

### the SVM
# -> idea: show a few frames in super-slo mo with ball probabilities.
script_line = "To this end, we train a support vector machine (SVM) to recognize shapes that look like a ball."

duration = 15
start, stop = stop, stop + duration

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

# annotate each frame
currStm = Stream(cap, start, stop)
sub_shadStm = shadStms[2].conform(start, stop) ### (so we can filter out the contours in the shadow...)

out_frames = []
for i in range(duration):
    cnts = sub_cntSF[start + i]
    probs = normTable.loc[start + i, 'isBall_proba']

    currFrm_undimmed = currStm.get(start + i)
    currFrm = currFrm_undimmed.dim(dim_factor)
    shadFrm = sub_shadStm.get(start + i)
    regMask = shadFrm.zero()

    x_offset = 10
    y_offset = 5

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
        currFrm = currFrm.put(pStr, where = (x, y))

    mixFrm = currFrm_undimmed.mask(regMask) + currFrm.mask(~regMask)

    mixFrm = mixFrm.rescale(3)
    mixFrms = BGRFrames.from_frame(mixFrm, 25)

    out_frames.append(mixFrms)

probFrms = BGRFrames.from_frames(out_frames)
DecapStream(probFrms, 0, 100).show()

all_frames.append(probFrms)
script_lines.append(script_line)

### intermezzo
script_line =   """
                    The Support Vector Machine helps us exclude moving shapes which are obviously not a ball;
                    However, it is set to err on the side of caution, and therefore occasionally yields false positives:
                    it believes shapes to be the ball which in fact are not the ball.
                """
                ### -> TODO: give examples with a freeze frame!
                ### -> how? once you vet the code above, and have it recorded it, come back here, 
                ### and look for a good example to freeze on within the duration (not too close to beginning or end).
                ### then, we simply freeze on that frame, and continue with the duration. 
duration = 100
start, stop = stop, stop + duration

cand_color = const.BGR['leafy_green']
non_cand_color = const.BGR['intense_red']

from ATMANpy.types.Contours import Contours
my_cntSF = cntSF.conform(list(range(start, stop))).fillna(Contours([]))
my_cndSF = cndSF.conform(list(range(start, stop))).fillna(Contours([]))

# annotate each frame
currStm = Stream(cap, start, stop)
sub_shadStm = shadStms[2].conform(start, stop) ### (so we can filter out the contours in the shadow...)

initial_frames_per_frame = 25
decay_rate = 0.05

out_frames = []
for i in range(duration):
    cnts = my_cntSF[start + i]
    cnds = my_cndSF[start + i]
    # non_cnds = Contours([cnt for cnt in cnts if cnt not in cnds])
    non_cnds = cnts

    currFrm_undimmed = currStm.get(start + i)
    currFrm = currFrm_undimmed.dim(dim_factor)
    shadFrm = sub_shadStm.get(start + i)
    regMask = shadFrm.zero()

    x_offset = 10
    y_offset = 5

    for cnt in non_cnds:
        cntPnt = cnt.center()
        
        ### check if the contour is in the shadow.
        if shadFrm.eval_at(cntPnt) == 255:
            continue
        
        x, y = cntPnt.coords
        x, y = x + x_offset, y + y_offset
        reg = cnt.enclosing_region(margin = 3)

        regMask = regMask | regMask.within_region(reg)
        currFrm = currFrm.put(reg, color = non_cand_color, thickness = 2)
    
    for cnt in cnds:
        reg = cnt.enclosing_region(margin = 3)

        regMask = regMask | regMask.within_region(reg)
        currFrm = currFrm.put(reg, color = cand_color, thickness = 2)

    mixFrm = currFrm_undimmed.mask(regMask) + currFrm.mask(~regMask)

    mixFrm = mixFrm.rescale(3)

    frames_repeat = max(1, int(initial_frames_per_frame * np.exp(-decay_rate * i)))
    mixFrms = BGRFrames.from_frame(mixFrm, frames_repeat)

    out_frames.append(mixFrms)

interFrms = BGRFrames.from_frames(out_frames)

all_frames.append(interFrms)
script_lines.append(script_line)

DecapStream(interFrms, 0, 100).show()

### The Particle Filter in Action.
hitcntSF = StreamFeature.from_store(vid, sto, featname = 'hitCnt')
hitflagSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')

### Combine ball contours with hypothesized hit contours
cntplusSF = cndSF.rename('candCnts').merge(hitcntSF, how = 'outer').apply(sa.merge_contour_into_contours)

hitcntSFs = hitcntSF.conform(segBF.filter(segBF)).split()
hitflagSFs = hitflagSF.conform(segBF.filter(segBF)).split()
cntSFs_high = cndSF.rename('candCnts').conform(segBF.filter(segBF)).split()
cntplusSFs = cntplusSF.conform(segBF.filter(segBF)).split()

### Create model switching points at and around hit points
hitSFs = hitflagSFs.dropna().thin().merge(hitcntSFs) ### thin out hitflags so we have one flag for each hit exactly.

hitSF = hitSFs[2]

propagation_params = prm_mod.ball_inference_particles.propagation_params
filter_init_params = prm_mod.ball_inference_particles.filter_init_params
filter_run_params = prm_mod.ball_inference_particles.filter_run_params

bSF = sf.create_propagation_models(hitSF, courtLines = courtLines)
bSF = sf.widen_propagation_models(bSF, **propagation_params)

cntplusSF = cntplusSFs[2]
hitSF = hitSFs[2]

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
part_color = const.BGR['fair_yellow']
kita_color = const.BGR['ball_yellow']

script_line =   """
                    To remove those ball shapes that the SVM believes to look like a ball, 
                    we implement a Particle Filter. The Particle Filter tracks candidate ball shapes on the screen;
                    the particles that track the ball obey simple motion laws, and therefore 
                    stay close to the ball shapes that behave like a ball in flight.
                """
duration = 100
start, stop = stop, stop + duration

subStm = Stream(cap, start, stop - 1)
subFrms = subStm.decap().frames * dim_factor
subFrms_resc = subFrms.rescale(3) ### scale up, so we render the points on the big screen directly!

my_partSF = partSF['particles'].conform(list(range(start, stop)))
partArrs = np.array(list(my_partSF.values()))
partArrs_resc = partArrs * 3 ### scale up, so we render the points on the big screen directly!

part_frames = [subFrm.put(particles_to_points(partArr, N), color = part_color, size = 0) for subFrm, partArr in zip(subFrms_resc, partArrs_resc)]
partFrms = BGRFrames.from_single_frames(part_frames)

DecapStream(partFrms, 0, 100).show(ms_between_frames = 100)

all_frames.append(partFrms)
script_lines.append(script_line)

script_line =   """
                    In flight, the particles consistently gather around the travelling ball, and therefore confirm its position;
                    However, whenever the ball is hit by either player, the subsequent flight of the ball is uncertain:
                    This is reflected in the particles, which scatter for a short moment to re-locate the ball.
                    This scattering can be reduced using a technique called Kitagawa Smoothing.
                """
duration = 100
start, stop = stop, stop + duration

subStm = Stream(cap, start, stop - 1)
subFrms = subStm.decap().frames * dim_factor
subFrms_resc = subFrms.rescale(3) ### scale up, so we render the points on the big screen directly!

my_partSF = kitaSF.conform(list(range(start, stop)))
partArrs = np.array(list(my_partSF.values()))
partArrs_resc = partArrs * 3 ### scale up, so we render the points on the big screen directly!

### perturb the kitagawa particles a bit. 
# (even though this does not happen in the code, it would yield the same results and looks better on screen.)
partArr = partArrs_resc[0]
def perturb_array(partArr, N = 100, scale = (10, 10, 1, 1)):
    uniques = np.unique(partArr, axis = 0)

    randArrs = []
    for _ in range(N):
        idx = np.random.choice(len(uniques))
        randArr = np.random.normal(uniques[idx], scale = scale)
        randArrs.append(randArr)

    return np.array(randArrs)

scale = (10, 10, 1, 1)
partArrs_resc = np.array([perturb_array(partArr, N = N, scale = scale) for partArr in partArrs_resc])

part_frames = [subFrm.put(particles_to_points(partArr, N), color = kita_color, size = 0) for subFrm, partArr in zip(subFrms_resc, partArrs_resc)]
partFrms = BGRFrames.from_single_frames(part_frames)

all_frames.append(partFrms)
script_lines.append(script_lines)

DecapStream(partFrms, 0, 100).show(ms_between_frames = 100)

script_line =   """
                    With the support vector machine finding ball candidates based on how they look, 
                    and the particle filter confirming them based on how they behave in flight,
                    we can fairly confidently identify the ball in flight. 

                    Note that in some frames, no ball is detected. Given the low resolution and bitrate of the video, 
                    this is no surprise; however, it means that our job in detecting the ball is not done:
                    We have to reconstruct the entire ball trajectory, at all times.

                    To this end, we use the frames in which a ball is detected, but also additional information, to improve the quality
                    of the trajectory detected. 
                """
### -> go until the end!
start = stop
stop = segBFs[2].last()[0]

sub_pntSF = pntSF.conform(list(range(start, stop)))

subStm = Stream(cap, start, stop - 1)
subFrms = subStm.decap().frames * dim_factor

from ATMANpy.types.Lines import Region

out_frames = []
for pnt, frm in zip(sub_pntSF.values(), subFrms):
    if pnt is None:
        reg = None
    else:
        reg = Region.from_center(pnt, radius = 5)
    ball_frm = frm.put(reg, color = ball_color, thickness = 1)

    ball_frm = ball_frm.rescale(3)
    out_frames.append(ball_frm)
    
ballFrms = BGRFrames.from_single_frames(out_frames)
all_frames.append(ballFrms)
script_lines.append(script_line)

### Intermezzo to the next step: sound recognition.

show(all_frames)