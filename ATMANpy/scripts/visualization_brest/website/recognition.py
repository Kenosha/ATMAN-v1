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

mtc_idx, pnt_idx = 0, 111 ### the first point of atman short intro video
myPnt = allPnts.get(mtc_idx, pnt_idx)

##### NOTES
# court, net and players have to be breathing in and out of the thing with a fade
seg_idx = 0

dim_factor = 0.25
myStm = myPnt.stream(confs = confs)
myDtm = myStm.widen(100).decap()
stream_start = myDtm.start_frame

sec_frms = 75
brt_frms = sec_frms // 3

blackFrm = BGRFrame.make_black(640, 360)

line_color = const.BGR['rose_orange']
net_color = const.BGR['salmon_pink']
player_color = const.BGR['strong_pink']

### LINES ###
start, stop = stream_start, stream_start + sec_frms

HSF = ConstFeature.from_store(vid, sto, featname = 'H')
H = HSF.feature
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
mixFrms_out = courtFrms_out.dim_progressively(1, 0) + currFrms_out.dim_progressively(dim_factor, 1)

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

currFrms = myDtm.conform(start, stop - 1).frames
currFrms_in, currFrms_hold, currFrms_out = currFrms[:brt_frms], currFrms[brt_frms:2*brt_frms], currFrms[2*brt_frms:3*brt_frms]
currFrms = BGRFrames.from_frames([currFrms_in.dim_progressively(1, dim_factor),
                                  currFrms_hold.dim(dim_factor),
                                  currFrms_out.dim_progressively(dim_factor, 1)])

cam0 = Camera.calibrated_brest_behind()
netFrms, _ = vs.animate_sequence(netScns, start_cam = cam0, stop_cam = cam0, animate_over = mixFrms)
# DecapStream(netFrms, 0, 100).show()

### PLAYERS ###
sec_frms = 60
in_cumul_share, hold_cumul_share, out_cumul_share = 0.1, 0.9, 1
in_frms = int(in_cumul_share * sec_frms)
hold_frms = int(hold_cumul_share * sec_frms) 
out_frms = int(out_cumul_share * sec_frms)

start, stop = stop, stop + sec_frms

plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])

my_plySF = plySF.split()[seg_idx]
my_plySF = plySF[list(range(start,stop))]

currStm = Stream(cap, start, stop).attach(plySF)
blnkFrms = currStm.decap().frames
currFrms = currStm.decap_with_features().frames


currFrms_in, currFrms_hold, currFrms_out = currFrms[:in_frms], currFrms[in_frms:hold_frms], currFrms[hold_frms:]
currFrms = BGRFrames.from_frames([currFrms_in.dim_progressively(0, 1 - dim_factor),
                                  currFrms_hold.dim(1 - dim_factor),
                                  currFrms_out.dim_progressively(1 - dim_factor, 0)])

blnkFrms_in, blnkFrms_hold, blnkFrms_out = blnkFrms[:in_frms], blnkFrms[in_frms:hold_frms], blnkFrms[hold_frms:]
blnkFrms = BGRFrames.from_frames([blnkFrms_in.dim_progressively(1, dim_factor),
                                  blnkFrms_hold.dim(dim_factor),
                                  blnkFrms_out.dim_progressively(dim_factor, 1)])

plyFrms = currFrms + blnkFrms

DecapStream(plyFrms, 0, 100).show()

### 
sec_frms = 75
start, stop = stop, stop + sec_frms

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
dim_factor = 0.5
p_threshold = 0.1

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

    # dimmed frame
    currFrm_dimmed = currFrm.dim(dim_factor)

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
# DecapStream(regFrms, 0, 100).show(ms_between_frames = 250)

### AUDIO ###
sec_frms = 75
start, stop = stop, stop + sec_frms

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
                        header_text = 'Sound\nAnalysis', header_height = 1, header_size = 1, header_horizontal_offset = 10,
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

    annPnts.append(annPnts[2].left(115).back(150).retext(currStr))

    # waveView = View.make(waveGraph.cam, waveScn, annotatedPoints = annPnts, resolution = 20)
    waveView = View.make(Camera.calibrated_brest_behind(), waveScn, annotatedPoints = annPnts, resolution = 20)
    
    # courtView = View.make(Camera.calibrated_brest_behind(), Scene.make_court(), resolution = 10)
    
    waveFrm = waveView.snap(over_array = myStm.get(frame).array)
    wave_graphs.append(waveFrm)

    start_seconds += (1 / 25)

DecapStream(BGRFrames.from_single_frames(wave_graphs), 0, 100).show(ms_between_frames = 200)






### SVM ###




