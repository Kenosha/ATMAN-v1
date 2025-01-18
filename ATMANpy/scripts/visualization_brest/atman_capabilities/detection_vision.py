
from ATMANpy.types.Contours import Contours
from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
from ATMANpy.models.TemporalCNN.utils import load_cnn

import numpy as np
import pandas as pd
import sounddevice as sd
        
from ATMANpy.funcs.util_funcs import get_folder_struct, split
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Abstract import Points, Shots, read_in_data_table, read_in_placeholder_table
from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames, MaskFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Visual3D import Camera, View, Scene

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'vision'
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

### sound and vision -> segment 5 and 6
seg_idx = 5
fs_segBF = segBFs[seg_idx] 
ls_segBF = segBFs[seg_idx + 1]

how_many_frames_in_vision = 100 ### -> how many frames belonging to the intermezzo are needed in the next part (also needed in vision)

myStm = Stream.from_SF(fs_segBF, cap).widen(how_many_frames_in_vision)
myStm2 = Stream.from_SF(ls_segBF, cap).widen(how_many_frames_in_vision)

### prep
model_params = prm_mod.hits_classify_video.model_params
top_mod = model_params['top']
model, params = load_cnn(top_mod, SmallFullConvModel, device = 'cpu')

plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])

topFrm = ff.read_feature_frame('top_hitTable', vid, sto)
botFrm = ff.read_feature_frame('bot_hitTable', vid, sto)

sub_topFrm = topFrm.loc[fs:ls]
sub_botFrm = botFrm.loc[fs:ls]

sub_topFrm = vis_ut.adjust_probs_for_handedness(sub_topFrm, hand = 'right')
sub_botFrm = vis_ut.adjust_probs_for_handedness(sub_botFrm, hand = 'right')

T = params['T_CROP_SIZE'] + 1 ### add one so there is a clear "central" frame.
D = params['D_CROP_SIZE']

### region params
scale_factor = 3
dim_factor = 0.5
p_threshold = 0.1

############################################
######## REGION STREAMS (FIRST SEG.) #######
############################################
fs_plySF = plySF.conform(fs_segBF)

##### PLAYER CONTOUR REGIONS #####
left_frames = fs_plySF.get_frames()[:-T]
allDict = {}
for side in ['top', 'bot']:
    key = f'{side}PlayerCnt'
    plySF_one_player = fs_plySF[key]

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
for frame in fs_plySF:
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
for frame in fs_plySF:
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
    if top_best_prb > p_threshold:
        topStr = f'{top_best_str}: {top_best_prb:.0%}'
    else:
        topStr = ''

    botSer_event = botSer[botSer.index != 'bot_oth']
    bot_best_idx = botSer_event.argmax()
    bot_best_evt = botSer_event.index[bot_best_idx]
    bot_best_prb = botSer_event[bot_best_evt]
    bot_best_str = vis_ut.map_label_to_explicit(bot_best_evt)
    if bot_best_prb > p_threshold:
        botStr = f'{bot_best_str}: {bot_best_prb:.0%}'
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

        currFrm_over = currFrm_over.put(topStr, where = top_tlPt.coords).put(botStr, where = bot_tlPt.coords)

        frameList.append(currFrm_over)

regFrms = BGRFrames.from_single_frames(frameList)
# DecapStream(regFrms, 0, 100).show(ms_between_frames = 100)

############################################
####### REGION STREAMS (SECOND SEG.) #######
############################################
ls_plySF = plySF.conform(ls_segBF)

##### PLAYER CONTOUR REGIONS #####
left_frames = ls_plySF.get_frames()[:-T]
allDict = {}
for side in ['top', 'bot']:
    key = f'{side}PlayerCnt'
    plySF_one_player = ls_plySF[key]

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
for frame in ls_plySF:
    currFrm = myStm2.get(frame)

    try:
        topReg = allDict['top'][frame]
        botReg = allDict['bot'][frame]
    except KeyError:
        frameList.append(currFrm)
    else:
        frameList.append(currFrm.put(topReg).put(botReg))

regFrms2 = BGRFrames.from_single_frames(frameList)
# DecapStream(regFrms2, 0, 100).show(ms_between_frames = 100)

sub_topFrm_int = (sub_topFrm * 100).astype(int)
sub_botFrm_int = (sub_botFrm * 100).astype(int)

sub_topFrm_int_ls = sub_topFrm_int.loc[ls_segBF.get_frames()]
sub_botFrm_int_ls = sub_botFrm_int.loc[ls_segBF.get_frames()]

import os
save_folder = get_folder_struct(None)['main']
sub_topFrm_int_ls.to_csv(os.path.join(save_folder, 'top.csv'))
sub_botFrm_int_ls.to_csv(os.path.join(save_folder, 'bot.csv'))

hitSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')
single_hit_SFs = hitSF.conform(segBF).split()[6].dropna().split()
last_frms_and_ids = [sf.last() for sf in single_hit_SFs]
last_frms = [sf.last()[0] for sf in single_hit_SFs]
### what are the maxima at these last frames
# this is mostly to reduce the confusion around topspin/slice 
# (we have shown confusion of the visual model already, and the confusion is fixed by the use of sound and ball flight data; no need to re-iterate)
top_frms = last_frms[::2]
bot_frms = last_frms[1::2]

### expand these, to have a zone where only the top event at time of hit is shown.
widen_before, widen_after = 5, 5

top_dict = {}
for frame, rowSer in sub_topFrm.loc[top_frms].iterrows():
    rowSer = rowSer.drop('top_oth')
    max_idx = rowSer.argmax()
    max_evt = rowSer.index[max_idx]

    for wide_frame in range(frame - widen_before, frame + widen_after + 1):
        top_dict[wide_frame] = max_evt

bot_dict = {}
for frame, rowSer in sub_botFrm.loc[bot_frms].iterrows():
    rowSer = rowSer.drop('bot_oth')
    max_idx = rowSer.argmax()
    max_evt = rowSer.index[max_idx]
    
    for wide_frame in range(frame - widen_before, frame + widen_after + 1):
        bot_dict[wide_frame] = max_evt

### SHOW DIMMED FRAME WITH BRIGHT REGIONS
frameList = []
for frame in ls_plySF:
    currFrm = myStm2.get(frame).rescale(scale_factor)
    topSer = sub_topFrm.loc[frame]
    botSer = sub_botFrm.loc[frame]

    # dimmed frame
    currFrm_dimmed = currFrm.dim(dim_factor)

    # reactive dimming factor
    top_p_event = 1 - topSer['top_oth']
    bot_p_event = 1 - botSer['bot_oth']

    top_dim_factor = top_p_event + (1 - top_p_event) * dim_factor
    bot_dim_factor = bot_p_event + (1 - bot_p_event) * dim_factor

    topSer_event = topSer[topSer.index != 'top_oth']

    # probability strings
    if frame not in top_dict:    
        top_best_idx = topSer_event.argmax()
        top_best_evt = topSer_event.index[top_best_idx]
    else:
        top_best_evt = top_dict[frame]
    
    top_best_prb = topSer_event[top_best_evt]
    top_best_str = vis_ut.map_label_to_explicit(top_best_evt)
    if top_best_prb > p_threshold:
        topStr = f'{top_best_str}: {top_best_prb:.0%}'
    else:
        topStr = ''

    botSer_event = botSer[botSer.index != 'bot_oth']
    if frame not in bot_dict:
        bot_best_idx = botSer_event.argmax()
        bot_best_evt = botSer_event.index[bot_best_idx]
    else:
        bot_best_evt = bot_dict[frame]

    bot_best_prb = botSer_event[bot_best_evt]
    bot_best_str = vis_ut.map_label_to_explicit(bot_best_evt)
    if bot_best_prb > p_threshold:
        botStr = f'{bot_best_str}: {bot_best_prb:.0%}'
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

        currFrm_over = currFrm_over.put(topStr, where = top_tlPt.coords).put(botStr, where = bot_tlPt.coords)

        frameList.append(currFrm_over)

regFrms2 = BGRFrames.from_single_frames(frameList)
# DecapStream(regFrms2, 0, 100).show(ms_between_frames = 100)
### TODO: Issue here: Slice is not unambiguous for top player! (fix by enforcing the detected shot at the time of hit for the detection around it?)
### FIXED! Done with the confusion avoider above! (which mirrors what happens once the visual swing info is joined with timing of hit and ball flight.)

### ANIMATION ###
all_frames = []
script_lines = []

### intro
script_line =   """
                    To this end, a neural network is trained that can detect the type of shot being hit at any given moment.
                    The neural network distinguishes between serves, groundstrokes and volleys; it can tell a topspin shot from a slice,
                    and accounts for player-handedness.
                """ ## 15 seconds
preFrms = myStm.decap().frames[:how_many_frames_in_vision].rescale(3)
pre_preFrms = BGRFrames.from_frame(preFrms[0], 250)
comb_preFrms = BGRFrames.from_frames([pre_preFrms, preFrms[:60], preFrms[60:90].slow(2), preFrms[90:100].slow(2).slow(2)])

all_frames.append(comb_preFrms)
script_lines.append(script_line)

### serve
# TODO: ADJUST PLAYER REGIONS WITH PROBABILITIES SO THEY START A BIT EARLIER -> serve is detected too late because segBF is too tight.
script_line = "The neural network correctly identifies the serve (with the highest degree of confidence roughly at the moment of impact)."
duration = 25
start, stop = 0, duration
currFrms = regFrms[start:stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

### return
script_line = "The return is correctly identified as a topspin backhand groundstroke."
duration = 18
start, stop = stop, stop + duration
currFrms = regFrms[start:stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

### slight confusion
script_line =   """
                    At this point, the neural network confuses the wind-down of the Gasquet's backhand motion with a forehand.
                    Though this is not optimal, once the visual information is combined with ball flight and sound data, 
                    such confusions are easily eliminated, as we will see in a bit. 
                """ ### 17 seconds
duration = 14
start, stop = stop, stop + duration
freeze_within = 6

currFrms_pre = regFrms[start:start+freeze_within]
currFrms_stat = regFrms[start+freeze_within]
currFrms_post = regFrms[start+freeze_within:stop]

currFrms_pre = currFrms_pre.slow(2).slow(2).slow(2)
currFrms_stat = BGRFrames.from_frame(currFrms_stat, 350)
currFrms_post = currFrms_post.slow(2).slow(2).slow(2)

currFrms = BGRFrames.from_frames([currFrms_pre, currFrms_stat, currFrms_post])
# DecapStream(currFrms, 0, 100).show()

all_frames.append(currFrms)
script_lines.append(script_line)

### serve + 1
script_line =   """
                    The next shot is also correctly identified as a topspin backhand, with the highest degree of certainty (96%)
                    coming at the moment of impact. 
                """ ### 10 seconds
duration = 35
start, stop = stop, stop + duration
currFrms = regFrms[start:stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

### return + 1
script_line =   """
                    Another topspin backhand, and again the moment of impact correlates very well with the highest level of certainty.
                """ ### 7 seconds
duration = 25
start, stop = stop, stop + duration
currFrms = regFrms[start:stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

### serve + 2
script_line =   """
                    Thomas Fabbiano's topspin backhand is recognized as such also, at 99% certainty around the moment of impact.
                    In this way, visual analysis of player motion supplements ball flight and sound information, 
                    and tells us additionally what kind of shot is being hit.  
                """ ### 15 seconds
duration = 50
start, stop = stop, stop + duration
currFrms = regFrms[start:stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

### after
script_line = ""
currFrms = regFrms[stop:]
currFrms = BGRFrames.from_frames([currFrms[:10].slow(2).slow(2), currFrms[10:].slow(2)])

all_frames.append(currFrms)
script_lines.append(script_line)

### intermezzo
script_line =   """
                    In this rally, we have seen a serve and topspin backhands only; let's verify that ATMAN also automatically recognizes
                    forehands, and is able to tell a topspin shot from a slice.
                """ ### 10 secs
intermezzo_early_end = 25
start, stop = segBFs[seg_idx].last()[0], segBFs[seg_idx + 1].first()[0] - intermezzo_early_end
interStm = Stream(cap, start, stop - 1)
# interStm.show()
interFrms = interStm.decap().frames
interFrms = vis_ut.speed_up_frames_progressively(interFrms, ramp_up_frames = 100)
interFrms = interFrms.rescale(3)
# DecapStream(interFrms, 0, 100).show()

all_frames.append(interFrms)
script_lines.append(script_line)

### intermezzo slow down
script_line = ""
start, stop = stop, stop + intermezzo_early_end
interStm = Stream(cap, start, stop - 1)
# interStm.show()
interFrms = interStm.decap().frames
interFrms = BGRFrames.from_frames([interFrms[:15].slow(2), interFrms[15:].slow(2).slow(2)])
interFrms = interFrms.rescale(3)

all_frames.append(interFrms)
script_lines.append(script_line)

# DecapStream(interFrms, 0, 100).show()

### next rally
script_line =   """
                    The serve is recognized...
                """
duration = 20
start, stop = 0, duration
currFrms = regFrms2[start: stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)


script_line =   """
                    The topspin forehand return is also recognized...
                """
duration = 25
start, stop = stop, stop + duration
currFrms = regFrms2[start:stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   """
                    The slice is recognized, but with a small hiccup: 
                    in the windup, the model thinks it sees a low-confidence topspin backhand, but then quickly corrects to slice.
                """ # 9 seconds
duration = 33
start, stop = stop, stop + duration
currFrms = regFrms2[start: stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)


script_line =   """
                    The next slice is flawlessy recognized, with the highest certainty coming around the time of impact.
                """
duration = 32
start, stop = stop, stop + duration
currFrms = regFrms2[start: stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)


script_line =   """
                    Once more, the slice recognized, despite some uncertainty in the windup.
                """
duration = 40
start, stop = stop, stop + duration
currFrms = regFrms2[start: stop]
currFrms = currFrms.slow(2).slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   ""
duration = 10
start, stop = stop, stop + duration
currFrms = regFrms2[start: stop]
currFrms = currFrms.slow(2).slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

script_line =   ""
duration = 10
start, stop = stop, stop + duration
currFrms = regFrms2[start: stop]
currFrms = currFrms.slow(2)

all_frames.append(currFrms)
script_lines.append(script_line)

script_line = "We now have all the ingredients necessary to reconstruct the trajectory of the ball in flight." # 5 seconds
fadeFrms = vs.interpolate_frames(currFrms[-1], currFrms[-1].zero(), 150)

all_frames.append(fadeFrms)
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
