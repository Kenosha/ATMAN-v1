
import pandas as pd
        

from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D

from ATMANpy.types.Abstract import Points, Shots, read_in_data_table, read_in_placeholder_table

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa

import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

import ATMANpy.auxil.constants as const

###### FUNCTIONS #####
def map_label_to_explicit(label):
    hand_map = {'lf': 'Left-Handed',
                'rt': 'Right-Handed'}
    side_map = {'fh': 'Forehand',
                'bh': 'Backhand'}
    shot_map = {'gs': 'Groundstroke',
                'ov': 'Overhead',
                'vl': 'Volley'}
    spin_map = {'ts': 'Topspin',
                'sf': 'Slice'}

    player, what = label.split('_', 1)

    if what == 'oth':
        return 'Other'
    
    side, rest = what.split('_', 1)

    if side == 'srv':
        handStr = hand_map[rest]
        return f'Serve ({handStr})'

    hand, spin, shot = rest.split('_')

    sideStr = side_map[side]
    shotStr = shot_map[shot]
    spinStr = spin_map[spin]
    handStr = hand_map[hand]

    return f'{sideStr} {spinStr} {shotStr} ({handStr})'

def map_desc(desc):
    side_map = {'fh': 'Forehand',
                'bh': 'Backhand'}
    spin_map = {'ts': 'Topspin',
                'sf': 'Slice'}

    if desc == 'srv':
        return 'Serve'
    elif desc == 'ovh':
        return 'Overhead'
    else:
        side, spin = desc.split('_')
        sideStr = side_map[side]
        spinStr = spin_map[spin]
        return f'{sideStr} {spinStr}'

def get_shot_string(myShot):
    if myShot.is_serve():
        dirStr = 'Ad' if myShot.is_serve_ad() else 'Deuce'
        kindStr = 'Serve (Let)' if myShot.is_let() else 'Serve'
        whichStr = 'First' if myShot.is_first_serve() else 'Second'
        return f'{whichStr} {kindStr} ({dirStr})'
    else:
        descStr = map_desc(myShot.get_hit_event_descriptor())
        return descStr
#######

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

SHOW = False
DO_LINES = False
DO_PLAYERS = False
DO_BALL = False
DO_SOUND = False
DO_VISION = False
DO_SEGMENTATION = False
DO_SEGMENTATION_FINE = True
DO_TRAJECTORY = True

################################
######### COURT LINES ##########
################################

vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
fs, ls = conf['start_frame'], conf['stop_frame']
cap = conf['cap']

segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()

seg_idx = 0
my_segBF = segBFs[seg_idx]

widen_frames = 30

myStm = Stream.from_SF(my_segBF, cap).widen(widen_frames)
myDtm = myStm.decap()
myFrms = myDtm.frames

fs, ls = myStm.start_frame, myStm.stop_frame

P = ConstFeature.from_store(vid, sto, featname = 'P').feature

if DO_LINES:
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    H = HSF.feature
    courtLines = sa.find_courtlines(H)
    courtlineSF = ConstFeature('courtLines', courtLines)
    courtlineSF_with_net = ConstFeature('courtLines', courtLines.add_singles_netline())

    if SHOW: 
        myStm.show()

    showStm = myStm.attach(courtlineSF)
    if SHOW: 
        showStm.show()

    showStm = myStm.attach(courtlineSF_with_net)
    if SHOW: 
        showStm.show() ## ? where is the net

## Question: Do we want also the substeps? Hough Line Finding, etc.

###########################
########## PLAYERS ########
###########################
if DO_PLAYERS: 

    foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])

    cache_size = prm_mod.player_candidates.params['frame_blur']

    burnCtms = StreamList.from_store('foregroundMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask',
                                        cache_size = cache_size, force_fill_cache = True)
    shadCtms = StreamList.from_store('shadowMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask',
                                        cache_size = cache_size, force_fill_cache = True)

    if SHOW: 
        burnCtms[seg_idx].show()
    if SHOW: 
        shadCtms[seg_idx].show()

    if SHOW: 
        showStm = myStm.attach(foreSF)
    if SHOW: 
        showStm.show()

    showStm = myStm.attach(plySF)
    if SHOW: 
        showStm.show()


###########################
########## BALL ###########
###########################
if DO_BALL: 
        
    ####### court_masks
    motiStms = StreamList.from_store('fine_motMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 
    if SHOW: 
        motiStms[seg_idx].show()

    ####### ball_filtered_motion_mask
    motiStms_filt = StreamList.from_store('fine_motMask_filt', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 
    if SHOW: 
        motiStms_filt[seg_idx].show()

    ####### ball_contours
    cntSF = StreamFeature.from_store(vid, sto, featname = 'movCnts')
    showStm = myStm.attach(cntSF)
    if SHOW: 
        showStm.show()

    ####### ball_inference_svm
    cntSF = StreamFeature.from_store(vid, sto, featname = 'cand2Cnts') 
    showStm = myStm.attach(cntSF)
    if SHOW: 
        showStm.show()

    ### TODO: need the particles themselves, to explain particle filter? maybe with a better example.
    ### (in segment 0, the ball is perfectly found by the SVM)

    ####### ball_inference_particles
    pntSF = StreamFeature.from_store(vid, sto, featname = 'ball_choicePt') 
    showStm = myStm.attach(pntSF)
    if SHOW: 
        showStm.show()

    ####### ball_inference_hit_candidates
    hitSF = StreamFeature.from_store(vid, sto, featname = 'hitCnt') 
    showStm = myStm.attach(hitSF)
    if SHOW: 
        showStm.show()


##############################
####### SOUND AND VISION #####
##############################

if DO_SOUND:

    audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)

    from ATMANpy.models.SoundCNN.data import waveform_generator, point_sound_generator
    from ATMANpy.models.SoundCNN.nets import SoundModel_MN
    from ATMANpy.models.SoundCNN.utils import load_cnn, first_window_with_sample, last_window_with_sample

    model_params = prm_mod.hits_classify_sound.model_params
    model_name = model_params['model_name']

    model, params = load_cnn(model_name, SoundModel_MN, device = 'cpu')
        
    wave_gen = waveform_generator(vid, sto, yield_frame_indices = True)
    spec_gen = point_sound_generator(vid, sto, mel_bands = params['mel_bands'], 
                                    hop_length = params['hop_length'], window_length = params['window_length'],
                                    to_db = params['to_db'], norm_value = params['norm_value'], db_reference = params['db_reference'],
                                    yield_frame_indices = False)

    my_wave, fs_wave, ls_wave = next(wave_gen)
    my_spec, fs_spec, ls_spec = next(spec_gen)
    my_spec = my_spec[0]

    import numpy as np
    from ATMANpy.types.Overlays import ScatterChart, LineChart, Graph
    from ATMANpy.types.Visual3D import View

    from ATMANpy.types.Frames import GreyFrame, BGRFrame

    spec_min_color = 0 ### grey scale!
    spec_max_color = 255 ## grey scale
    spec_min, spec_max = my_spec.min(), my_spec.max()

    window_width_in_seconds = 0.2
    sr = 44100
    spectrogram_wing_size = 100
    spectrogram_fader_left = np.linspace(0, 1, spectrogram_wing_size) ** 2
    spectrogram_fader_right = np.linspace(1, 0, spectrogram_wing_size) ** 2
    chart_dims = (1920 // 2, 1080 // 2)
    wav_color = const.BGR['bluey_green']
    hit_color = const.BGR['rose_orange']
    bnc_color = const.BGR['soft_yellow']
    now_line_points = 400

    start_seconds = 0
    wave_graphs = []
    prob_graphs = []
    spec_graphs = []
    for frame in range(fs_wave, ls_wave + 1):
        stop_seconds = start_seconds + window_width_in_seconds
        start_sample = int(start_seconds * sr)
        stop_sample = int(stop_seconds * sr)

        snippet = my_wave[start_sample:stop_sample]
        xs = np.linspace(0, 1, num = len(snippet))
        waveChart = ScatterChart(xs, snippet, color = wav_color)

        nowChart = ScatterChart(np.ones(now_line_points) * 0.5, np.linspace(-1, 1, now_line_points), color = const.BGR['intense_red'])

        waveGraph = Graph([waveChart, nowChart], make_default_x_labels = False, as_density = False, resolution = 20,
                            tick_label_size = 0, color = const.BGR['bright_grey'], dimensions = chart_dims, offset = 0.25,
                            header_text = 'Sound Waveform', header_height = 1, header_size = 2, header_horizontal_offset = 10,
                        )
        waveScn, annPnts = waveGraph.preview()
        waveScn = waveScn.drop(1) ### drop the y axis

        now_annPnt = annPnts[2].retext('Now').back(12.3).right(2.3).recolor(const.BGR['intense_red'])
        annPnts.append(now_annPnt)

        waveView = View.make(waveGraph.cam, waveScn, annotatedPoints = annPnts, resolution = 20)
        waveFrm = waveView.snap()
        wave_graphs.append(waveFrm)

        start_frame = fs_wave + start_seconds * 25
        stop_frame = fs_wave + stop_seconds * 25
        # print(start_frame, stop_frame) ### make sure to choose window width so that these line up with exact frames, not fractional
        start_frame = int(start_frame)
        stop_frame = int(stop_frame)

        sub_audFrm = audFrm.loc[start_frame - 1:stop_frame]
        subArr = sub_audFrm.values
        prev_val, inner_vals, next_val = subArr[0], subArr[1:-1], subArr[-1]
        
        # note: since window width lines up with exact frames (see above), 
        # the first row of inner vals is the frame that begins exactly at the beginning of the waveform snippet.
        # the last row of inner vals is the frame that ends with exactly the ending of the waveform snippet.
        # therefore, to find the points that line up with the beginning and ending of the snippet, 
        # we must calculate the mean value between prev_val and the first inner_val, and the last inner_val and next_val
        # these values are then displayed at 0 and 1...
        y0 = (prev_val + inner_vals[0]) / 2
        y1 = (inner_vals[-1] + next_val) / 2
        subArr_enh = np.vstack([y0, inner_vals, y1])    
        hitArr, bncArr = subArr_enh[:, 0], subArr_enh[:, 1]

        # ... whereas the inner values are displayed centered at the point that corresponds to the center of the frame.
        # this is achieved by observing that y0 and y1 must be spaced half as far away from their right- and left neighbours respectively
        # compared to the inner values. This is because y0 and y1 represent values at a half-step between two frames; whereas the inner values are 
        # at the frames center.
        # therefore, since the window contains exactly window_width * 25 frames, which rounds perfectly to len(inner_vals) (because of lining up...)
        num_frames = len(inner_vals)
        single_frame_share = (1 / num_frames)
        x0, x1 = single_frame_share / 2, 1 - single_frame_share / 2

        inf_inner_xs = np.linspace(x0, x1, num = num_frames)
        inf_xs = np.array([0, *inf_inner_xs, 1])

        hitChart = LineChart(inf_xs, hitArr, color = hit_color, thickness = 0.05)
        bncChart = LineChart(inf_xs, bncArr, color = bnc_color, thickness = 0.05)
        
        nowChart = ScatterChart(np.ones(now_line_points) * 0.5, np.linspace(0, 1, now_line_points), color = const.BGR['intense_red'])

        probGraph = Graph([hitChart, bncChart, nowChart], make_default_x_labels = False, as_density = False, resolution = 30,
                        tick_label_size = 1, color = const.BGR['bright_grey'], dimensions = chart_dims, offset = 0.25,
                        y_label = 'Prob. of\nEvent', y_label_format = '>4.0%', y_label_dist = 1.35, label_size = 1,
                        header_text = 'Convolutional Neural Network Inference', header_height = 1, header_size = 2, header_horizontal_offset = 4)
        probScn, annPnts = probGraph.preview()

        probScn = probScn.drop(1) ### drop y axis

        now_annPnt = annPnts[0].retext('Now').right(10.4).back(1).resize(2).recolor(const.BGR['intense_red'])
        annPnts.append(now_annPnt)

        probView = View.make(probGraph.cam, probScn, annotatedPoints = annPnts, resolution = 20)
        probFrm = probView.snap()

        prob_graphs.append(probFrm)

        ### get the spectrogram
        # a) central -> what is happening currently
        fs_win = first_window_with_sample(start_sample, params['hop_length'], params['window_length'])
        ls_win = last_window_with_sample(stop_sample, params['hop_length'], params['window_length'])

        curr_spec = my_spec[:, fs_win: ls_win]
        norm_spec = (curr_spec - spec_min) / (spec_max - spec_min)
        grey_spec = norm_spec * spec_max_color

        curr_specFrm = GreyFrame(grey_spec.astype('uint8'))

        # b) previous -> what happened before
        pre_win = max(0, fs_win - spectrogram_wing_size)
        pre_spec = my_spec[:, pre_win:fs_win]
        norm_pre_spec = (pre_spec - spec_min) / (spec_max - spec_min)
        grey_pre_spec = norm_pre_spec * spec_max_color

        pre_specFrm = GreyFrame(grey_pre_spec.astype('uint8'))
        pre_specFrm = pre_specFrm.pad(to_width = spectrogram_wing_size, width_how = 'right')

        pre_fadeArr = (pre_specFrm.array * spectrogram_fader_left)
        pre_specFrm = GreyFrame(pre_fadeArr.astype('uint8'))

        # c) post -> what happens in a bit.
        post_win = ls_win + spectrogram_wing_size
        post_spec = my_spec[:, ls_win:post_win]
        norm_post_spec = (post_spec - spec_min) / (spec_max - spec_min)
        grey_post_spec = norm_post_spec * spec_max_color

        post_specFrm = GreyFrame(grey_post_spec.astype('uint8'))
        post_specFrm = post_specFrm.pad(to_width = spectrogram_wing_size, width_how = 'left')

        post_fadeArr = (post_specFrm.array * spectrogram_fader_right)
        post_specFrm = GreyFrame(post_fadeArr.astype('uint8'))

        # d) stitch them all together
        specArr = np.hstack([pre_specFrm.array, curr_specFrm.array, post_specFrm.array])
        specFrm = GreyFrame(specArr)
        spec_graphs.append(specFrm)

        start_seconds += (1 / 25)


    ####### MAKE 4 FRAMES -> all these are 1920 // 2, 1080 // 2, we then resize them by a factor.

    ### 1) Wave Graph
    wave_graphFrms = BGRFrames.from_single_frames(wave_graphs)
    DecapStream(wave_graphFrms, 0, 100).show(ms_between_frames = 100)

    ### 2) Probability Graph
    prob_graphFrms = BGRFrames.from_single_frames(prob_graphs)
    DecapStream(prob_graphFrms, 0, 100).show(ms_between_frames = 100)

    ### 3) Mel Spectrogram
    # we make a graph so that we can superimpose our mel spectrogram with consistent header size, etc.
    melGraph = Graph([], make_default_x_labels = False, as_density = False, resolution = 20,
                            tick_label_size = 0, color = const.BGR['bright_grey'], dimensions = chart_dims, offset = 0.7,
                            header_text = 'Mel Spectrogram', header_height = 1, header_size = 2, header_horizontal_offset = 9.5)
    melFrm = melGraph.render()
    melFrms = BGRFrames.from_frame(melFrm, prob_graphFrms.length())

    spec_graphs_padded = [myFrm.pad(to_width = 20 + 2 * spectrogram_wing_size, width_how = 'left') for myFrm in spec_graphs]
    specFrms = GreyFrames.from_single_frames(spec_graphs_padded)

    specArr_remapped = specFrms.array[:, :, :, np.newaxis] / 255 * np.array(wav_color)[np.newaxis, np.newaxis, np.newaxis]
    specFrms_remapped = BGRFrames(specArr_remapped)
    specFrms_remapped_rescaled = specFrms_remapped.rescale(3.5)

    over_cols, over_rows = (melFrms.cols() - specFrms_remapped_rescaled.cols()) // 2, (melFrms.rows() - specFrms_remapped_rescaled.rows()) // 2
    tlPt = Point2D((over_cols, over_rows + 45))

    specFrms_over = overlay(melFrms, specFrms_remapped_rescaled, top_left_point = tlPt)
    DecapStream(specFrms_over, 0, 100).show()

    from ATMANpy.funcs.vis_funcs import animate_sequence
    nowLn = probScn[-1]
    specFrms_over_with_line, _ = animate_sequence([probScn[-1] for i in range(specFrms_over.length())], annotated_points = [now_annPnt], 
                                    start_cam = probGraph.cam, stop_cam = probGraph.cam, animate_over = specFrms_over)
    DecapStream(specFrms_over_with_line, 0, 100).show()

    ### 4) Legend
    from ATMANpy.types.Overlays import Legend, LegendRow
    hitRow = LegendRow('... Hit', hit_color, text_size = 2, text_color = (255, 255, 255))
    bncRow = LegendRow('... Bounce', bnc_color, text_size = 2, text_color = (255, 255, 255))
    myLeg = Legend([hitRow, bncRow], legend_dimensions = (1920 // 3, 1080 // 3), borders = False, header_text = 'Sound identified as...',
                    header_offset = 4.3, header_thickness = 1, resolution = 200)
    legFrm = myLeg.render()
    legFrm = legFrm.pad(to_height = 1080 // 2, to_width = 1920 // 2)

    #### resize to cols, rows

    rescale_ratio = 0.4

    myRst = FramesRaster(2, 2)

    myRst.add(wave_graphFrms.rescale(rescale_ratio), 0, 0)
    myRst.add(prob_graphFrms.rescale(rescale_ratio), 1, 0)
    myRst.add(specFrms_over_with_line.rescale(rescale_ratio), 0, 1)
    myRst.add(legFrm.rescale(rescale_ratio), 1, 1)

    showFrms = myRst.render()
    DecapStream(showFrms, 0, 100).show(ms_between_frames = 100)

    myFrms_full = myFrms.rescale(3)

    tl_x, tl_y = myFrms_full.cols() - showFrms.cols() - 1, 0
    tlPt = Point2D((tl_x, tl_y))

    overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)

    import sounddevice as sd
    sd.play(my_wave, samplerate = sr)
    DecapStream(overFrms, 0, 100).show(ms_between_frames = 40)

    from time import sleep
    sleep(2)
    DecapStream(overFrms, 0, 100).show(ms_between_frames = 100)


### TODO: slight color change when a bounce or a hit is detected
### TODO: Show moving spectrogram... how?
### TODO: legend (hit/bounce color!)


if DO_VISION:    
    from ATMANpy.types.Contours import Contours
    from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
    from ATMANpy.models.TemporalCNN.utils import load_cnn
    
    model_params = prm_mod.hits_classify_video.model_params
    top_mod = model_params['top']
    model, params = load_cnn(top_mod, SmallFullConvModel, device = 'cpu')

    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])

    topFrm = ff.read_feature_frame('top_hitTable', vid, sto)
    botFrm = ff.read_feature_frame('bot_hitTable', vid, sto)

    sub_topFrm = topFrm.loc[fs:ls]
    sub_botFrm = botFrm.loc[fs:ls]

    ### fold bot and top into correct handedness -> this is done automatically in the code.
    def adjust_probs_for_handedness(someFrm, hand = 'right'):
        right_cols, left_cols = someFrm.columns.str.contains('_rt'), someFrm.columns.str.contains('_lf')
        
        rtFrm = someFrm.loc[:, right_cols]
        lfFrm = someFrm.loc[:, left_cols]
        
        rtFrm.columns = rtFrm.columns.str.replace('_rt', '_hand')
        lfFrm.columns = lfFrm.columns.str.replace('_lf', '_hand')

        if hand == 'right':
            base_probs = rtFrm.copy()
            repl_probs = lfFrm.copy()
        elif hand == 'left':
            base_probs = lfFrm.copy()
            repl_probs = rtFrm.copy()

        repl_probs.columns = repl_probs.columns.str.replace('_bh', '_chng').str.replace('_fh', '_bh').str.replace('_chng', '_fh') ### triple swap to avoid replacing same row twice

        probs = base_probs + repl_probs

        if hand == 'right':
            probs.columns = probs.columns.str.replace('_hand', '_rt')
        elif hand == 'left':
            probs.columns = probs.columns.str.replace('_hand', '_lf')

        notFrm = someFrm.loc[:, (~right_cols) & (~left_cols)]

        return pd.concat([probs, notFrm], axis = 1)

    sub_topFrm = adjust_probs_for_handedness(sub_topFrm, hand = 'right')
    sub_botFrm = adjust_probs_for_handedness(sub_botFrm, hand = 'right')

    T = params['T_CROP_SIZE'] + 1 ### add one so there is a clear "central" frame.
    D = params['D_CROP_SIZE']

    my_plySF = plySF.conform(my_segBF)

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

    DecapStream(regFrms, 0, 100).show(ms_between_frames = 100)

    # they don't look to be the same area...
    # for side in ['bot', 'top']:
    #     print('____')
    #     print(side)
        
    #     plregDict = allDict[side]
    #     for frame in plregDict:
    #         reg = plregDict[frame]
    #         print(reg.area())
    # ... but they are!!!!!! strange.

    ### SHOW DIMMED FRAME WITH BRIGHT REGIONS
   
    scale_factor = 3
    dim_factor = 0.5
    p_threshold = 0.1
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
        top_best_str = map_label_to_explicit(top_best_evt)
        if top_best_prb > p_threshold:
            topStr = f'{top_best_str}: {top_best_prb:.0%}'
        else:
            topStr = ''

        botSer_event = botSer[botSer.index != 'bot_oth']
        bot_best_idx = botSer_event.argmax()
        bot_best_evt = botSer_event.index[bot_best_idx]
        bot_best_prb = botSer_event[bot_best_evt]
        bot_best_str = map_label_to_explicit(bot_best_evt)
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
    DecapStream(regFrms, 0, 100).show(ms_between_frames = 100)

if DO_SEGMENTATION:
    ### SEGMENTATION ###
        # in general, segment often during the pipeline, so the recognition system can be parallelized
            ### -> work on many "segments" (not necessarily points) at the same time!
        # in this pipeline: three times:
            # once with lines, to segment into sets
            # then again with sound, to find stretches where sound and bounce sounds can be made out
            # then finally with the visual info: every point begins with a serve, which is visually detected.
        # however, the system pipeline is highly modular and can be adapted to the realities of the video data used, 
        # so any subtask, not just segmentation but also recognition, can be changed, adapted, removed or added.
    myRst = FramesRaster(3, 3)
    
    frmsList = []
    for idx in range(9):
        segStm = Stream.from_SF(segBFs[idx], cap).widen(widen_frames)
        segFrms = segStm.decap().frames
        frmsList.append(segFrms)

    # our example exchange - a short rally.
    segFrms = frmsList[0]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100) ### returns itself!
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 0, 0)

    # next up: a longer rally ## -> have to speed it up a bit...
    segFrms = frmsList[1]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 0, 1)
    
    # another long rally
    segFrms = frmsList[2]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 0, 2)
    
    # a let on first serve. 
    # Is this auto-recognized? i dont think so. in that case: say it! emphasize that:
        # 1) the let is flagged (because we see three serves in a row, or we see a serve in that is not an ace.)
        # 2) the let is manually checked after flagging. the manual checking script is simple to integrate into workflow, and requires very few actual checks.
    allPnts[3][0].get_hit_event_info() ### -> this is registered as a Let!
    segFrms = frmsList[3]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 1, 0)

    # first serve re-do
    segFrms = frmsList[4]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 1, 1)

    # second serve
    segFrms = frmsList[5]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 1, 2)

    # a rally -> widen!
    segFrms = frmsList[6]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 2, 0)

    # first serve fault
    segFrms = frmsList[7]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 2, 1)
    
    # a return error.
    segFrms = frmsList[8]
    segFrms = vis_ut.speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100)
    DecapStream(segFrms, 0, 100).show()
    myRst.add(segFrms, 2, 2)

    ### put it all together
    rstFrms = myRst.render()
    DecapStream(rstFrms, 0, 100).show()

if DO_SEGMENTATION_FINE:
    ### SEGMENTATION INTO SHOTS ###
    # -> once we have a point, we must segment the point into shots, that comprise it.
    # -> to this end, we combine the timing information of the sound and the vision to detect when a ball has been hit.
    # -> from the visual recognition, and a simple viterbi algorithm (the ball is hit in alternating manner), we can easily know which player (top or bot) has hit the ball.
    # -> we use player location to have a rough idea of where the ball has been hit, at that point.
    ### -> this way, we combine all available information in a consistent manner. 
    # The modular nature of the pipeline easily allows for adding or changing sources of truth,
    # which can be flexibly combined (using the viterbi algorithm.)
    dim_factor = 0.5

    myPnt = allPnts[0]

    sht0, sht1, sht2 = myPnt[0], myPnt[1], myPnt[2]
    stm0, stm1, stm2 = sht0.stream(cap), sht1.stream(cap), sht2.stream(cap)
    
    frms0, frms1, frms2 = stm0.decap().frames.rescale(1.5), stm1.decap().frames.rescale(1.5), stm2.decap().frames.rescale(1.5)
    frms0_ls, frms1_ls, frms2_ls = BGRFrames.from_frame(frms0[-1].dim(dim_factor), 25), BGRFrames.from_frame(frms1[-1].dim(dim_factor), 25), BGRFrames.from_frame(frms2[-1].dim(dim_factor), 25)


    str0, str1, str2 = get_shot_string(sht0), get_shot_string(sht1), get_shot_string(sht2)

    tab_params = {  'index_offset': 1, 'index_col_width': 0.5, 'dimensions': (960, 540), 'skeleton_shrinkage': 0.2, 
                    'automatic_vertical_offset_text_line_height': 0.4,
                    'null_columns': 1
                    }

    tab_data0 = {'1st Shot': {'Description': str0}}
    tab_data1 = {'1st Shot': {'Description': str0}, '2nd Shot': {'Description': str1}}
    tab_data2 = {'1st Shot': {'Description': str0}, '2nd Shot': {'Description': str1}, '3rd Shot': {'Description': str2}}
    
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

    myRst = FramesRaster(2, 2)

    myRst.add(frms0, 0, 0)
    myRst.add(tab_frms0, 1, 1)
    rstFrms0 = myRst.render()
    DecapStream(rstFrms0, 0, 100).show()

    myRst.add(frms0_ls, 0, 0)
    myRst.add(frms1, 0, 1)
    myRst.add(tab_frms1, 1, 1)
    rstFrms1 = myRst.render()
    DecapStream(rstFrms1, 0, 100).show()

    myRst.add(frms1_ls, 0, 1)
    myRst.add(frms2, 1, 0)
    myRst.add(tab_frms2, 1, 1)
    rstFrms2 = myRst.render()
    DecapStream(rstFrms2, 0, 100).show()


if DO_TRAJECTORY:
    ### BALL TRAJECTORY ### 
    # Once segmentation into Shots within a point has been established, it remains to calculate the trajectory of the ball.
    # since this task is the most challenging, but also the most important, it is left to the end, when as
    # much of all other data about the tennis point (lines, players, ball candidates, segmentation, shot timing) is known.
    
    # -> issue: ball candidates are 2D -> we don't know exactly where in 3D they are!
    # fix: using the rough idea about where the shot begins and ends, we construct planes on which a ball lies.
    # given that the ball flies within this specific 3D candidate plane, we can construct actual (candidate) trajectory in 3D.
    # this candidate trajectory is checked against:
        # simple unviolable trajectory principles: a ball can not be below ground, it has to be non-monotonous in movement, it can not accelerate past the initial impact.
            # -> does this plane imply a trajectory that's at all possible?
        # a physics model: how does a tennis ball behave when struck, given its launch angle, speed, and imparted spin?
            # -> does this plane imply a trajectory that's physically possible, given the nature of the ball, raquet, air, surface...?
        # the moment at which this plane would imply a bounce -> does it match up with the bounce we heard?
        # the spin implied by the visual recognition system (topspin or slice can be distniguished visually)
            # -> is the observed trajectory one of slice or topspin?
        # does the trajectory imply that the ball is still in play (in the court/not in the net), 
            # when the rally is known to go on for longer (due to the point sub-segmentation)
        # on the other hand, does the trajectory imply that the ball bounces twice, is in the net, or out of play,
            # when the rally is known to end?
        
        ### none of these criteria alone could identify the ball alone, but joined together in a consistent manner, 
        # the likeliest ball trajectory is very often very close to the real trajectory.
        # an advantage of the physics matching is that we get, without extra work as a by-product, the 
        # initial launch angle, speed and spin of the ball.

    dim_factor = 0.5

    myPnt = allPnts[0]
    myPnt.stream_ball(P, cap = cap).show()

    sht0, sht1, sht2 = myPnt[0], myPnt[1], myPnt[2]
    stm0, stm1, stm2 = sht0.stream(cap), sht1.stream(cap), sht2.stream(cap)
    
    frms0, frms1, frms2 = stm0.decap().frames.rescale(1.5), stm1.decap().frames.rescale(1.5), stm2.decap().frames.rescale(1.5)
    
    ### get ball splines
    
    ball_color = const.BGR['ball_yellow']

    splns0 = sht0.feat_spline_cumul(which_endpoint = 'penultimate').values()
    splns1 = myPnt[1].feat_spline_cumul(which_endpoint = 'penultimate').values()
    splns2 = myPnt[2].feat_spline_cumul(which_endpoint = 'penultimate').values()

    from ATMANpy.types.Visual3D import Camera, Scene
    myCam = Camera.calibrated_brest_behind().resize(1.5)

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
    str0, str1, str2 = get_shot_string(sht0), get_shot_string(sht1), get_shot_string(sht2)
    dat0, dat1, dat2 = get_data_string(sht0), get_data_string(sht1), get_data_string(sht2)

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

    ### Check out trajectories for the 9 other examples:
    from ATMANpy.types.Abstract import NullPoint

    def has_nan_point_or_is_none(myLn):
        if myLn is None:
            return True
        else:
            return np.isnan(myLn.leftPt.coords).all() or np.isnan(myLn.rightPt.coords).all()

    raster_frames = []
    for frm_idx, myPnt in enumerate(allPnts[:9]):
        print(frm_idx)

        if isinstance(myPnt, NullPoint):
            stm = Stream.from_SF(segBFs[frm_idx], cap)
            frms = stm.decap().frames
            raster_frames.append(frms)
            continue

        point_splns = myPnt.feat_spline_cumul(which_endpoint = 'penultimate').values()

        ### cull points in the net -> have a nan Point or two.
        ### also cull None

        point_splns = [[spln for spln in splns if not has_nan_point_or_is_none(spln)] for splns in point_splns ]

        scns = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns]) for splns in point_splns]

        stm = myPnt.stream(cap = cap)
        frms = stm.decap().frames[:-1]

        myCam = Camera.calibrated_brest_behind()

        frms_spln, _ = animate_sequence(scns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = frms)

        raster_frames.append(frms_spln)

    for frms in raster_frames:
        DecapStream(frms, 0, 100).show()

