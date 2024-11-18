

from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures


import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
### CHOOSE VIDEO
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
for idx, vid in enumerate(avail_vids):
    print(idx, vid)

idx = 12
conf = cnf_mod.get_conf(avail_vids[idx])

SHOW = False
DO_LINES = False
DO_PLAYERS = False
DO_BALL = False

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

myStm = Stream.from_SF(my_segBF, cap)
myDtm = myStm.decap()
myFrms = myDtm.frames

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

topFrm = ff.read_feature_frame('top_hitTable', vid, sto)
botFrm = ff.read_feature_frame('bot_hitTable', vid, sto)
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

import ATMANpy.auxil.constants as const
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

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames
from ATMANpy.types.Overlays import FramesRaster, overlay
from ATMANpy.types.Points import Point2D

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

DecapStream(overFrms, 0, 100).show(ms_between_frames = 100)


import sounddevice as sd
sd.play(my_wave, samplerate = sr)

### TODO: slight color change when a bounce or a hit is detected
### TODO: Show moving spectrogram... how?
### TODO: legend (hit/bounce color!)




