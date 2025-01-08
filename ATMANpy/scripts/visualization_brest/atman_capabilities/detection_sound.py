
import numpy as np
import pandas as pd
import sounddevice as sd
        
from ATMANpy.funcs.util_funcs import get_folder_struct, split
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

vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
fs, ls = conf['start_frame'], conf['stop_frame']
cap = conf['cap']

segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()

### sound and vision -> segment 3 and 4
seg_idx = 3
fs_segBF = segBFs[seg_idx] 
seg_idx_2 = 4
ls_segBF = segBFs[seg_idx_2]

how_many_frames_in_sound = 750 ### -> how many frames belonging to the intermezzo are needed here (also needed in detection balls)
how_many_frames_in_vision = 100 ### -> how many frames belonging to the intermezzo are needed in the next part (also needed in vision)

###########################
####### SOUND #############
###########################
audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)

from ATMANpy.models.SoundCNN.data import waveform_generator, point_sound_generator
from ATMANpy.models.SoundCNN.nets import SoundModel_MN
from ATMANpy.models.SoundCNN.utils import load_cnn, first_window_with_sample, last_window_with_sample

model_params = prm_mod.hits_classify_sound.model_params
model_name = model_params['model_name']

model, params = load_cnn(model_name, SoundModel_MN, device = 'cpu')
    
wave_gen = waveform_generator(vid, sto, yield_frame_indices = True, widen = how_many_frames_in_sound)
spec_gen = point_sound_generator(vid, sto, mel_bands = params['mel_bands'], 
                                hop_length = params['hop_length'], window_length = params['window_length'],
                                to_db = params['to_db'], norm_value = params['norm_value'], db_reference = params['db_reference'],
                                yield_frame_indices = False, widen = how_many_frames_in_sound)

for _ in range(seg_idx + 1):
    my_wave, fs_wave, ls_wave = next(wave_gen)
    my_spec, fs_spec, ls_spec = next(spec_gen)
my_spec = my_spec[0]

### TODO: narration -> use the opportunity to talk about Let sound not included 
# (which a model could be trained for, and easily included, since the pipeline is very modular.)

######### GRAPH PARAMETERS #########
import numpy as np
from ATMANpy.types.Overlays import ScatterChart, LineChart, Graph
from ATMANpy.types.Visual3D import View

from ATMANpy.types.Frames import GreyFrame, BGRFrame

spec_min_color = 0 ### grey scale!
spec_max_color = 255 ## grey scale
spec_min, spec_max = my_spec.min(), my_spec.max()

window_width_in_seconds = 0.2
window_width_in_frames = window_width_in_seconds * 25
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
    ### TODO: actually, the waveforms, spectrograms and graphs produced are not lined up with the frame in the middle,
    # but with the frame at the start... either fix here or just shift the Stream that is shown by half the window_width_in_seconds * 25 frames
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

#### BEGIN ANIMATING ###
all_frames = []
script_lines = []

# idea: talk over a still frame. introduce all elements: waves, mel, graph.
# then, play sound, and just the sound, for a bit.
# then, say "Currently...model" over the sound, as we approach the hitting moment. 
# since we begin with a still frame, we can greatly reduce the time spent in detection_balls intermezzo: go to freeze frame earlier!
# this gains us a lot of time to actually play the sound, before saying "Currently..."


### raster params
rescale_ratio = 0.4
myRst = FramesRaster(2, 2)
tl_x, tl_y = int((1 - rescale_ratio) * 1920) - 1, 0
tlPt = Point2D((tl_x, tl_y))

#### Intermezzo: introduce graph elements
script_line = '[FADE IN SOUND]'
duration = 100
start, stop = fs_wave, fs_wave + duration

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

all_frames.append(myFrms_full)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND] 
                    The sound of the ball being hit or bouncing on the ground gives us a very good idea of when exactly these events occur.
                    Knowing the timing of hits and bounces greatly improves our ability to construct ball trajectories.
                """
duration = 100
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
# DecapStream(overFrms, 0, 100).show(ms_between_frames = 400)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND]
                    To identify hits and bounces from the sound, we train a neural network on the Mel Spectrogram of the sound waveform.
                """
duration = 100
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
# DecapStream(overFrms, 0, 100).show(ms_between_frames = 40)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND]
                    The neural network spits out the probability that the sound currently being heard is a hit or a bounce,
                    shown here in red and yellow respectively. Currently, the model is correctly detecting neither hit nor bounce. 
                    Look out now for the spike in the red curve as soon as the serve is hit, indicating that a hit has been detected. 
                    This is followed by a spike in the yellow curve, indicating that the bounce has been heard and identified by the model.
                """ ### currently to model -> 17s
duration = 150
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
# DecapStream(overFrms, 0, 100).show(ms_between_frames = 100)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND slowed to half]
                """
duration = 35
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
overFrms = overFrms.slow(2)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND slowed to fourth]
                """
duration = 15
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
overFrms = overFrms.slow(2).slow(2)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND slowed to eigth]
                """
duration = 25
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
overFrms = overFrms.slow(2).slow(2).slow(2)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND slowed to eigth] 
                    In this manner, the model detectes hits and bounces when they occur, for the entire rally.
                """
duration = 15
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
overFrms = overFrms.slow(2).slow(2).slow(2)

all_frames.append(overFrms)
script_lines.append(script_line)


script_line =   """
                    [PLAY SOUND slowed to fourth] 
                """
duration = 15
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
overFrms = overFrms.slow(2).slow(2)

all_frames.append(overFrms)
script_lines.append(script_line)


script_line =   """
                    [PLAY SOUND slowed to half] 
                """
duration = 15
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)
overFrms = overFrms.slow(2)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND] 
                """
duration = 100
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)

all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    [PLAY SOUND] 
                    Note that the serve here is a let: the sound of the umpire declaring the let is not detected,
                    even though a model could be trained for it, and integrated into the detection pipeline.
                """
duration = 100
start, stop = stop, stop + duration
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)

all_frames.append(overFrms)
script_lines.append(script_line)


script_line =   """
                    [FADE OUT SOUND] 
                    Paired with the visual recognition of the ball in flight, the sound information gives us 
                    a very good idea about when and where the ball is hit, and when and where it bounces. 
                    To supplement our knowledge on timing and location even further, and to find out what type of shot is
                    actually being hit, we analyse the hitting motion of both players visually.
                """

start, stop = stop, segBFs[5].first()[0] - how_many_frames_in_vision
start_rect, stop_rect = start - fs_wave, stop - fs_wave

myStm = Stream(cap, start, stop - 1)
myFrms = myStm.decap().frames
myFrms_full = myFrms.rescale(3)
# DecapStream(myFrms_full, 0, 100).show()

curr_wave_graphFrms = wave_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_spec_graphFrms = specFrms_over_with_line[start_rect:stop_rect].rescale(rescale_ratio)
curr_prob_graphFrms = prob_graphFrms[start_rect:stop_rect].rescale(rescale_ratio)
curr_legFrm = legFrm.rescale(rescale_ratio)
myRst.add(curr_wave_graphFrms, 0, 0, set_size = True)
myRst.add(curr_spec_graphFrms, 0, 1, set_size = True)
myRst.add(curr_prob_graphFrms, 1, 0, set_size = True)
myRst.add(curr_legFrm, 1, 1, set_size = True)

showFrms = myRst.render()
overFrms = overlay(myFrms_full, showFrms, top_left_point = tlPt)

all_frames.append(overFrms)
script_lines.append(script_line)

### intermezzo
vis_ut.show(all_frames)
