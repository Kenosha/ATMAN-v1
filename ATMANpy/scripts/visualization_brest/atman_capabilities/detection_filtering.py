"""
In this script, we want to showcase the filtering capabilities of ATMAN.

We can filter for Points and for Shots.

# Point filter ideas:
-> decisive points
-> high pressure points
-> rallyes of certain length.
-> by player
-> ad/deuce side serve
-> all points that begin with a first serve out wide (by player, on ad side?!)

# Shot filter ideas:
-> slice/topspin
-> groundstroke/volley
-> serve/return/rally
-> forehand/backhand
-> certain speeds
-> certain angles (down the line, cross court)
-> launch angle
-> by player

### -> how about:
first, filter by points (i.e. all points that begin with a first serve out wide (by/to player, on ad side?!))
then, filter return shots. --> has to be to player!

"""


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

from ATMANpy.types.Abstract import Points, Shots, read_in_data_tables, read_in_placeholder_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

from ATMANpy.models.ParticleFilter import ParticleFilter
from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points

def get_winners_from_score(retPnts, allPnts):
    winner_codes = []
    for myPnt in retPnts:
        match_idx, point_idx = myPnt.get_match_index(), myPnt.get_point_index()
        nextPnt = allPnts.get(match_idx, point_idx + 1)

        winner = myPnt.get_winner_from_score(nextPnt, verbose = True)

        if winner is None:
            winner_codes.append(0)
        elif winner == myPnt.get_server().get_name():
            winner_codes.append(1)
        elif winner == myPnt.get_returner().get_name():
            winner_codes.append(-1)

    return np.array(winner_codes)

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)


prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
  
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

myCam = Camera.calibrated_brest_behind()

courtScn = Scene.make_court()

ball_color = const.BGR['ball_yellow']

### GET ABSTRACT POINT REPRESENTATION 
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)
allPnts = dataPnts.join(nullPnts).sort()

### FILTER
import ATMANpy.funcs.filter_shots_funcs as filt_s
import ATMANpy.funcs.filter_points_funcs as filt_p

filtPnts1 = dataPnts.filter(filt_p.is_returner, name = 'Brandon Nakashima').filter(filt_p.is_longer_than, threshold = 1)

filtPnts2 = filtPnts1.filter('is_first_serve')
decArr = get_winners_from_score(filtPnts2, allPnts) != 0
filtPnts2 = filtPnts2.filter(decArr)

filtPnts3 = filtPnts2.filter('is_serve_ad')

filtPnts4 = filtPnts3.filter(filt_p.is_serve_out_wide)

### GET RETURNS
retShots = filtPnts4.get_returns()
retShots_rect = retShots.enforce_direction('Top')

slc_retShots = retShots.filter('is_slice_visual')
top_retShots = retShots.filter('is_topspin_visual')

####### TAKE A LOOK AT FILTERED RESULTS

# TODO: only valid first serves! (no let, no fault!)
# TODO: return has been observed
# TODO: serve bounces in other half of the court (two pathological serves...)

filtered_spline_frames = []
for idx, myPnt in enumerate(filtPnts4):
    idx_num = idx + 1
    
    ### SERVE
    srvShot = myPnt[0]
    srvStm = srvShot.stream(confs = confs)
    srvFrms = srvStm.decap().frames
    srv_splns = srvShot.feat_spline_cumul(which_endpoint = 'penultimate').values()
    srvScns = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)]) for splns in srv_splns]
    srv_spln_frms, _ = vs.animate_sequence(srvScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = srvFrms)

    ### RETURN
    try:
        retShot = myPnt[1]
    except IndexError:
        retStm = Stream(srvStm.cap, srvStm.stop_frame, srvStm.stop_frame + 20)
        retFrms = retStm.decap().frames
        ret_spln_frms = retFrms
    else:
        retStm = retShot.stream(confs = confs)
        retFrms = retStm.decap().frames
        ret_splns = retShot.feat_spline_cumul(which_endpoint = 'penultimate').values()
        retScns = [Scene([spln.recolor(ball_color).resize(1.5) for spln in splns if not vis_ut.has_nan_point_or_is_none(spln)]) for splns in ret_splns]
        
        retFrms = retFrms[:len(retScns)]
        ret_spln_frms, _ = vs.animate_sequence(retScns, start_cam = myCam, stop_cam = myCam, cumulate_argument = False, animate_over = retFrms)

    splnFrms = BGRFrames.from_frames([srv_spln_frms, ret_spln_frms])
    splnFrms = splnFrms.rescale(1.5)
     ### add indices later, after sorting, so that we display the correct order.
    # splnFrms_with_text = BGRFrames.from_single_frames([splnFrm.put(f'{idx_num:.0f}', size = 2) for splnFrm in splnFrms])
    # DecapStream(splnFrms, 0, 100).show()
    # DecapStream(splnFrms_with_text, 0, 100).show()
    filtered_spline_frames.append(splnFrms)

### some extra script data
p0, s0 = len(allPnts), sum([len(p) for p in dataPnts])
p1, s1 = len(filtPnts1), sum([len(p) for p in filtPnts1])
p2, s2 = len(filtPnts2), sum([len(p) for p in filtPnts2])
p3, s3 = len(filtPnts3), sum([len(p) for p in filtPnts3])
p4, s4 = len(filtPnts4), sum([len(p) for p in filtPnts4])

pStr, sStr = 'Num. Points', 'Num. Shots'
row0 = {pStr: p0, sStr: s0}
row1 = {pStr: p1, sStr: s1}
row2 = {pStr: p2, sStr: s2}
row3 = {pStr: p3, sStr: s3}
row4 = {pStr: p4, sStr: s4}

##### COUNT TABLES ###

tab_dims = (1920 // 2, 1080 // 2)
kwargs = {'number_format': '>11,', 'index_offset': 0.25, 'index_col_width': 2, 'skeleton_shrinkage': 0.025,
            'dimensions': tab_dims, 'data_text_size': 2, 'index_text_size': 2, 'header_text_size': 2}

table_data = {'All': row0}
myTab = Table(table_data, **kwargs, null_rows = 4)
tabFrm0 = myTab.render()
# tabFrm0.show()

table_data = {'All': row0,
                ' ... Brandon Nakashima returns.': row1}

myTab = Table(table_data, **kwargs, null_rows = 3)
tabFrm1 = myTab.render()
tabFrm1.show()

table_data = {'All': row0,
                ' ... Brandon Nakashima returns ...': row1,
                ' ... a (valid) first serve.': row2}

myTab = Table(table_data, **kwargs, null_rows = 2)
tabFrm2 = myTab.render()
tabFrm2.show()

table_data = {'All': row0,
                ' ... Brandon Nakashima returns ...': row1,
                ' ... a (valid) first serve ...': row2,
                ' ... from the ad side.': row3}

myTab = Table(table_data, **kwargs, null_rows = 1)
tabFrm3 = myTab.render()
tabFrm3.show()

table_data = {'All': row0,
                ' ... Brandon Nakashima returns ...': row1,
                ' ... a (valid) first serve ...': row2,
                ' ... from the ad side ...': row3,
                ' ... served out wide.': row4}

myTab = Table(table_data, **kwargs)
tabFrm4 = myTab.render()
tabFrm4.show()

### AGG TABLES ###

retShots = filtPnts4.get_returns()
retShots_fair = retShots.filter('is_fair', nextShot = None)

avg_speed = np.mean(retShots.get_initial_speed())
avg_spin = np.mean(retShots.get_initial_spin())
avg_depth = np.mean([d for d in retShots_fair.get_bounce_distance_from_baseline() if not np.isnan(d)]) / const.SCALE * const.FOOT_TO_METER

winArr = get_winners_from_score(filtPnts4, allPnts)
ret_wins = (winArr == -1)
cnt_ret_wins = ret_wins.sum()
all_ret_wins = len(ret_wins)
avg_ret_wins = cnt_ret_wins / all_ret_wins

tab_dims = (1920 // 2, 1080 // 2)
kwargs = {'number_format': '>11,', 'index_offset': 0.25, 'index_col_width': 1, 'skeleton_shrinkage': 0.025,
            'dimensions': tab_dims, 'data_text_size': 2, 'index_text_size': 2, 'header_text_size': 2}

col_name = 'Return Statistics\n(Simple Averages)'
row0 = {col_name: f'{avg_speed:.0f} km/h'}
row1 = {col_name: f'{avg_spin:.0f} RPM'}
row2 = {col_name: f'{avg_depth:.1f} meters'}
row3 = {col_name: f'{avg_ret_wins * 100:.0f}% ({cnt_ret_wins} of {all_ret_wins})'}

table_data = {  'Speed': row0}
aggTab1 = Table(table_data, **kwargs, null_rows = 3)
aggFrm1 = aggTab1.render()
aggFrm1.show()

table_data = {  'Speed': row0,
                'Spin': row1}
aggTab2 = Table(table_data, **kwargs, null_rows = 2)
aggFrm2 = aggTab2.render()
aggFrm2.show()

table_data = {  'Speed': row0,
                'Spin': row1,
                'Dist. to Baseline': row2}
aggTab3 = Table(table_data, **kwargs, null_rows = 1)
aggFrm3 = aggTab3.render()
aggFrm3.show()

table_data = {  'Speed': row0,
                'Spin': row1,
                'Dist. to Baseline': row2,
                'Returner Win%': row3}
aggTab4 = Table(table_data, **kwargs, null_rows = 0)
aggFrm4 = aggTab4.render()
aggFrm4.show()


### TABLE POSITIONING ###
zeroFrm = BGRFrame(np.zeros((1080, 1920, 3), dtype = 'uint8'))

tlPt_tab = (Point2D((1920, 1080)) - Point2D(tab_dims)) // 2
tlPt_tab_corner = Point2D((0, 0))

#################
### ANIMATION ###
#################

all_frames = []
script_lines = []

script_line =   f"""
                    Having detected lines, players and trajectories of the ball, and having segmented every Match into 
                    distinct Points and Shots, we now show how the data gathered is accessed directly within the ATMAN framework.

                    We have analysed 20 matches played in the course of the 2021 Brest challenger tournament; 
                    in the course of these 20 matches, {p0} points and {s0} distinct shots have been played and detected. 
                """
zeroFrms = BGRFrames.from_frame(zeroFrm, 25)
all_frames.append(zeroFrms)
script_lines.append(script_line)

script_line =   """
                    In the face of this large amount of data even for just a single tournament, 
                    is imperative to be able to search and filter the database easily.

                    Luckily, this is possible directly within ATMAN.
                """

overFrm = overlay(zeroFrm, tabFrm0, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    To showcase how, consider a simple use case: let's say we are interested in the return behaviour of one specific
                    player, for example Brandon Nakashima. This, and a myriad more filters can be done in a single line of code.
                    Should the user wish to add a filter, he can easily do so due to the intuitive representation
                    of Trajectories, Players, Shots and Points in the dataclass system of ATMAN.
                """

overFrm = overlay(zeroFrm, tabFrm1, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)

script_line =   """
                    Let us further narrow down the scope of the exercise by concentrating on valid first serves...
                """

overFrm = overlay(zeroFrm, tabFrm2, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)


script_line =   """
                    From the ad side...
                """

overFrm = overlay(zeroFrm, tabFrm3, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)


script_line =   f"""
                    That are targeted out wide, at Nakashima's backhand. This leaves us with {p4} points.
                """

overFrm = overlay(zeroFrm, tabFrm4, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)


### move the table into the corner
script_line =   """
                    Having filtered with ease, Let's quickly check the remaining serves and returns.
                """

itplPts = [tlPt_tab * (1 - lmbd) + tlPt_tab_corner * lmbd for lmbd in np.linspace(0, 1, 50, endpoint = True)]
itplPts = [itplPt.round_point() for itplPt in itplPts]
itpl_frames = []
for tlPt in itplPts:
    overFrm_itpl = overlay(zeroFrm, tabFrm4, top_left_point = tlPt)
    itpl_frames.append(overFrm_itpl)

itplFrms = BGRFrames.from_single_frames(itpl_frames)
# DecapStream(itplFrms, 0, 100).show()
all_frames.append(itplFrms)
script_lines.append(script_line)

### show rasters of returns made
script_line =   f"""
                    Since there are {p4} points that match our filter, we show them in groups of three.
                """

myRst = FramesRaster(2, 2)
tabFrms = BGRFrames.from_frame(tabFrm4, 25)
myRst.add(tabFrms, 0, 0)

filtered_spline_frames_sorted = sorted(filtered_spline_frames, key = lambda frms: frms.length(), reverse = False)
filtered_spline_frames_sorted = [frms.put(f'{idx + 1}', size = 3, thickness = 2) for idx, frms in enumerate(filtered_spline_frames_sorted)]

raster_frames = []
for start in range(0, 27, 3):
    frms1, frms2, frms3 = filtered_spline_frames_sorted[start:start+3]
    myRst.add(frms1, 0, 1)
    myRst.add(frms2, 1, 0)
    myRst.add(frms3, 1, 1)
    rstFrms = myRst.render()
    raster_frames.append(rstFrms)

rstFrms = BGRFrames.from_frames(raster_frames)
# DecapStream(rstFrms, 0, 100).show()

all_frames.append(rstFrms)
script_lines.append(script_line)

### aggregates
script_line =   """
                    On our filtered data, we can easily produce aggregate statistics...
                """

from ATMANpy.types.Overlays import reverse_cascade
_, revFrms = reverse_cascade(myRst)
all_frames.append(revFrms)
script_lines.append(script_line)

script_line =   """
                    ... such as, for example, the average return speed...
                """
overFrm = overlay(zeroFrm, aggFrm1, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)

script_line = "  ... spin ... "
overFrm = overlay(zeroFrm, aggFrm2, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)

script_line = " ... depth ... "
overFrm = overlay(zeroFrm, aggFrm3, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)

script_line = " and find out how many of these 27 points Brandon Nakashima won. "
overFrm = overlay(zeroFrm, aggFrm4, top_left_point = tlPt_tab)
overFrms = BGRFrames.from_frame(overFrm, 25)
all_frames.append(overFrms)
script_lines.append(script_line)

### scatter map/heat map

script_line =   """
                    Beyond these simple numeric averages, ATMAN enables us to display visual representations of data.
                """
### fade in a point up until moment of return that has Brandon Nakashima on far side.
pt_idx = 6
ls_frame = 16
pre_frames = 150
total_frames = ls_frame + pre_frames

# the spline
exFrms = filtered_spline_frames[pt_idx][:ls_frame]
# the pre-stuff
preFrms = filtPnts4[pt_idx].stream(confs = confs).widen(pre_frames).decap().frames[:pre_frames].rescale(1.5)
# joint
prexFrms = BGRFrames.from_frames([preFrms, exFrms])

prexArr_dimmed = np.stack([f.dim((idx / total_frames) ** 2).array for idx, f in enumerate(prexFrms)], axis = 0)
prexFrms_dimmed = BGRFrames(prexArr_dimmed).rescale(2)

all_frames.append(prexFrms_dimmed)
script_lines.append(script_line)

script_line =   """
                    In this simple example, we can for example show the bounce points of all returns made,
                    after rectifying them to be all on one side of the court. 
                    Note that two bounce points on the other side of the court stem from mishit returns that were practically aces.
                """
lsFrm = prexFrms_dimmed[-1]

retShots_rect_out = retShots_rect.filter('is_fair', nextShot = None, negate = True)
retShots_rect_in = retShots_rect.filter('is_fair', nextShot = None)

markScn_in = vs.scatter_markmap(retShots_rect_in, mark_size = 5, one_color = const.BGR['leafy_green'])
markScn_out = vs.scatter_markmap(retShots_rect_out, mark_size = 5, one_color = const.BGR['intense_red'])

markFrms, _ = vs.animate_static(markScn_in.combine(markScn_out), static_cam = myCam.resize(3), num_frames = 25,
                                animate_over = lsFrm.dim(0.25))

markFrms_itpl = vs.interpolate_frames(lsFrm, markFrms[0], 100)

all_frames.append(markFrms_itpl)
script_lines.append(script_line)

script_line =   """
                    Heatmaps can also be created with ease.
                """

### two regions -> opponents court, and behind the own baseline... a bit clumsy, but hey
from ATMANpy.types.Lines import CourtLines, Region
courtLines = CourtLines.make().add_singles_netline()
oppReg = courtLines.get_half_region(side = 'Top').scale(1.55)
ownReg_behind = Region.from_lines([courtLines['Bottom Baseline'], courtLines['Bottom Baseline'].down(300)])

in_res = 2
bh_res = 2.5
opp_res = 1.5

heatScn_in = vs.kde_heatmap(retShots_rect_in, one_color = const.BGR['leafy_green'], zero_color = const.BGR['leafy_green'],
                            bandwidth = 60, resolution = in_res)
heatScn_out1 = vs.kde_heatmap(retShots_rect_out, one_color = const.BGR['intense_red'], zero_color = const.BGR['intense_red'],
                            bandwidth = 30, showReg = ownReg_behind, resolution = bh_res)
heatScn_out2 = vs.kde_heatmap(retShots_rect_out, one_color = const.BGR['intense_red'], zero_color = const.BGR['intense_red'],
                            bandwidth = 30, showReg = oppReg, resolution = opp_res)

heatFrms, _ = vs.animate_static(heatScn_in.combine(heatScn_out1).combine(heatScn_out2), num_frames = 25, static_cam = myCam.resize(3), animate_over = lsFrm.dim(0.25))
# heatFrms[-1].show()

heatFrms_itpl = vs.interpolate_frames(markFrms[-1], heatFrms[1], num_frames = 100)
# DecapStream(heatFrms_itpl, 0, 100).show()

all_frames.append(heatFrms_itpl)
script_lines.append(script_line)

show(all_frames)