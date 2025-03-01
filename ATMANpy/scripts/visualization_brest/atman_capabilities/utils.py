import pandas as pd
import numpy as np

import ATMANpy.auxil.constants as const

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Overlays import Table
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Shapes3D import Point3D
from ATMANpy.types.Visual3D import Scene

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)


###### USEFUL MAPPING FUNCTIONS #####
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

def get_data_string(myShot):
    velo = myShot.get_initial_speed()
    spin = myShot.get_initial_spin()
    angl = myShot.get_launch_angle()

    veloStr = f'{int(round(velo, 0))} km/h'
    spinStr = f'{int(round(spin, 0))} RPM'
    anglStr = f'{round(angl, 1)} deg.'

    return f'Speed: {veloStr} \n\nSpin: {spinStr} \n\nLaunch Angle: {anglStr}'

#######

def speed_up_frames_progressively(segFrms, ramp_up_frames = 50, min_frames_for_speedup = 100):
    if segFrms.length() <= ramp_up_frames * 2:
        return segFrms
    else:
        segFrms_start = segFrms[:ramp_up_frames]
        segFrms_middle = segFrms[ramp_up_frames:-ramp_up_frames]
        segFrms_end = segFrms[-ramp_up_frames:]

        if segFrms_middle.length() <= min_frames_for_speedup:
            return segFrms
        elif min_frames_for_speedup < segFrms_middle.length() <= min_frames_for_speedup + 2 * ramp_up_frames:
            segFrms_middle = segFrms_middle[::2]
            return segFrms_start.join(segFrms_middle).join(segFrms_end)
        else: ### in this case, two levels of speedup
            segFrms_middle_start = segFrms_middle[:ramp_up_frames:2]
            segFrms_middle_middle = segFrms_middle[ramp_up_frames:-ramp_up_frames:3]
            segFrms_middle_end = segFrms_middle[-ramp_up_frames::2]
            return segFrms_start.join(segFrms_middle_start).join(segFrms_middle_middle).join(segFrms_middle_end).join(segFrms_end)

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

def has_nan_point_or_is_none(myLn):
    if myLn is None:
        return True
    else:
        return np.isnan(myLn.leftPt.coords).all() or np.isnan(myLn.rightPt.coords).all()


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

def make_matrix_table(matArr, title = None, dimensions = (600, 400), horizontal_offset = 4.5):
    rs, cs = matArr.shape
    table_data = {i: {j: matArr[i, j] for j in range(cs)} for i in range(rs)}

    rename_null_dict = {0: '', 1: '', 2: '', 3: ''}

    myTab = Table(table_data, columns_rename_dict = rename_null_dict, index_rename_dict = rename_null_dict,
                  horizontal_lines = False, header_row_height = 0.5, index_col_width = 0, skeleton_shrinkage = 0.25,
                  number_format = '>10.2f', dimensions = dimensions,
                  title = title, title_horizontal_offset = horizontal_offset, title_vertical_offset = 0.5)
    return myTab


def render_over_frames(myTab, myFrms, tlPt = Point2D((0, 0))):
    from ATMANpy.types.Overlays import overlay
    testFrm = myTab.render()
    rs, cs = testFrm.rows, testFrm.cols
    cx, cy = tlPt.coords

    rendered_frames = []
    for frm in myFrms:
        overArr = frm.array[cy:cy+rs, cx:cx+cs]
        overFrm = myTab.render(over_array = overArr)
        rendered_frames.append(overFrm)
    
    renFrms = BGRFrames.from_single_frames(rendered_frames)
    outFrms = overlay(myFrms, renFrms, top_left_point = tlPt)
    return outFrms

def render_many_over_frames(myTabs, myFrms, tlPt = Point2D((0, 0))):
    from ATMANpy.types.Overlays import overlay

    rendered_frames = []
    for tab, frm in zip(myTabs, myFrms):
        testFrm = tab.render()
        rs, cs = testFrm.rows, testFrm.cols
        cx, cy = tlPt.coords

        overArr = frm.array[cy:cy+rs, cx:cx+cs]
        overFrm = tab.render(over_array = overArr)
        rendered_frames.append(overFrm)
    
    renFrms = BGRFrames.from_single_frames(rendered_frames)
    outFrms = overlay(myFrms, renFrms, top_left_point = tlPt)
    return outFrms

def get_shot_scenes(myPnt, ball_color = const.BGR['ball_yellow'], 
                    mark_size = 2, mark_color = const.BGR['bright_blue']):
    from ATMANpy.types.ShapesTennis3D import Mark

    bounce_scenes = []
    spline_scenes = []
    for myShot in myPnt:
        # splines
        shot_splines = myShot.feat_spline(which_endpoint = 'penultimate').values()
        # n = len(shot_splines)
        n = myShot.get_time_until_hit()

        if n != len(shot_splines):
            print(f'Probably a lob, but should be able to handle!')
            print(f'Frames until next hit: {n}')
            print(f'Shot splines created: {len(shot_splines)}')

        shotScns = [Scene([spln.recolor(ball_color).resize(1.5)]) for spln in shot_splines if spln is not None]
        ### pad the initial None
        shotScns = [Scene.make_empty(), *shotScns] 
        ### pad the final frames, if splines are nan
        n_spln = len(shotScns)
        shotScns = [*shotScns, *[Scene.make_empty() for _ in range(n - n_spln)]]
        ### join to previous
        spline_scenes = [*spline_scenes, *shotScns]

        # bounce points
        all_frms = myShot.get_frames()
        bnc_frms = myShot.get_bounce_frames()
        bnc_x, bnc_y = myShot.get_first_bounce_position()
        
        try:
            prevScn = bounce_scenes[-1]
        except IndexError:
            prevScn = Scene.make_empty()

        if len(bnc_frms) > 0:
            bncPt = Point3D((bnc_x, bnc_y, 0))
            bncMark = Mark.make(bncPt, size = mark_size, color = mark_color)
            bnc_idx = all_frms.index(bnc_frms[0])
        else:
            bnc_idx = len(all_frms) ### this way, we'll never actually need the bncMark

        for i in range(n):
            if i >= bnc_idx:
                bounce_scenes.append(prevScn.put(bncMark))
            else:
                bounce_scenes.append(prevScn)
        
    return spline_scenes, bounce_scenes
    