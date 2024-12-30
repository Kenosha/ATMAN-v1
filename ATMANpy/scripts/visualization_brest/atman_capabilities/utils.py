import pandas as pd
import numpy as np

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.MultiFrames import BGRFrames

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

