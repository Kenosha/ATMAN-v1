
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
