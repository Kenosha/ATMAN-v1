"""Headless replay of human review decisions for the first four Brest minutes.

Detection/segmentation/trajectory inputs always come from conf's fresh store.
The supplied JSON contains human labels; its serve timestamps are matching
references, never emitted as fresh detections.
"""
import json
import os
import re
import sys
import shutil


def _annotations(human_json):
    if isinstance(human_json, dict):
        return human_json
    with open(human_json) as handle:
        return json.load(handle)


def _validate(conf, annotation):
    lo, hi = int(conf['start_frame']), int(conf['stop_frame'])
    if lo < annotation['start_frame'] or hi > annotation['stop_frame']:
        raise ValueError('Human review does not cover the requested frame interval.')
    source_id = conf.get('source_video_id', conf['video_id'])
    if source_id != annotation['video_id']:
        raise ValueError('Human review belongs to a different source video.')
    return lo, hi


def _load_feature(conf, name):
    from ATMANpy.types.Features import StreamFeature
    return StreamFeature.from_store(conf['video_id'], conf['store_name'], featname=name)


def _save_feature(conf, name, values):
    from ATMANpy.types.Features import StreamFeature, BoolFeature
    feature_class = BoolFeature if name.startswith('is') else StreamFeature
    feature = feature_class(values, name=name)
    feature.save(conf['video_id'], conf['store_name'])
    return feature


def _empty_feature_directory(conf, name):
    import ATMANpy.funcs.util_funcs as ut
    path = os.path.join(ut.mainDir, 'features_stored', conf['video_id'], conf['store_name'], name)
    os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        raise ValueError('Expected an empty no-correction feature directory: ' + name)


def apply_player_identity(conf, human_json):
    """Annotate fresh court frames using the reviewed identities/handedness."""
    annotation = _annotations(human_json)
    lo, hi = _validate(conf, annotation)
    court = _load_feature(conf, 'isCourt')
    sound_rallies = _load_feature(conf, 'isRally_sound')
    names = ['top_name_Flag', 'bot_name_Flag', 'top_hand_Flag', 'bot_hand_Flag']
    values = {name: {} for name in names}
    required_frames = {frame for frame, value in sound_rallies.items()
                       if value and lo <= frame <= hi}
    uncovered = []
    for frame, is_court in court.items():
        if not is_court or not lo <= frame <= hi:
            continue
        matches = [item for item in annotation['player_identity_intervals']
                   if item['start_frame'] <= frame <= item['stop_frame']]
        if not matches:
            if frame in required_frames:
                uncovered.append(frame)
            continue
        if len(matches) != 1:
            raise ValueError('Ambiguous player identity annotation at frame ' + str(frame))
        row = matches[0]
        for name in names:
            values[name][frame] = row[name]
    if uncovered:
        raise ValueError('Fresh play extends beyond reviewed player identity intervals: '
                         + str([min(uncovered), max(uncovered)]))
    for name in names:
        _save_feature(conf, name, values[name])
    return {'stage': 'player_identity', 'origin': 'existing human identity review',
            'fresh_court_frames_annotated': len(values[names[0]])}


def apply_no_tiebreak(conf, human_json):
    """Persist the reviewed absence of any tiebreak identity correction."""
    annotation = _annotations(human_json)
    _validate(conf, annotation)
    if annotation['tiebreak_corrections']:
        raise ValueError('This adapter requires explicit no-tiebreak review.')
    for side in ('top', 'bot'):
        for kind in ('name', 'hand'):
            _empty_feature_directory(conf, side + '_' + kind + '_tb_Flag')
    return {'stage': 'tiebreak_review', 'origin': 'existing human review', 'corrections': 0}


def apply_warmup_policy(conf, human_json):
    """Apply the reviewed no-warmup decision to newly inferred rally boundaries."""
    annotation = _annotations(human_json)
    lo, hi = _validate(conf, annotation)
    if annotation['warmup_policy'] != 'none_in_interval':
        raise ValueError('Unsupported warmup review policy.')
    rallies = _load_feature(conf, 'isRally')
    out = {frame: value for frame, value in rallies.items() if lo <= frame <= hi}
    _save_feature(conf, 'isRally_warmup_removed', out)
    return {'stage': 'warmup_review', 'origin': 'existing human review',
            'new_rally_frame_count': len(out), 'removed_frames': 0}


def apply_serve_annotations(conf, human_json, max_offset_frames=75):
    """Map reviewed serve categories one-to-one onto freshly inferred serve frames."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import ATMANpy.funcs.feature_funcs as ff
    annotation = _annotations(human_json)
    lo, hi = _validate(conf, annotation)
    trajectories = ff.read_feature_frame('trajTable', conf['video_id'], conf['store_name'], keep_only='last')
    rallies = _load_feature(conf, 'isRally_warmup_removed')
    active = {frame for frame, value in rallies.items() if value and lo <= frame <= hi}
    new_frames = sorted({int(frame) for frame, event in trajectories['event'].items()
                         if event in ('SBA', 'SBD', 'STA', 'STD') and int(frame) in active})
    labels = [item for item in annotation['serve_reviews'] if lo <= item['reference_frame'] <= hi]
    if not new_frames:
        raise ValueError('Fresh trajectories contain no serve detections to annotate.')
    costs = np.abs(np.asarray(new_frames)[:, None] - np.asarray([x['reference_frame'] for x in labels])[None, :])
    rows, columns = linear_sum_assignment(costs)
    matched = [(int(i), int(j)) for i, j in zip(rows, columns) if costs[i, j] <= max_offset_frames]
    matched_rows = {i for i, _ in matched}
    if len(matched_rows) != len(new_frames):
        unknown = [frame for i, frame in enumerate(new_frames) if i not in matched_rows]
        raise ValueError('Fresh serve detections need additional human classification: ' + str(unknown))
    out = {new_frames[i]: labels[j]['classification'] for i, j in matched}
    _save_feature(conf, 'srvFlag', out)
    return {'stage': 'serve_review', 'origin': 'existing human categories remapped to new detections',
            'matched': [{'detected_frame': new_frames[i], 'review_reference_frame': labels[j]['reference_frame'],
                         'offset_frames': new_frames[i] - labels[j]['reference_frame'],
                         'classification': labels[j]['classification']} for i, j in matched],
            'undetected_review_references': [item['reference_frame'] for j, item in enumerate(labels)
                                             if j not in {x[1] for x in matched}]}


def _normal_game_score(value):
    import pandas as pd
    if value is None or pd.isna(value) or str(value).strip() == '':
        return '0'
    if isinstance(value, (int, float)) and int(value) == value:
        return str(int(value))
    value = str(value).strip()
    if not re.fullmatch(r'[0-7](?:-[0-7])*', value):
        raise ValueError('Unrecognized OCR game score: ' + repr(value))
    return value


def apply_score_cleanup(conf):
    """Run the automatic historical game-score cleanup rules; never copy old OCR."""
    from ATMANpy.types.Features import StreamFeatures
    games = StreamFeatures.from_store(conf['video_id'], conf['store_name'],
                                     featnameList=['sb_games_upper_Flag', 'sb_games_lower_Flag'])
    rallies = _load_feature(conf, 'isRally_warmup_removed')
    segments = games.conform(rallies).filter(rallies).split()
    previous = ['0', '0']
    values = {'sb_games_upper_cleaned_Flag': {}, 'sb_games_lower_cleaned_Flag': {}}
    missing_to_zero = 0
    for segment in segments:
        current = []
        for side in ('upper', 'lower'):
            raw_values = segment['sb_games_' + side + '_Flag'].unique()
            if len(raw_values) != 1:
                raise ValueError('Multiple OCR scores in a freshly inferred rally at ' + str(segment.get_frames()[0]))
            current.append(_normal_game_score(raw_values[0]))
            if raw_values[0] is None or str(raw_values[0]).strip().lower() in ('', 'nan'):
                missing_to_zero += 1
        prev = [[int(x) for x in text.split('-')] for text in previous]
        curr = [[int(x) for x in text.split('-')] for text in current]
        valid = len(curr[0]) == len(curr[1])
        if valid and len(curr[0]) == len(prev[0]):
            valid = all(c[:-1] == p[:-1] and c[-1] in (p[-1], p[-1] + 1) for c, p in zip(curr, prev))
        elif valid and len(curr[0]) == len(prev[0]) + 1:
            valid = all(c[:-1] == p and c[-1] == 0 for c, p in zip(curr, prev))
        else:
            valid = False
        if not valid:
            raise ValueError('OCR game progression needs human review at frame '
                             + str(segment.get_frames()[0]) + ': ' + repr(previous) + ' -> ' + repr(current))
        for side, score in zip(('upper', 'lower'), current):
            values['sb_games_' + side + '_cleaned_Flag'].update({frame: score for frame in segment.get_frames()})
        previous = current
    for name, data in values.items():
        _save_feature(conf, name, data)
    return {'stage': 'score_cleanup', 'origin': 'fresh OCR plus automatic progression checks',
            'rallies': segments.length(), 'missing_scores_normalized_to_zero': missing_to_zero}


def select_colinear_variant(conf, variant='fair'):
    """Expose one fresh hypothesis under the name expected by the historical join.

The first four minutes of the original combined table equal its fair hypothesis
within 1e-6 for all 1,452 stored rows. The two hypotheses are alternatives;
concatenating them would duplicate frames. Keep both original output tables.
"""
    import ATMANpy.funcs.feature_funcs as ff
    if variant not in ('fair', 'fault'):
        raise ValueError('variant must be fair or fault')
    table = ff.read_feature_frame('colinearTable_' + variant, conf['video_id'], conf['store_name'], keep_only='last')
    if table.index.has_duplicates:
        raise ValueError('Colinear hypothesis has duplicate source frames.')
    ff.save_feature_frame(table, 'colinearTable', conf['video_id'], conf['store_name'])
    return {'stage': 'colinear_selection', 'source': 'newly computed colinearTable_' + variant,
            'rows': len(table), 'policy': 'explicit alternative hypothesis selection'}


def configure_ocr(runtime_root):
    """Configure the portable runtime, or an installed Tesseract if absent.

runtime_root is the directory containing root/ and python/ in the prepared
bundle. For installed Tesseract, local whitelist configs are written under
runtime_root/configs. Pass the returned configs as conf['ocr_configs'].
"""
    python_support = os.path.join(runtime_root, 'python')
    if os.path.isdir(python_support) and python_support not in sys.path:
        sys.path.insert(0, python_support)
    import pytesseract
    base = os.path.join(runtime_root, 'root')
    binary = os.path.join(base, 'usr', 'bin', 'tesseract')
    library = os.path.join(base, 'usr', 'lib', 'x86_64-linux-gnu')
    data = os.path.join(base, 'usr', 'share', 'tesseract-ocr', '5', 'tessdata')
    if os.path.isdir(base):
        if not os.path.isfile(binary) or not os.path.isfile(os.path.join(data, 'eng.traineddata')):
            raise FileNotFoundError('Incomplete portable Tesseract runtime: ' + runtime_root)
        mode = 'portable'
        config_dir = os.path.join(data, 'configs')
        os.environ['LD_LIBRARY_PATH'] = library + (os.pathsep + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
        os.environ['TESSDATA_PREFIX'] = data
    else:
        binary = shutil.which('tesseract')
        if binary is None:
            raise FileNotFoundError('Tesseract is not installed and no portable runtime was provided: ' + runtime_root)
        mode = 'system'
        config_dir = os.path.join(runtime_root, 'configs')
        os.makedirs(config_dir, exist_ok=True)
        whitelists = {'letters_tennis': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-.',
                      'digits_tennis': '0123456789AD', 'games_tennis': '01234567'}
        for name, characters in whitelists.items():
            config_path = os.path.join(config_dir, name)
            if not os.path.isfile(config_path):
                with open(config_path, 'w') as handle:
                    handle.write('tessedit_char_whitelist ' + characters + '\n')
    pytesseract.pytesseract.tesseract_cmd = binary
    os.environ['OMP_THREAD_LIMIT'] = '1'
    configs = {name: os.path.join(config_dir, name)
               for name in ('letters_tennis', 'digits_tennis', 'games_tennis')}
    return {'binary': binary, 'version': str(pytesseract.get_tesseract_version()).splitlines()[0],
            'mode': mode, 'configs': configs}
