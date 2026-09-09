"""Validate and illustrate a completed, isolated ATMAN broadcast pipeline run.

Read-only with respect to input features: no inference, label editing, or saved
feature writes. Rendered media use ATMAN's original Graph/Points/Scene/View and
vis utilities. This checks execution and internal consistency, not accuracy
against independent ground truth or equality with a historical prediction run.
"""

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
import sys
import time
import traceback


STAGE_OUTPUTS = {
    'court_lines': ('H', 'P', 'M_int', 'M_ext'),
    'court_roi': ('courtroiReg', 'topserveReg', 'botserveReg'),
    'segmentation_with_lines': ('isCourt',),
    'hits_classify_sound': ('sound_hitTable',),
    'segmentation_with_sound': ('isRally_sound',),
    'manual_player_names_and_handedness': ('top_name_Flag', 'bot_name_Flag', 'top_hand_Flag', 'bot_hand_Flag'),
    'manual_fix_tiebreaks': ('top_name_tb_Flag', 'bot_name_tb_Flag', 'top_hand_tb_Flag', 'bot_hand_tb_Flag'),
    'court_background': ('median_backFrm', 'std_backFrm'),
    'court_masks': ('foregroundMask', 'shadowMask', 'fine_motMask'),
    'player_candidates': ('foreCnts',),
    'player_tracking': ('topPlayerCnt', 'botPlayerCnt'),
    'ball_filtered_motion_mask': ('fine_motMask_filt',),
    'ball_contours': ('movCnts',),
    'ball_inference_svm': ('cand0Cnts', 'cand1Cnts', 'cand2Cnts'),
    'hits_classify_video': ('top_hitTable', 'bot_hitTable'),
    'segmentation_with_sound_and_vision': ('hitFlag', 'isRally'),
    'manual_remove_warm_ups': ('isRally_warmup_removed',),
    'ball_inference_hit_candidates': ('hitCnt',),
    'ball_inference_particles': ('ball_choicePt',),
    'ball_interpolation_plane': ('trajTable',),
    'manual_serve_identify': ('srvFlag',),
    'score_detection_testing': ('sb_name_upper_Flag', 'sb_name_lower_Flag', 'sb_score_upper_Flag', 'sb_score_lower_Flag', 'sb_games_upper_Flag', 'sb_games_lower_Flag'),
    'score_detection_cleanup': ('sb_games_upper_cleaned_Flag', 'sb_games_lower_cleaned_Flag'),
    'ball_interpolation_colinear': ('colinearTable_fair', 'colinearTable_fault'),
    'select_colinear_variant': ('colinearTable',),
    'join_traj_and_other_info': ('dataTable', 'dataTable_colinear', 'placeTable'),
}
TABLES = ('sound_hitTable', 'top_hitTable', 'bot_hitTable', 'trajTable',
          'colinearTable_fair', 'colinearTable_fault', 'colinearTable',
          'dataTable', 'dataTable_colinear', 'placeTable')
TEMPORAL_LABELS = {'oth', 'srv_lf', 'srv_rt'} | {
    '_'.join((stroke, hand, spin, height)) for stroke in ('fh', 'bh')
    for hand in ('lf', 'rt') for spin in ('sf', 'ts') for height in ('gs', 'ov', 'vl')}


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def clean_json(value):
    """Strict JSON: missing/nonfinite numeric observations become null."""
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if hasattr(value, 'tolist'):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def within(path, parent):
    return os.path.commonpath([os.path.realpath(path), os.path.realpath(parent)]) == os.path.realpath(parent)


def add_check(report, name, condition, detail=None):
    report['checks'].append({'name': name, 'passed': bool(condition), 'detail': clean_json(detail)})


def inventory(folder):
    records = {}
    if not os.path.isdir(folder):
        return records
    for base, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(base, name)
            if not within(path, folder):
                raise ValueError('Feature file resolves outside the isolated store: ' + path)
            stat = os.stat(path)
            records[os.path.relpath(path, folder)] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return records


def check_stages(run_dir, conf, store, report):
    stages = []
    human_hash = sha256(os.path.join(run_dir, 'human_review.json'))
    for name, expected in STAGE_OUTPUTS.items():
        stage_path = os.path.join(run_dir, 'stages', name + '.json')
        resource_path = os.path.join(run_dir, 'stages', name + '_resources.json')
        stage = read_json(stage_path) if os.path.isfile(stage_path) else {'status': 'missing'}
        resources = read_json(resource_path) if os.path.isfile(resource_path) else {}
        empty_allowed = name == 'manual_fix_tiebreaks'
        absent = [feature for feature in expected
                  if not os.path.isdir(os.path.join(store, feature))
                  or (not empty_allowed and not inventory(os.path.join(store, feature)))]
        recorded = stage.get('output_features', {})
        unrecorded = [feature for feature in expected if feature not in recorded]
        complete = stage.get('status') == 'complete' and resources.get('exit_code') == 0
        add_check(report, 'stage:' + name, complete and not absent and not unrecorded,
                  {'status': stage.get('status'), 'exit_code': resources.get('exit_code'),
                   'missing_artifacts': absent, 'unrecorded_outputs': unrecorded})
        if stage.get('package_root'):
            add_check(report, 'source:' + name,
                      os.path.realpath(stage['package_root']) == os.path.realpath(conf['repo_root']))
        if name.startswith('manual_') and stage.get('status') == 'complete':
            add_check(report, 'human_review:' + name, stage.get('human_review_sha256') == human_hash)
        stages.append({'name': name, 'status': stage.get('status'),
                       'seconds': resources.get('elapsed_seconds', stage.get('elapsed_seconds')),
                       'peak_rss_mib': resources.get('peak_rss_bytes', 0) / 1024 ** 2,
                       'source_sha256': stage.get('stage_source_sha256'),
                       'annotation_result': stage.get('annotation_result'),
                       'log': os.path.relpath(os.path.join(run_dir, 'logs', name + '.log'), report['output'])})
    first = os.path.join(run_dir, 'stages', 'court_lines.json')
    add_check(report, 'isolated_store_started_empty',
              os.path.isfile(first) and read_json(first).get('input_features') == {})
    report['stages'] = stages
    report['resources'] = {
        'completed_stages': sum(stage['status'] == 'complete' for stage in stages),
        'required_stages': len(stages),
        'recorded_stage_seconds': sum(stage['seconds'] or 0 for stage in stages),
        'peak_stage_rss_mib': max(stage['peak_rss_mib'] for stage in stages),
        'note': 'Stage wall time excludes preparation, failed attempts and this report; RSS includes child processes.'}
    attempts = []
    for base, _, files in os.walk(os.path.join(run_dir, 'attempts')):
        for filename in sorted(files):
            if not filename.endswith('_resources.json'):
                continue
            path = os.path.join(base, filename)
            prior = read_json(path)
            prior['record'] = os.path.relpath(path, report['output'])
            log = os.path.join(base, filename.replace('_resources.json', '.log'))
            prior['log'] = os.path.relpath(log, report['output'])
            attempts.append(prior)
    report['previous_attempts'] = attempts
    report['resources']['previous_attempt_seconds'] = sum(attempt.get('elapsed_seconds', 0) for attempt in attempts)
    if attempts:
        report['warnings'].append(f'{len(attempts)} earlier stage attempt(s) are archived with their logs and resource records.')
    observed_warnings = set()
    for stage in stages:
        path = os.path.join(run_dir, 'logs', stage['name'] + '.log')
        if not os.path.isfile(path):
            continue
        with open(path, errors='replace') as handle:
            for line in handle:
                if 'InconsistentVersionWarning' in line:
                    observed_warnings.add('The SVM checkpoint was saved with an older scikit-learn version; the loader emitted a version compatibility warning.')
                if 'mel filterbank has all zero values' in line:
                    observed_warnings.add('The original sound model parameters produced a zero-valued mel filterband warning; original preprocessing was preserved.')
    report['warnings'].extend(sorted(observed_warnings))


def read_tables(conf, store, report):
    import numpy as np
    import pandas as pd
    import ATMANpy.funcs.feature_funcs as ff

    tables = {}
    for name in TABLES:
        try:
            table = ff.read_feature_frame(name, conf['video_id'], conf['store_name'], keep_only='last')
        except pd.errors.EmptyDataError:
            if name != 'placeTable':
                raise
            table = pd.DataFrame()
            report['warnings'].append('placeTable contains no rows: there are no saved placeholder points.')
        tables[name] = table
        files = inventory(os.path.join(store, name))
        for relative, record in files.items():
            record['sha256'] = sha256(os.path.join(store, name, relative))
        info = {'rows': len(table), 'columns': list(table.columns), 'files': files}
        # placeTable may be empty: it records rallies without reconstructed shots.
        add_check(report, 'table:' + name, name == 'placeTable' or len(table) > 0, info)
        if len(table) and name != 'placeTable':
            index = np.asarray(table.index, dtype=float)
            add_check(report, 'frame_bounds:' + name,
                      np.isfinite(index).all() and (index >= conf['start_frame']).all()
                      and (index <= conf['stop_frame']).all(),
                      {'first': float(index.min()), 'last': float(index.max())})
    report['table_rows'] = {name: len(table) for name, table in tables.items()}
    return tables


def probabilities(table, name, report, side=None):
    import numpy as np
    values = table.to_numpy(dtype=float)
    expected = {'hit', 'bounce'} if side is None else {side + '_' + label for label in TEMPORAL_LABELS}
    finite = np.isfinite(values)
    correct_labels = set(table.columns) == expected
    sums = values.sum(axis=1) if len(values) else np.asarray([])
    bounded = bool(len(values)) and finite.all() and (values >= -1e-6).all() and (values <= 1 + 1e-6).all()
    normalized = side is None or np.allclose(sums, 1, atol=1e-5)
    add_check(report, 'probabilities:' + name, correct_labels and bounded and normalized,
              {'rows': len(table), 'classes': len(table.columns), 'expected_classes': len(expected),
               'all_finite': finite.all(), 'range_min': values.min() if values.size else None,
               'range_max': values.max() if values.size else None,
               'max_sum_error': np.abs(sums - 1).max() if side and len(sums) else None})
    return {'rows': len(table), 'classes': list(table.columns),
            'most_likely_frame_classes': dict(Counter(table.idxmax(axis=1))) if len(table) else {},
            'note': 'Temporal rows include the original between-frame interpolation; argmax counts are frames, not strokes.'
                    if side else 'Sound hit and bounce are independent sigmoid scores; their sum need not be one.'}


def presence_coverage(feature, active_feature, reference_name):
    """Use active frame IDs as the denominator: storage may omit None rows."""
    stored = dict(feature.items())
    present = {int(frame) for frame, value in stored.items() if value is not None}
    active = {int(frame) for frame, value in active_feature.items() if value}
    present_active = present & active
    return {'reference_feature': reference_name, 'active_frames': len(active),
            'stored_frames': len(stored), 'present_stored_frames': len(present),
            'stored_null_frames': len(stored) - len(present),
            'present_active_frames': len(present_active),
            'missing_active_frames': len(active - present),
            'present_outside_active_frames': len(present - active),
            'active_coverage_fraction': len(present_active) / len(active) if active else None}


def feature_summary(conf, report):
    import numpy as np
    from ATMANpy.types.Features import StreamFeature

    features = {}
    report['segmentation'] = {}
    report['ball'] = {}
    for name in ('isCourt', 'isRally_sound', 'isRally', 'isRally_warmup_removed',
                 'cand0Cnts', 'cand1Cnts', 'cand2Cnts', 'ball_choicePt',
                 'topPlayerCnt', 'botPlayerCnt', 'hitFlag', 'srvFlag'):
        feature = StreamFeature.from_store(conf['video_id'], conf['store_name'], featname=name)
        features[name] = feature
        if name.startswith('is'):
            active = sorted(int(frame) for frame, value in feature.items() if value)
            segments = sum(idx == 0 or frame != active[idx - 1] + 1 for idx, frame in enumerate(active))
            report['segmentation'][name] = {'active_frames': len(active), 'segments': segments}
        elif name.startswith('cand'):
            counts = [value.length() if value is not None else 0 for value in feature.values()]
            report['ball'][name] = {'stored_frames': len(counts), 'frames_with_candidates': sum(count > 0 for count in counts),
                                    'total_contours': sum(counts)}
        elif name == 'ball_choicePt':
            coords = [value.coords for value in feature.values() if value is not None]
            array = np.asarray(coords, dtype=float).reshape(-1, 2)
            add_check(report, 'finite_particle_ball_detections', len(array) > 0 and np.isfinite(array).all())
            report['ball']['particle_selection'] = {
                **presence_coverage(feature, features['isRally'], 'isRally'),
                'selected_frames': len(coords),
                'note': 'ball_choicePt is the saved particle-filter/Kitagawa selection. Coverage uses freshly inferred '
                        'active rally frames, including frames omitted from storage. Particle clouds are not persisted.'}
        elif name.endswith('PlayerCnt'):
            report.setdefault('players', {})[name] = {
                **presence_coverage(feature, features['isRally_sound'], 'isRally_sound'),
                'tracked_frames': sum(value is not None for value in feature.values()),
                'note': 'Coverage uses the fresh audio-play frames supplied to player tracking.'}
    hit_events = [{'frame': int(frame), 'class': str(value)}
                  for frame, value in features['hitFlag'].thin().items()]
    report['classified_events'] = {'events': len(hit_events),
                                    'classes': dict(Counter(row['class'] for row in hit_events))}
    add_check(report, 'fresh_classified_hit_events', len(hit_events) > 0)
    add_check(report, 'svm_candidate_outputs', report['ball']['cand2Cnts']['total_contours'] > 0)
    return features, hit_events


def shot_observation_counts(table, detections, rallies):
    """Count real selected-ball detections, never inferred x_o/y_o/z_o rows.

    make_trajectory_frame numbers shots separately in each rally and omits the
    final next-hit row (iloc[:-1]). Restore that endpoint for input coverage.
    Joined tables add point_idx; raw trajectory tables require a rally ID from
    the fresh isRally segments to avoid merging identically numbered shots.
    """
    import numpy as np

    active = sorted(int(frame) for frame, value in rallies.items() if value)
    rally_by_frame = {}
    rally_idx = -1
    for idx, frame in enumerate(active):
        if idx == 0 or frame != active[idx - 1] + 1:
            rally_idx += 1
        rally_by_frame[frame] = rally_idx
    selected_frames = set()
    for frame, value in detections.items():
        if value is not None:
            xy = np.asarray(value.coords, dtype=float)
            if xy.shape == (2,) and np.isfinite(xy).all():
                selected_frames.add(int(frame))
    frame = table.copy()
    frame['_audit_rally_idx'] = [rally_by_frame.get(int(number)) for number in frame.index]
    if frame['_audit_rally_idx'].isna().any():
        raise ValueError('Cannot group trajectory rows outside fresh isRally segments.')
    group_columns = ['point_idx', 'shot_idx'] if 'point_idx' in frame.columns else ['_audit_rally_idx', 'shot_idx']
    rows = []
    for _, shot in frame.groupby(group_columns, sort=True):
        shot_rallies = set(int(value) for value in shot['_audit_rally_idx'])
        if len(shot_rallies) != 1:
            raise ValueError('One joined point/shot spans more than one fresh rally.')
        rally_idx = shot_rallies.pop()
        saved_frames = {int(number) for number in shot.index}
        first, last = min(saved_frames), max(saved_frames)
        stop = last + 1
        observed = {number for number in selected_frames if first <= number <= stop
                    and rally_by_frame.get(number) == rally_idx}
        rows.append({'rally_idx': rally_idx,
                     'point_idx': int(shot['point_idx'].iloc[0]) if 'point_idx' in shot else None,
                     'shot_idx': int(shot['shot_idx'].iloc[0]),
                     'event_descriptor': str(shot['event_descriptor'].iloc[0]) if 'event_descriptor' in shot else None,
                     'start_frame': first, 'last_saved_frame': last, 'fit_stop_frame': stop,
                     'saved_rows': len(shot), 'selected_ball_detections': len(observed),
                     'selected_ball_detections_on_saved_rows': len(selected_frames & saved_frames),
                     'endpoint_detection': stop in observed,
                     'fewer_than_3_detections': len(observed) < 3})
    return rows


def projection_equivalence(saved, camera, tolerance=1e-8):
    """Camera projection matrices represent the same map up to nonzero scale."""
    import numpy as np

    left, right = np.asarray(saved, dtype=float), np.asarray(camera, dtype=float)
    if left.shape != (3, 4) or right.shape != (3, 4) or not np.isfinite(left).all() or not np.isfinite(right).all():
        return False, {'reason': 'Projection matrices must be finite 3-by-4 arrays.'}
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return False, {'reason': 'A zero matrix is not a camera projection.'}
    left_unit, right_unit = left / left_norm, right / right_norm
    if np.linalg.matrix_rank(left_unit) != 3 or np.linalg.matrix_rank(right_unit) != 3:
        return False, {'reason': 'Projection matrices must have rank three.'}
    alignment = float(np.vdot(right_unit, left_unit))
    error = float(np.linalg.norm(left_unit - alignment * right_unit))
    return error <= tolerance, {'scale_to_saved_P': alignment * left_norm / right_norm,
                                'relative_frobenius_error_after_scale': error,
                                'tolerance': tolerance}


def trajectory_checks(tables, conf, detections, report, rallies):
    import numpy as np
    from ATMANpy.types.Features import ConstFeature
    from ATMANpy.types.Visual3D import Camera

    matrices = [ConstFeature.from_store(conf['video_id'], conf['store_name'], featname=name).feature
                for name in ('M_int', 'M_ext')]
    camera = Camera(*matrices, (conf['width'], conf.get('height', conf.get('heigth', 360))))
    add_check(report, 'finite_calibrated_camera', all(np.isfinite(matrix).all() for matrix in matrices))
    projection = ConstFeature.from_store(conf['video_id'], conf['store_name'], featname='P').feature
    equivalent, comparison = projection_equivalence(projection, camera.get_P())
    add_check(report, 'saved_projection_matches_camera', equivalent, comparison)
    report['trajectories'] = {}
    report['shot_observations'] = []
    residual_rows = []
    for name in ('trajTable', 'colinearTable_fair', 'colinearTable_fault', 'dataTable', 'dataTable_colinear'):
        table = tables[name]
        coordinates = table[['x_t', 'y_t', 'z_t']].to_numpy(dtype=float)
        valid = np.isfinite(coordinates).all(axis=1)
        add_check(report, 'finite_3d:' + name, valid.sum() >= 2 and not np.isinf(coordinates).any(),
                  {'finite_rows': int(valid.sum()), 'missing_rows': int((~valid).sum()),
                   'note': 'Missing trajectory samples are counted explicitly, not filled.'})
        frames = np.asarray(table.index)[valid]
        xyz = coordinates[valid]
        camera_xyz = camera.world_points_to_camera_points(xyz)
        in_front = camera_xyz[:, 2] < 0  # ATMAN View uses negative camera-space z for visible points.
        pixels = camera.project_points(xyz)
        finite_projection = np.isfinite(pixels).all(axis=1)
        pairs = []
        for frame, xy, front, finite in zip(frames, pixels, in_front, finite_projection):
            detection = detections.maybe_get(int(frame))
            if detection is None or not front or not finite:
                continue
            observed = np.asarray(detection.coords, dtype=float)
            if not np.isfinite(observed).all():
                continue
            residual = float(np.linalg.norm(xy - observed))
            pairs.append(residual)
            if name in ('dataTable', 'dataTable_colinear'):
                residual_rows.append({'table': name, 'frame': int(frame), 'projected_x': float(xy[0]),
                                      'projected_y': float(xy[1]), 'selected_ball_x': float(observed[0]),
                                      'selected_ball_y': float(observed[1]), 'residual_px': residual})
        report['trajectories'][name] = {
            'rows': len(table), 'finite_3d_rows': int(valid.sum()),
            'missing_3d_rows': int((~valid).sum()), 'behind_camera_rows': int((~in_front).sum()),
            'nonfinite_projections': int((~finite_projection).sum()),
            'projection_pairs': len(pairs), 'median_residual_px': float(np.median(pairs)) if pairs else None,
            'p90_residual_px': float(np.percentile(pairs, 90)) if pairs else None}
        support = [{'table': name, **row} for row in shot_observation_counts(table, detections, rallies)]
        report['shot_observations'].extend(support)
        report['trajectories'][name]['shots_with_fewer_than_3_detections'] = sum(row['fewer_than_3_detections'] for row in support)
        report['trajectories'][name]['shots_audited'] = len(support)
        add_check(report, 'projectable_3d:' + name,
                  finite_projection.all() and in_front.all() and len(pairs) > 0)
    report['shot_observation_note'] = (
        'Counts intersect finite ball_choicePt detections with each fitted shot interval in the fresh rally. '
        'The dropped next-hit endpoint is restored as last_saved_frame + 1. '
        'Stored x_o/y_o/z_o columns are not treated as detection evidence. '
        'Counts describe available input observations; plane/net constraints may exclude some from a fit. '
        'Low-support fits are retained, and this metric does not establish accuracy.')
    low_support = {name: info['shots_with_fewer_than_3_detections'] for name, info in report['trajectories'].items()
                   if info['shots_with_fewer_than_3_detections']}
    if low_support:
        report['warnings'].append('Fits supported by fewer than 3 actual selected-ball detections are retained: '
                                  + ', '.join(f'{name}: {count}' for name, count in low_support.items())
                                  + '. These are limited observations, not an accuracy assessment; see shot_observations.csv.')
    report['projection_note'] = ('Reprojection residual compares fitted 3D trajectories with this run\'s own '
                                 'particle-selected ball positions. It measures internal fit consistency; '
                                 'these observations also informed the fit. It is not independent accuracy.')
    joined = tables['dataTable']
    required = {'point_idx', 'shot_idx', 'event', 'event_descriptor', 'hit_name', 'rec_name'}
    add_check(report, 'joined_shot_and_player_fields', required.issubset(joined.columns))
    if required.issubset(joined.columns):
        shot_finite = []
        for _, shot in joined.groupby(['point_idx', 'shot_idx']):
            shot_finite.append(int(np.isfinite(shot[['x_t', 'y_t', 'z_t']].to_numpy(dtype=float)).all(axis=1).sum()))
        add_check(report, 'each_joined_shot_has_finite_trajectory',
                  len(shot_finite) > 0 and min(shot_finite) >= 2,
                  {'shots': len(shot_finite), 'minimum_finite_rows': min(shot_finite) if shot_finite else 0})
    return camera, residual_rows


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(points, camera, features, conf, output, max_frames, report):
    import cv2
    import numpy as np
    from ATMANpy.cli.demo import save_image, write_video
    from ATMANpy.types.Overlays import Graph, ScatterChart, LineChart
    from ATMANpy.types.Visual3D import Camera, Scene, View
    import ATMANpy.auxil.constants as const
    import ATMANpy.funcs.vis_funcs as vis

    yellow, blue = const.BGR['ball_yellow'], const.BGR['bright_blue']
    shots = points.flatten()
    colors = (yellow, blue)
    charts = []
    player_names = sorted(set(shots.get_hit_player_name()))
    for name, color in zip(player_names, colors):
        selected = shots.filter(lambda shot: shot.get_hit_player_name() == name)
        x, y = selected.get_initial_speed(), selected.get_initial_spin()
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            charts.append(ScatterChart(x[finite], y[finite], color=color, thickness=0.08))
    graph = Graph(charts, dimensions=(960, 540), resolution=20,
                  color=const.BGR['bright_grey'], header_text='Freshly reconstructed shots',
                  x_label='Estimated speed (km/h)', y_label='Estimated spin (RPM)',
                  label_size=1.0, tick_label_size=0.9)
    frame = graph.render()
    for idx, (name, color) in enumerate(zip(player_names, colors)):
        frame = frame.put(name, where=(620, 68 + 23 * idx), color=color, size=0.55)
    save_image(os.path.join(output, 'speed_spin.png'), frame)

    virtual_camera = Camera.calibrated_brest_behind()
    court = Scene.make_court(singles_net=False, net_mesh_density=3, net_alpha=0.35)
    court_view = View.make(virtual_camera, court, resolution=0.8)
    serves = shots.filter('is_serve').enforce_direction('Top')
    placement = vis.scatter_markmap(serves, one_color=yellow, mark_size=3)
    frame = court_view.combine(View.make(virtual_camera, placement, resolution=0.8)).snap()
    frame = frame.put('Fresh serve bounce estimates | normalized direction', size=0.5)
    save_image(os.path.join(output, 'serve_placement.png'), frame)

    timeline = []
    for name, color in (('cand2Cnts', blue),):
        values = features[name]
        xs = np.asarray(list(values.keys()), dtype=float) / 25
        ys = np.asarray([item.length() if item is not None else 0 for item in values.values()])
        if len(xs):
            timeline.append(LineChart(xs, ys, color=color))
    frame = Graph(timeline, dimensions=(960, 540), resolution=16,
                  color=const.BGR['bright_grey'], header_text='SVM high-confidence ball candidates',
                  x_label='Seconds in fresh excerpt', y_label='Candidate contours',
                  label_size=1.0, tick_label_size=0.9).render()
    save_image(os.path.join(output, 'ball_candidates.png'), frame)

    # The first usable joined point is deterministic; do not select by fit quality.
    point = next((point for point in points if len(point) >= 2), next(iter(points)))
    scenes, frame_numbers, descriptions = [], [], []
    for shot in point:
        splines = shot.feat_spline_cumul(which_endpoint='penultimate', thickness=4, color=yellow)
        for number, lines in zip(splines.get_frames(), splines.values()):
            valid = [line for line in lines if line is not None
                     and np.isfinite(line.leftPt.coords).all() and np.isfinite(line.rightPt.coords).all()]
            scenes.append(Scene(valid))
            frame_numbers.append(int(number))
            descriptions.append(str(shot.get_hit_event_descriptor()))
            if len(scenes) >= max_frames:
                break
        if len(scenes) >= max_frames:
            break
    if not scenes or not any(len(scene) for scene in scenes):
        raise ValueError('The first reconstructed point has no finite drawable trajectory.')
    animation, _ = vis.animate_sequence(scenes, start_cam=virtual_camera, stop_cam=virtual_camera,
                                        background_scene=court, background_resolution=0.8, argument_resolution=0.8)
    count = write_video(os.path.join(output, 'trajectory.mp4'), animation)
    preview = max(range(len(scenes)), key=lambda index: len(scenes[index]))
    save_image(os.path.join(output, 'trajectory.png'), animation[preview])
    capture = cv2.VideoCapture(conf['video_path'])
    if not capture.isOpened():
        raise ValueError('Cannot open the registered fresh input video.')
    if not np.isclose(capture.get(cv2.CAP_PROP_FPS), 25, atol=0.05):
        capture.release()
        raise ValueError('The fresh pipeline report requires its normalized 25 fps input.')
    visible_detections = []
    def overlays():
        try:
            for number, scene, descriptor in zip(frame_numbers, scenes, descriptions):
                capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                ok, array = capture.read()
                if not ok:
                    raise ValueError('Cannot decode fresh input frame ' + str(number))
                if (array.shape[1], array.shape[0]) != camera.sensor_dims:
                    raise ValueError('Video dimensions do not match the fresh calibration.')
                frame = View.make(camera, scene, resolution=0.8).snap(over_array=array.copy())
                detection = features['ball_choicePt'].maybe_get(number)
                if detection is not None:
                    frame = frame.put(detection, color=blue, radius=5, thickness=2)
                    visible_detections.append(number)
                frame = frame.put('Fresh fit (yellow), selected ball (blue)', where=(10, 20), size=0.5)
                frame = frame.put(f'{number / 25:.2f}s | {descriptor}', where=(10, 343), size=0.5)
                if number == frame_numbers[preview]:
                    save_image(os.path.join(output, 'broadcast_overlay.png'), frame)
                yield frame
        finally:
            capture.release()
    overlay_count = write_video(os.path.join(output, 'broadcast_overlay.mp4'), overlays())
    report['rendering'] = {'selection': 'First joined point with at least two shots, otherwise the first joined point.',
                            'point_index': point.get_point_index(), 'frames': count, 'overlay_frames': overlay_count,
                            'first_source_frame': frame_numbers[0], 'last_source_frame': frame_numbers[-1],
                            'source_frames': frame_numbers, 'fps': 25,
                            'frames_with_particle_detection': len(visible_detections),
                            'note': 'Silent diagnostic clips, rendered through the original visualization utilities.'}


def verify_media(output, conf, report):
    """Decode every frame and PNG; successful encoding alone is insufficient."""
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    expected = len(report['rendering']['source_frames'])
    if not 1 <= expected <= 250:
        raise ValueError('Media verification is limited to 1–250 rendered source frames.')
    records = {}
    for name in ('trajectory.mp4', 'broadcast_overlay.mp4'):
        path = os.path.join(output, name)
        parameters = [cv2.CAP_PROP_N_THREADS, 1] if hasattr(cv2, 'CAP_PROP_N_THREADS') else []
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, parameters)
        info = {'expected_frames': expected, 'decoded_frames': 0, 'nonblank_frames': 0,
                'changing_frame_pairs': 0, 'unique_decoded_frames': 0, 'decoded_to_eof': False,
                'max_mean_abs_frame_change': 0.0, 'decoder_threads': 1 if parameters else None}
        digest = hashlib.sha256()
        hashes, shapes = set(), set()
        previous = None
        try:
            if not capture.isOpened():
                raise ValueError('Video decoder could not open the encoded clip.')
            info['fps'] = float(capture.get(cv2.CAP_PROP_FPS))
            info['metadata_frame_count'] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            while info['decoded_frames'] <= expected:
                ok, frame = capture.read()
                if not ok:
                    info['decoded_to_eof'] = True
                    break
                info['decoded_frames'] += 1
                shapes.add(tuple(frame.shape))
                pixels = np.ascontiguousarray(frame).tobytes()
                digest.update(pixels)
                hashes.add(hashlib.sha256(pixels).hexdigest())
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                info['nonblank_frames'] += int(float(grey.std()) > 1.0)
                if previous is not None:
                    difference = cv2.absdiff(frame, previous)
                    changed_pixels = np.any(difference > 2, axis=2).sum()
                    # Require several pixels to change beyond tiny codec noise.
                    threshold = max(4, int(frame.shape[0] * frame.shape[1] * 0.00001))
                    info['changing_frame_pairs'] += int(changed_pixels >= threshold)
                    info['max_mean_abs_frame_change'] = max(info['max_mean_abs_frame_change'], float(difference.mean()))
                previous = frame
            info['unique_decoded_frames'] = len(hashes)
            info['decoded_shapes'] = sorted(shapes)
            info['decoded_pixels_sha256'] = digest.hexdigest()
        except Exception as error:
            info['error'] = f'{type(error).__name__}: {error}'
        finally:
            capture.release()
        shape_ok = len(shapes) == 1 and next(iter(shapes))[2:] == (3,)
        if name == 'broadcast_overlay.mp4':
            shape_ok = shapes == {(conf.get('height', conf.get('heigth', 360)), conf['width'], 3)}
        valid = ('error' not in info and info['decoded_to_eof'] and info['decoded_frames'] == expected
                 and info.get('metadata_frame_count') == expected and shape_ok
                 and math.isclose(info.get('fps', 0), 25, abs_tol=1e-6)
                 and info['nonblank_frames'] == expected and info['unique_decoded_frames'] >= 2
                 and info['changing_frame_pairs'] > 0)
        add_check(report, 'decoded_media:' + name, valid, info)
        records[name] = info
    for name in ('speed_spin.png', 'serve_placement.png', 'ball_candidates.png',
                 'trajectory.png', 'broadcast_overlay.png'):
        frame = cv2.imread(os.path.join(output, name), cv2.IMREAD_UNCHANGED)
        valid = frame is not None and frame.size > 0 and frame.ndim == 3 and frame.shape[2] in (3, 4)
        info = {'decoded': valid}
        if valid:
            info.update({'shape': frame.shape, 'dtype': str(frame.dtype),
                         'decoded_pixels_sha256': hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()})
        add_check(report, 'decoded_media:' + name, valid, info)
        records[name] = info
    report['media_verification'] = records


def write_report(output, report):
    report = clean_json(report)
    with open(os.path.join(output, 'report.json'), 'w') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    esc = lambda value: html.escape(str(value))
    sections = ['<!doctype html><html lang="en"><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                '<title>ATMAN fresh pipeline validation</title>',
                '<style>body{font:16px/1.5 system-ui,sans-serif;color:#202623;background:#f8faf8;max-width:1060px;margin:40px auto;padding:0 20px}'
                'h1,h2{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}'
                'th,td{border-bottom:1px solid #cdd7cd;padding:7px;text-align:left}img,video{max-width:100%;height:auto}'
                'figure{margin:24px 0}figcaption{color:#526052}code{font-size:13px}a{color:#1b6951}pre{white-space:pre-wrap}'
                '.status{font-size:21px;font-weight:650}details{margin-top:24px}</style>',
                '<h1>ATMAN: fresh broadcast pipeline run</h1>',
                '<p class="status">' + esc(report['status'].upper()) + '</p>',
                '<p>' + esc(report['purpose']) + '</p>']
    summary = report.get('summary', {})
    if summary:
        sections.append('<p>' + esc(f"{summary['points']} reconstructed points · {summary['shots']} shots · "
                                      f"{summary['serves']} serves · {summary['feature_rows']} joined rows") + '</p>')
    sections += ['<h2>Where the results come from</h2>',
                 '<p><strong>Automated stages:</strong> court calibration and segmentation, sound scores, '
                 'player tracking, SVM candidates, temporal stroke scores, combined event inference, particle ball selection, '
                 '3D reconstruction, and scoreboard OCR use the registered video excerpt. '
                 'The execution table below records which stages completed.</p>',
                 '<p><strong>Replayed human review:</strong> player identity and handedness, the no-tiebreak and no-warmup decisions, '
                 'and first/second/let serve categories. Serve review timestamps are matched to newly detected serves when that stage completes. '
                 'This run is therefore assisted by human labels.</p>',
                 '<p>Both colinear hypotheses are retained; the explicit selected hypothesis is recorded in the stage table. '
                 'No historical detector or trajectory predictions are read by this report.</p>']
    for warning in report.get('warnings', []):
        sections.append('<p><strong>Limit:</strong> ' + esc(warning) + '</p>')
    coverages = {'Particle-selected ball': report.get('ball', {}).get('particle_selection', {}),
                 **report.get('players', {})}
    if any(coverage for coverage in coverages.values()):
        sections += ['<h2>Detection coverage</h2>',
                     '<p>Coverage counts active frame IDs, including frames with no stored detection. '
                     'Presence does not establish that the selected object is correct.</p>',
                     '<table><tr><th>Feature</th><th>Active-frame reference</th><th>Present / active</th><th>Missing active</th></tr>']
        for name, coverage in coverages.items():
            if coverage:
                sections.append('<tr>' + ''.join('<td>' + esc(value) + '</td>' for value in
                                (name, coverage['reference_feature'],
                                 f"{coverage['present_active_frames']} / {coverage['active_frames']}",
                                 coverage['missing_active_frames'])) + '</tr>')
        sections.append('</table>')
    for filename, caption in (
            ('broadcast_overlay.mp4', 'Fresh 3D projection in yellow; particle-selected ball observations in blue.'),
            ('trajectory.mp4', 'The same freshly reconstructed point from a virtual camera.'),
            ('speed_spin.png', 'Estimated speed and spin. These are model outputs, without independent ground-truth validation.'),
            ('serve_placement.png', 'Fresh serve bounce estimates, normalized to one direction.'),
            ('ball_candidates.png', 'The high-confidence SVM candidate count across the excerpt.')):
        if os.path.isfile(os.path.join(output, filename)):
            media = ('<video controls preload="metadata" src="' + filename + '"></video>' if filename.endswith('.mp4')
                     else '<img loading="lazy" src="' + filename + '" alt="' + esc(caption) + '">')
            sections.append('<figure>' + media + '<figcaption>' + esc(caption) + '</figcaption></figure>')
    sections += ['<h2>3D projection consistency</h2><p>' + esc(report.get('projection_note', 'Not available.')) + '</p>',
                 '<table><tr><th>Table</th><th>Finite / all rows</th><th>Compared frames</th><th>Median / p90 (px)</th><th>Shots with &lt;3 detections</th></tr>']
    for name, value in report.get('trajectories', {}).items():
        median, p90 = value['median_residual_px'], value['p90_residual_px']
        error = 'n/a' if median is None else f'{median:.2f} / {p90:.2f}'
        sections.append('<tr>' + ''.join('<td>' + esc(cell) + '</td>' for cell in
                        (name, f"{value['finite_3d_rows']} / {value['rows']}", value['projection_pairs'], error,
                         value['shots_with_fewer_than_3_detections'])) + '</tr>')
    sections.append('</table><p>' + esc(report.get('shot_observation_note', '')) + '</p>')
    sections += ['<h2>Stage execution</h2>', '<table><tr><th>Stage</th><th>Status</th><th>Seconds</th><th>Peak MiB</th></tr>']
    for stage in report.get('stages', []):
        seconds = 'n/a' if stage['seconds'] is None else f"{stage['seconds']:.1f}"
        sections.append('<tr><td><a href="' + esc(stage['log']) + '">' + esc(stage['name']) + '</a></td>'
                        + ''.join('<td>' + esc(value) + '</td>' for value in
                                  (stage['status'], seconds, f"{stage['peak_rss_mib']:.1f}")) + '</tr>')
    sections += ['</table><h2>Checks</h2>']
    failures = [check for check in report['checks'] if not check['passed']]
    sections.append('<p>' + esc(f"{len(report['checks']) - len(failures)} / {len(report['checks'])} checks passed.") + '</p>')
    if failures:
        sections.append('<ul>' + ''.join('<li>' + esc(check['name']) + ': ' + esc(check['detail']) + '</li>' for check in failures) + '</ul>')
    if report.get('previous_attempts'):
        sections.append('<details><summary>Earlier stage attempts</summary><ul>')
        for attempt in report['previous_attempts']:
            sections.append('<li><a href="' + esc(attempt['log']) + '">' + esc(attempt.get('stage'))
                            + '</a> — exit ' + esc(attempt.get('exit_code')) + ', '
                            + esc(attempt.get('elapsed_seconds')) + ' seconds.</li>')
        sections.append('</ul></details>')
    sections.append('<p><a href="report.json">Full JSON evidence</a> · <a href="shots.csv">Joined shots</a> · '
                    '<a href="classified_events.csv">Inferred events</a> · <a href="projection_residuals.csv">Projection residuals</a> · '
                    '<a href="shot_observations.csv">Selected-ball detections per shot</a></p>')
    sections.append('<details><summary>Human review and applied annotations</summary><pre>'
                    + esc(json.dumps(report.get('human_review', {}), indent=2)) + '</pre></details>')
    sections.append('</html>')
    with open(os.path.join(output, 'index.html'), 'w') as handle:
        handle.write('\n'.join(sections) + '\n')
    with open(os.path.join(output, 'README.md'), 'w') as handle:
        handle.write('# Fresh pipeline validation: ' + report['status'].upper() + '\n\n'
                     + report['purpose'] + '\n\nOpen [index.html](index.html) for the report, plots and two short diagnostic clips. '
                     'Exact checks and provenance are in [report.json](report.json).\n\n'
                     'Human identity/handedness, serve categories and no-warmup/tiebreak reviews are replayed explicitly. '
                     'Detector predictions and 3D reconstructions come from the new isolated run. '
                     'Internal checks do not establish prediction accuracy.\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Completed isolated run containing config.json and stages/.')
    parser.add_argument('--output', help='New or empty output directory; defaults to RUN_DIR/report.')
    parser.add_argument('--max-frames', type=int, default=100, help='Frames per diagnostic clip (1 to 250).')
    parser.add_argument('--no-render', action='store_true', help='Check numeric outputs only.')
    args = parser.parse_args(argv)
    if not 1 <= args.max_frames <= 250:
        parser.error('--max-frames must be between 1 and 250.')
    run_dir = os.path.abspath(args.run_dir)
    output = os.path.abspath(args.output or os.path.join(run_dir, 'report'))
    if os.path.exists(output) and (not os.path.isdir(output) or os.listdir(output)):
        parser.error('--output must be a new or empty directory.')
    conf = read_json(os.path.join(run_dir, 'config.json'))
    data = os.path.realpath(conf['data_root'])
    store = os.path.join(data, 'features_stored', conf['video_id'], conf['store_name'])
    if not within(data, run_dir) or not within(store, data) or not within(conf['video_path'], data):
        parser.error('Config must identify an isolated data store and video inside this run.')
    if within(output, data) or within(run_dir, output):
        parser.error('Keep output outside the input data; it must not contain the run directory.')
    os.makedirs(output, exist_ok=True)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ATMAN_FFMPEG_THREADS'):
        os.environ[name] = '1'
    os.environ.update({'ATMAN_HOME': data, 'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1',
                       'MPLCONFIGDIR': os.path.join(output, '.cache', 'matplotlib'),
                       'NUMBA_CACHE_DIR': os.path.join(output, '.cache', 'numba'),
                       'XDG_CACHE_HOME': os.path.join(output, '.cache')})
    sys.dont_write_bytecode = True
    sys.path.insert(0, conf['repo_root'])
    started = time.monotonic()
    report = {'status': 'incomplete', 'checks': [], 'run_dir': run_dir, 'output': output,
              'purpose': f"Validate the historical ATMAN pipeline on {conf['duration_seconds']:g} seconds of fresh broadcast video.",
              'config': conf, 'human_review': read_json(os.path.join(run_dir, 'human_review.json')),
              'warnings': ['Model classifications and 3D speed/spin estimates are experimental; no independent ground truth is evaluated.']}
    before = inventory(store)
    try:
        check_stages(run_dir, conf, store, report)
        if any(not check['passed'] for check in report['checks']):
            report['warnings'].append('Required stage evidence is missing or inconsistent; no features or media were rendered.')
            return 2
        import cv2
        import ATMANpy
        import ATMANpy.funcs.util_funcs as ut
        from ATMANpy.cli.demo import load_points, summarize
        cv2.setNumThreads(1)
        if not within(ATMANpy.__file__, conf['repo_root']) or os.path.realpath(ut.mainDir) != data:
            raise RuntimeError('ATMAN source or feature root does not match the isolated run.')
        tables = read_tables(conf, store, report)
        report['classifiers'] = {
            'sound': probabilities(tables['sound_hitTable'], 'sound_hitTable', report),
            'top': probabilities(tables['top_hitTable'], 'top_hitTable', report, 'top'),
            'bot': probabilities(tables['bot_hitTable'], 'bot_hitTable', report, 'bot')}
        features, events = feature_summary(conf, report)
        camera, residuals = trajectory_checks(tables, conf, features['ball_choicePt'], report, features['isRally'])
        table, points = load_points(data, conf['video_id'], conf['store_name'])
        summary, rows = summarize(table, points)
        report['summary'] = summary
        report['joined_shot_classes'] = dict(Counter(str(shot.get_hit_event_descriptor()) for shot in points.flatten()))
        write_csv(os.path.join(output, 'shots.csv'), rows, list(rows[0]))
        write_csv(os.path.join(output, 'first_serves.csv'), [row for row in rows if row['is_first_serve']], list(rows[0]))
        write_csv(os.path.join(output, 'fast_shots.csv'), [row for row in rows if row['speed_kmh'] > 150], list(rows[0]))
        write_csv(os.path.join(output, 'classified_events.csv'), events, ['frame', 'class'])
        write_csv(os.path.join(output, 'projection_residuals.csv'), residuals,
                  ['table', 'frame', 'projected_x', 'projected_y', 'selected_ball_x', 'selected_ball_y', 'residual_px'])
        write_csv(os.path.join(output, 'shot_observations.csv'), report['shot_observations'],
                  ['table', 'rally_idx', 'point_idx', 'shot_idx', 'event_descriptor', 'start_frame',
                   'last_saved_frame', 'fit_stop_frame', 'saved_rows', 'selected_ball_detections',
                   'selected_ball_detections_on_saved_rows', 'endpoint_detection', 'fewer_than_3_detections'])
        if not args.no_render:
            render(points, camera, features, conf, output, args.max_frames, report)
            verify_media(output, conf, report)
        else:
            report['rendering'] = {'status': 'not_requested'}
        report['status'] = 'passed' if all(check['passed'] for check in report['checks']) else 'failed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        add_check(report, 'report_execution', False, report['error'])
    finally:
        origins = {name: os.path.realpath(module.__file__) for name, module in sys.modules.items()
                   if name == 'ATMANpy' or name.startswith('ATMANpy.') if getattr(module, '__file__', None)}
        report['source_imports'] = origins
        add_check(report, 'all_atman_imports_from_selected_repo', all(within(path, conf['repo_root']) for path in origins.values()))
        add_check(report, 'input_feature_files_unchanged', before == inventory(store),
                  {'checked_files': len(before), 'method': 'File paths, sizes and nanosecond mtimes before and after read-only reporting.'})
        if report['status'] == 'passed' and any(not check['passed'] for check in report['checks']):
            report['status'] = 'failed'
        report['report_seconds'] = round(time.monotonic() - started, 3)
        report['artifacts'] = {name: {'bytes': os.path.getsize(os.path.join(output, name)), 'sha256': sha256(os.path.join(output, name))}
                               for name in sorted(os.listdir(output)) if os.path.isfile(os.path.join(output, name))}
        write_report(output, report)
        print(json.dumps({'status': report['status'], 'report': os.path.join(output, 'index.html'),
                          'summary': report.get('summary'), 'error': report.get('error')}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
