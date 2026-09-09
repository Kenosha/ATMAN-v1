"""Run and audit stages of the original broadcast pipeline in an isolated data root."""

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback


STAGES = (
    'court_lines', 'court_roi', 'segmentation_with_lines', 'hits_classify_sound',
    'segmentation_with_sound', 'manual_player_names_and_handedness',
    'manual_fix_tiebreaks', 'court_background', 'court_masks', 'player_candidates',
    'player_tracking', 'ball_filtered_motion_mask', 'ball_contours',
    'ball_inference_svm', 'hits_classify_video', 'segmentation_with_sound_and_vision',
    'manual_remove_warm_ups', 'ball_inference_hit_candidates', 'ball_inference_particles',
    'ball_interpolation_plane', 'manual_serve_identify', 'score_detection_testing',
    'score_detection_cleanup', 'ball_interpolation_colinear', 'select_colinear_variant',
    'join_traj_and_other_info',
)

ASSETS = (
    'ball_svm/svm_brest/weights/svm_proba_dict_r75_p60_2023-08-15.pickle',
    'cnn_sound/weights/test_model_20240109_150057',
    'cnn_sound/params/test_model_20240109_150057.pkl',
    *('cnn_player/' + kind + '/' + name + ('.pkl' if kind == 'params' else '')
      for kind in ('weights', 'params') for name in (
          'top_foremotion_T16_S8_randcrop5_dim56_small_fullconv_model_foremotion_20231121_235426',
          'bot_foremotion_T16_S8_randcrop5_dim88_small_fullconv_model_foremotion_20231122_095800')),
    *('trajectories_stored/ShorterTrajectories_g60_v41_s41_h46/' + name
      for name in ('info.txt', 'xArr.npy', 'zArr.npy', 'rpmArr.npy')),
)


def prepare(args):
    """Create a new isolated run; never import historical match feature stores."""
    if not args.assets_root or not args.human_review:
        raise ValueError('--prepare-video requires --assets-root and --human-review.')
    if not os.path.isfile(args.prepare_video):
        raise ValueError('--prepare-video must be an existing local video.')
    if not math.isfinite(args.start) or args.start < 0:
        raise ValueError('--start must be finite and nonnegative.')
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError('--duration must be finite and positive.')
    if os.path.exists(args.run_dir) and (not os.path.isdir(args.run_dir) or os.listdir(args.run_dir)):
        raise ValueError('Preparation requires a new or empty --run-dir.')
    assets = os.path.abspath(args.assets_root)
    missing = [name for name in ASSETS if not os.path.isfile(os.path.join(assets, name))]
    if missing:
        raise FileNotFoundError('Missing model assets: ' + ', '.join(missing))
    review = read_json(args.human_review)
    frame_count = round(args.duration * 25)
    if frame_count < 1:
        raise ValueError('The requested duration contains no frames at 25 fps.')
    if review['start_frame'] != 0 or review['stop_frame'] < frame_count - 1:
        raise ValueError('Human review must use clip-relative frame numbers and cover this excerpt.')
    if review.get('fps') != 25:
        raise ValueError('Human review must refer to the normalized 25 fps clip.')
    source_probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json',
        os.path.abspath(args.prepare_video)], text=True))
    source_duration = float(source_probe['format']['duration'])
    if args.start + frame_count / 25 > source_duration + 0.04:
        raise ValueError('The requested excerpt extends past the video.')
    if not any(stream['codec_type'] == 'audio' for stream in source_probe['streams']):
        raise ValueError('The full pipeline requires the broadcast audio track.')
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data = os.path.join(args.run_dir, 'data')
    video_id = 'local_excerpt'
    for folder in ('logs', 'stages', 'data/video_files/' + video_id,
                   'data/features_stored/' + video_id + '/fresh',
                   'data/graphs/pipeline_checks/pipeline_brest_tightseg/fresh'):
        os.makedirs(os.path.join(args.run_dir, folder), exist_ok=True)
    for folder in ('ball_svm', 'cnn_sound', 'cnn_player', 'trajectories_stored'):
        os.symlink(os.path.join(assets, folder), os.path.join(data, folder), target_is_directory=True)
    if args.ocr_runtime:
        os.symlink(os.path.abspath(args.ocr_runtime), os.path.join(args.run_dir, 'ocr_runtime'),
                   target_is_directory=True)
    shutil.copyfile(args.human_review, os.path.join(args.run_dir, 'human_review.json'))
    video_path = os.path.join(data, 'video_files', video_id, 'match.mp4')
    subprocess.run([
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '1',
        '-filter_threads', '1', '-ss', str(args.start), '-i', os.path.abspath(args.prepare_video),
        '-t', str(frame_count / 25), '-map', '0:v:0', '-map', '0:a:0',
        '-vf', 'scale=640:360,fps=25', '-frames:v', str(frame_count),
        '-c:v', 'libx264', '-threads', '1', '-preset', 'ultrafast', '-crf', '0',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', video_path], check=True)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format',
        '-of', 'json', video_path], text=True))
    video = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    if int(video['nb_read_frames']) != frame_count:
        raise ValueError('Normalized excerpt has an unexpected decoded frame count.')
    write_json(os.path.join(args.run_dir, 'input_probe.json'), probe)
    conf = dict(video_id=video_id, source_video_id=review['video_id'], store_name='fresh',
                source_video=os.path.abspath(args.prepare_video), video_path=video_path,
                source_start_seconds=args.start, duration_seconds=frame_count / 25,
                start_frame=0, stop_frame=frame_count-1, num_workers=1, chunksize=250,
                add_scale=1, width=640, heigth=360, lossless=True, compression_ball=32,
                compression_players=48, natural_scale=1/3, scale_factor=1/3,
                court_sample_frames=64, device='cpu', torch_threads=1,
                model_batch_rallies=1, model_batch_size=4, trajectory_workers=1,
                trajectory_ram_gb=1, trajectory_mmap=True, visualize=False,
                save_checks=False, seed=20260909, data_root=data, repo_root=root,
                run_dir=args.run_dir)
    write_json(os.path.join(args.run_dir, 'config.json'), conf)
    print(f'Prepared {frame_count} fresh input frames ({frame_count / 25:g} seconds).', flush=True)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(value, handle, indent=2, default=lambda obj: obj.tolist())
        handle.write('\n')
    os.replace(temporary, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def files_in_store(conf):
    folder = os.path.join(conf['data_root'], 'features_stored', conf['video_id'], conf['store_name'])
    result = {}
    for feature in sorted(os.listdir(folder)):
        path = os.path.join(folder, feature)
        if os.path.isdir(path):
            files = [os.path.join(base, name) for base, _, names in os.walk(path) for name in names]
            result[feature] = {'files': len(files), 'bytes': sum(os.path.getsize(name) for name in files)}
    return result


def worker(args, conf):
    import cv2
    import numpy as np
    import ATMANpy

    cv2.setNumThreads(1)
    np.random.seed(conf['seed'])
    random.seed(conf['seed'])
    root = os.path.realpath(conf['repo_root'])
    assert os.path.realpath(ATMANpy.__file__).startswith(root + os.sep)
    stage_folder = os.path.join(root, 'ATMANpy', 'scripts', 'pipeline_brest_tightseg')
    manual_stages = {'manual_player_names_and_handedness': 'apply_player_identity',
                     'manual_fix_tiebreaks': 'apply_no_tiebreak',
                     'manual_remove_warm_ups': 'apply_warmup_policy',
                     'manual_serve_identify': 'apply_serve_annotations',
                     'score_detection_cleanup': 'apply_score_cleanup',
                     'select_colinear_variant': 'select_colinear_variant'}
    module_path = (os.path.join(root, 'ATMANpy', 'cli', 'pipeline_annotations.py')
                   if args.stage in manual_stages else os.path.join(stage_folder, args.stage + '.py'))
    module = load_module(module_path, '_atman_pipeline_' + args.stage)
    params_module = load_module(os.path.join(stage_folder, 'parameter_files', 'default.py'),
                                '_atman_pipeline_parameters')
    parameter_stage = 'ball_interpolation_plane' if args.stage == 'ball_interpolation_colinear' else args.stage
    params = getattr(params_module, parameter_stage, None)
    kwargs = {} if params is None else copy.deepcopy(params.get_parameter_dictionary())
    if args.stage == 'ball_interpolation_colinear':
        kwargs.pop('bounce_params', None)
    if args.stage not in manual_stages and 'name' in inspect.signature(module.main).parameters:
        kwargs['name'] = 'pipeline_brest_tightseg'
    # Large legacy diagnostic grids are optional; final checks render separate artifacts.
    if not conf.get('save_checks', True) and hasattr(module, 'check'):
        module.check = lambda *values, **options: None
    result_path = os.path.join(args.run_dir, 'stages', args.stage + '.json')
    report = {'stage': args.stage, 'status': 'running', 'package_root': root,
              'stage_source_sha256': hashlib.sha256(open(module_path, 'rb').read()).hexdigest(),
              'configuration': dict(conf),
              'parameters': kwargs, 'input_features': files_in_store(conf)}
    write_json(result_path, report)
    cap = cv2.VideoCapture(conf['video_path'])
    if not cap.isOpened():
        raise ValueError('Cannot open the registered local clip.')
    conf['cap'] = cap
    started = time.monotonic()
    try:
        if args.stage in manual_stages:
            function = getattr(module, manual_stages[args.stage])
            if args.stage.startswith('manual_'):
                annotations = os.path.join(args.run_dir, 'human_review.json')
                report['human_review_sha256'] = hashlib.sha256(open(annotations, 'rb').read()).hexdigest()
                report['annotation_result'] = function(conf, annotations)
            else:
                report['annotation_result'] = function(conf)
        else:
            if args.stage == 'score_detection_testing':
                from ATMANpy.cli.pipeline_annotations import configure_ocr
                report['ocr_runtime'] = configure_ocr(os.path.join(args.run_dir, 'ocr_runtime'))
                conf['ocr_configs'] = report['ocr_runtime']['configs']
            module.main(conf, **kwargs)
        report['status'] = 'complete'
        report['output_features'] = files_in_store(conf)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        cap.release()
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write_json(result_path, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--stage', choices=STAGES)
    mode.add_argument('--all', action='store_true', help='Run every stage, stopping at the first failure.')
    parser.add_argument('--resume', action='store_true', help='With --all, skip previously completed stages.')
    parser.add_argument('--prepare-video', help='Prepare an excerpt from a local video into a new run directory.')
    parser.add_argument('--assets-root', help='Data root containing the trusted model weights and trajectory library.')
    parser.add_argument('--human-review', help='JSON of reviewed human labels in clip-relative frame numbers.')
    parser.add_argument('--ocr-runtime', help='Optional directory containing the portable Tesseract root/ bundle.')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float, default=240)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.run_dir = os.path.abspath(args.run_dir)
    if args.resume and not args.all:
        parser.error('--resume requires --all.')
    if args.prepare_video:
        prepare(args)
    if not args.stage and not args.all:
        if args.prepare_video:
            return 0
        parser.error('Choose --stage, --all, or --prepare-video.')
    if args.all:
        conf = read_json(os.path.join(args.run_dir, 'config.json'))
        for stage in STAGES:
            report = os.path.join(args.run_dir, 'stages', stage + '.json')
            if os.path.isfile(report) and read_json(report).get('status') == 'complete':
                if args.resume:
                    completed = read_json(report)
                    current = files_in_store(conf)
                    for feature, inventory in completed.get('output_features', {}).items():
                        if current.get(feature) != inventory:
                            raise ValueError('Saved outputs changed or are missing after ' + stage
                                             + ': ' + feature + '. Rerun affected stages explicitly.')
                    previous_conf = completed.get('configuration')
                    if previous_conf is not None and previous_conf != conf:
                        raise ValueError('Configuration changed after ' + stage + '; use a new run directory.')
                    print('Already complete: ' + stage, flush=True)
                    continue
                parser.error('This run already has completed stages; use --resume or a new run directory.')
            status = main(['--run-dir', args.run_dir, '--stage', stage])
            if status:
                return status
        return 0
    conf = read_json(os.path.join(args.run_dir, 'config.json'))
    environment = os.environ.copy()
    environment.update({'ATMAN_HOME': conf['data_root'], 'PYTHONPATH': conf['repo_root'],
                        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'OPENBLAS_NUM_THREADS': '1',
                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                        'MPLBACKEND': 'Agg', 'ATMAN_FFMPEG_THREADS': '1',
                        'MPLCONFIGDIR': os.path.join(args.run_dir, '.cache', 'matplotlib'),
                        'NUMBA_CACHE_DIR': os.path.join(args.run_dir, '.cache', 'numba'),
                        'XDG_CACHE_HOME': os.path.join(args.run_dir, '.cache')})
    os.environ.update(environment)
    if args.worker:
        worker(args, conf)
        return 0
    import psutil

    command = [sys.executable, '-B', os.path.abspath(__file__), '--run-dir', args.run_dir,
               '--stage', args.stage, '--worker']
    started = time.monotonic()
    peak = 0
    heartbeat = started
    log_path = os.path.join(args.run_dir, 'logs', args.stage + '.log')
    if os.path.exists(log_path):
        attempt = 1
        while os.path.exists(os.path.join(args.run_dir, 'attempts', args.stage, str(attempt))):
            attempt += 1
        previous = os.path.join(args.run_dir, 'attempts', args.stage, str(attempt))
        os.makedirs(previous)
        for path in (log_path, os.path.join(args.run_dir, 'stages', args.stage + '.json'),
                     os.path.join(args.run_dir, 'stages', args.stage + '_resources.json')):
            if os.path.isfile(path):
                shutil.copyfile(path, os.path.join(previous, os.path.basename(path)))
    print(f'Running {args.stage} on frames 0–{conf["stop_frame"]}, one worker.', flush=True)
    with open(log_path, 'w') as log:
        process = subprocess.Popen(command, cwd=conf['repo_root'], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [monitored, *monitored.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
                peak = max(peak, rss)
                if time.monotonic() - heartbeat >= 30:
                    print(f'{args.stage}: {time.monotonic()-started:.0f}s, '
                          f'{rss/1024**3:.2f} GiB resident across {len(processes)} processes.', flush=True)
                    heartbeat = time.monotonic()
            except psutil.Error:
                pass
            time.sleep(0.5)
        status = process.returncode
    record = {'stage': args.stage, 'exit_code': status, 'peak_rss_bytes': peak,
              'elapsed_seconds': round(time.monotonic()-started, 3), 'log': log_path}
    write_json(os.path.join(args.run_dir, 'stages', args.stage + '_resources.json'), record)
    print(json.dumps(record), flush=True)
    if status:
        with open(log_path) as log:
            print(log.read()[-7000:])
    return status


if __name__ == '__main__':
    raise SystemExit(main())
