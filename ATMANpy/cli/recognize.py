"""Attempt recognition on a short local broadcast-video excerpt.

This entry point reuses v1 court, foreground, motion and ball-SVM extractors.
It does not replace the historical, partly manual shot/3D reconstruction pipeline.
"""

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time


WIDTH, HEIGHT, FPS = 640, 360, 25
SOUND_MODEL = 'test_model_20240109_150057'
BALL_MODEL = 'svm_proba_dict_r75_p60_2023-08-15.pickle'


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('video', help='Existing local video file; URLs are not accepted.')
    command.add_argument('--start', type=float, default=0, help='Start time in seconds.')
    command.add_argument('--duration', type=float, default=8,
                         help='Excerpt duration in seconds, greater than 0 and at most 30 (default: 8).')
    command.add_argument('--output', help='New or empty output directory.')
    command.add_argument('--data-root', default=os.environ.get('ATMAN_HOME'),
                         help='Optional data root containing ball_svm/ and cnn_sound/.')
    command.add_argument('--ball-model', help='Explicit trusted v1 SVM pickle file.')
    command.add_argument('--sound-model-dir', help='Explicit cnn_sound directory with weights/ and params/.')
    command.add_argument('--no-audio', action='store_true', help='Skip optional audio inference.')
    command.add_argument('--threads', type=int, default=1, choices=range(1, 5),
                         help='CPU/ffmpeg thread limit; no worker processes are used (default: 1).')
    command.add_argument('--calibration-samples', type=int, default=12,
                         help='Number of sampled frames for court calibration, 1 to 40.')
    return command


def validate_args(args):
    if not os.path.isfile(args.video):
        raise ValueError('VIDEO must be an existing local file.')
    if not math.isfinite(args.start) or args.start < 0:
        raise ValueError('--start must be a finite, nonnegative number of seconds.')
    if not math.isfinite(args.duration) or not 0 < args.duration <= 30:
        raise ValueError('--duration must be greater than 0 and at most 30 seconds.')
    if not 1 <= args.calibration_samples <= 40:
        raise ValueError('--calibration-samples must be between 1 and 40.')
    if args.ball_model and not os.path.isfile(args.ball_model):
        raise ValueError('--ball-model does not exist.')
    if args.sound_model_dir:
        for subdir, filename in [('weights', SOUND_MODEL), ('params', SOUND_MODEL + '.pkl')]:
            if not os.path.isfile(os.path.join(args.sound_model_dir, subdir, filename)):
                raise ValueError('--sound-model-dir must contain the expected v1 weights and parameters.')
    if args.output and os.path.exists(args.output):
        if not os.path.isdir(args.output) or os.listdir(args.output):
            raise ValueError('--output must be a new or empty directory; existing outputs are preserved.')


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run_ffmpeg(arguments, threads=1, **kwargs):
    return subprocess.run(
        ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
         '-threads', str(threads), '-filter_threads', str(threads)] + arguments,
        check=True, stderr=subprocess.PIPE, **kwargs,
    )


def open_video(path, threads=1):
    import cv2 as cv

    if hasattr(cv, 'CAP_PROP_N_THREADS'):
        return cv.VideoCapture(path, cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, threads])
    return cv.VideoCapture(path)


def normalize_video(args, output):
    path = os.path.join(output, 'input_excerpt.mp4')
    run_ffmpeg([
        '-ss', str(args.start), '-i', os.path.abspath(args.video), '-t', str(args.duration),
        '-map', '0:v:0', '-vf', f'scale={WIDTH}:{HEIGHT},fps={FPS}', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-threads', str(args.threads), '-movflags', '+faststart', path,
    ], args.threads)
    return path


def load_models(args, report):
    """Load only locally provided artifacts; missing optional weights are explicit."""
    svm = None
    svm_path = args.ball_model
    if not svm_path and args.data_root:
        candidate = os.path.join(args.data_root, 'ball_svm', 'svm_brest', 'weights', BALL_MODEL)
        if os.path.isfile(candidate):
            svm_path = candidate
    if svm_path:
        import pickle
        import warnings

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with open(svm_path, 'rb') as handle:
                    svm = pickle.load(handle)
            report['stages']['ball_svm'] = {
                'status': 'loaded', 'weights': os.path.abspath(svm_path),
                'warnings': [str(warning.message) for warning in caught],
            }
        except Exception as error:
            report['stages']['ball_svm'] = {'status': 'failed', 'error': str(error)}
    else:
        report['stages']['ball_svm'] = {'status': 'skipped', 'reason': 'No local SVM weights supplied.'}
    return svm


def calibrate(cap, count, samples, report):
    import cv2 as cv
    import numpy as np
    import ATMANpy.funcs.extract_funcs as ex
    import ATMANpy.funcs.sf_apply_funcs as sa
    from ATMANpy.types.Frames import BGRFrame
    from ATMANpy.types.Points import Point2D

    # Parameters from pipeline_brest_tightseg/parameter_files/default.py at 640x360.
    params = dict(width_of_line=8 / 3, int_diff_thresh=10, white_thresh=150,
                  rho=3, theta=np.pi / 180, min_votes=int(500 / 9),
                  minLineLength=8 / 3 + 1, maxLineGap=1, merge_threshold=16 / 3,
                  horizontal_filter_margin=40 / 3, vertical_filter_margin=40 / 3,
                  only_225=False, verbose=False)
    sample_indices = np.unique(np.linspace(0, count - 1, min(samples, count), dtype=int))
    corners, errors, found_frames = [], [], []
    for index in sample_indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(index))
        ok, array = cap.read()
        if not ok:
            errors.append(f'Could not decode frame {index}.')
            continue
        try:
            court, H, _, _ = ex.get_courtlines(BGRFrame(array).to_grey(), **params)
            if H is not None and np.isfinite(H).all():
                corners.append([point.coords for point in sa.get_court_corner_points(court)])
                found_frames.append(int(index))
        except Exception as error:
            errors.append(f'Frame {index}: {type(error).__name__}: {error}')
    result = {'status': 'not_detected', 'samples': len(sample_indices),
              'detected_frames': found_frames, 'errors': errors}
    report['stages']['court_calibration'] = result
    if not corners:
        return None, None
    median = np.median(np.asarray(corners), axis=0).astype(int)
    H = sa.calculate_homography(*[Point2D(tuple(point)) for point in median])
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        result.update(status='failed', error='Court homography is degenerate.')
        return None, None
    result.update(status='ok', homography=H.tolist(), median_corners=median.tolist())
    return sa.find_courtlines(H), H


def background(cap, count):
    import cv2 as cv
    import numpy as np
    from ATMANpy.types.Frames import HSVFrame

    arrays = []
    for index in np.unique(np.linspace(0, count - 1, min(32, count), dtype=int)):
        cap.set(cv.CAP_PROP_POS_FRAMES, int(index))
        ok, array = cap.read()
        if ok:
            arrays.append(cv.cvtColor(array, cv.COLOR_BGR2HSV))
    return HSVFrame(np.median(np.stack(arrays), axis=0).astype('uint8'))


def ball_candidates(contours, hsv, grey, svm):
    """Use the original feature extractor, training normalization and SVM classifier."""
    import ATMANpy.funcs.extract_funcs as ex
    from ATMANpy.types.Contours import Contours

    empty = Contours([])
    if svm is None or contours is None or contours.is_empty():
        return empty, []
    table = ex.get_feature_frames(hsv, grey, contours, **svm['feature_params'])
    if table is None or table.empty:
        return empty, []
    normalized = (table - svm['mean']) / svm['std']
    features = normalized[svm['features']].values
    probabilities = svm['svm'].predict_proba(features)[:, 1]
    predicted = svm['svm'].predict(features)
    accepted, rows = [], []
    for contour_index, probability, is_ball in zip(table.index, probabilities, predicted):
        contour = contours[int(contour_index)]
        point = contour.center()
        rows.append({'x': point.x, 'y': point.y, 'probability': float(probability),
                     'svm_positive': bool(is_ball)})
        if is_ball:
            accepted.append(contour)
    return Contours(accepted), rows


def process_frames(path, output, count, court, back, svm, args, report):
    import cv2 as cv
    import numpy as np
    import ATMANpy.funcs.extract_funcs as ex
    from ATMANpy.types.Contours import Contours
    from ATMANpy.types.Frames import BGRFrame

    cap = open_video(path, args.threads)
    cache, next_frame, rows, ball_rows = {}, 0, [], []
    roi = court.get_court_region('Doubles').scale(1.8) if court is not None else None
    video_path = os.path.join(output, 'recognition.mp4')
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{WIDTH}x{HEIGHT}',
               '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-threads', str(args.threads), '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_path]
    errors, foreground_count, motion_count, balls_count = [], 0, 0, 0
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as writer:
        try:
            for index in range(count):
                while next_frame <= min(count - 1, index + 8):
                    ok, array = cap.read()
                    if not ok:
                        raise RuntimeError(f'Could not decode normalized frame {next_frame}.')
                    color = BGRFrame(array)
                    cache[next_frame] = (color, color.to_grey())
                    next_frame += 1
                color, grey = cache[index]
                foreground, motion, balls = Contours([]), Contours([]), Contours([])
                fp, fn, is_court = None, None, False
                if court is not None:
                    fp, fn = ex.court_match_rates(grey, court, width_of_line=8 / 3,
                                                white_thresh=150, int_diff_thresh=10)
                    is_court = bool(fn < 0.3)
                if is_court:
                    hsv = color.to_HSV()
                    burn = ex.get_burn_in_mask_simple(
                        hsv, back, ROI=roi, h_threshold=16, v_threshold=16,
                        gaussian_kernel=(7, 7), closing_kernel=(3, 6))
                    shadow = ex.get_shadow_mask_simple(
                        hsv, back, ROI=roi, h_threshold=5, s_threshold=None,
                        opening_kernel=(5, 5), closing_kernel=(5, 5))
                    foreground = ex.find_player_candidate_contours_timeblur(
                        {0: burn}, {0: shadow}, {}, ROI=roi, frame_blur=0, min_player_size=50)
                    if 8 <= index < count - 8:
                        grey_cache = {offset: cache[index + offset][1] for offset in [0, -4, 4, -6, 6, -8, 8]}
                        mask = ex.find_motion_mask(grey_cache, ROI=roi, intensity_threshold=5,
                                                   open_kernel=None, close_kernel=(3, 3))
                        # The preview has no identity tracker: suppress every large foreground candidate.
                        if not foreground.is_empty():
                            exclusion = mask.within_contour(foreground[0])
                            for candidate in list(foreground)[1:]:
                                exclusion = exclusion | mask.within_contour(candidate)
                            mask = mask & ~exclusion.dilate((12, 7))
                        motion = ex.get_contours_from_mask(mask, max_number_of_contours=200)
                        motion = motion if motion is not None else Contours([])
                        if svm is not None:
                            try:
                                balls, candidate_rows = ball_candidates(motion, hsv, grey, svm)
                                for row in candidate_rows:
                                    ball_rows.append({'frame': index, 'source_seconds': args.start + index / FPS, **row})
                            except Exception as error:
                                if not errors:
                                    errors.append(f'{type(error).__name__}: {error}')
                                svm = None  # Fail once, report it, and retain the other stages' outputs.
                foreground_count += foreground.length()
                motion_count += motion.length()
                balls_count += balls.length()
                rows.append({'frame': index, 'source_seconds': args.start + index / FPS,
                             'is_court': is_court, 'court_fp_rate': fp, 'court_fn_rate': fn,
                             'foreground_candidates': foreground.length(),
                             'motion_candidates': motion.length(), 'ball_svm_candidates': balls.length()})
                shown = color
                if is_court:
                    shown = shown.put(court, color=(100, 230, 100), thickness=1)
                    shown = shown.put(foreground, color=(230, 180, 50), thickness=1)
                    shown = shown.put(motion, color=(160, 160, 160), thickness=1)
                    shown = shown.put(balls, color=(0, 230, 255), thickness=2)
                label = 'ATMAN v1 | green: court | blue: foreground | yellow: SVM ball candidates'
                shown = shown.put(label, where=(8, 18), color=(255, 255, 255), size=0.85)
                if not is_court:
                    shown = shown.put('Court not detected in this frame', where=(8, 38), color=(0, 190, 255))
                writer.stdin.write(np.ascontiguousarray(shown.array).tobytes())
                if index == min(8, count - 1):
                    if not cv.imwrite(os.path.join(output, 'preview.png'), shown.array):
                        raise RuntimeError('Could not write preview.png.')
                cache.pop(index - 8, None)
        finally:
            cap.release()
            writer.stdin.close()
        stderr = writer.stderr.read().decode('utf-8', errors='replace')
        if writer.wait() != 0:
            raise RuntimeError(f'Could not encode visualization: {stderr}')
    with open(os.path.join(output, 'frame_features.csv'), 'w', newline='', encoding='utf-8') as handle:
        out = csv.DictWriter(handle, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    with open(os.path.join(output, 'ball_candidates.csv'), 'w', newline='', encoding='utf-8') as handle:
        out = csv.DictWriter(handle, fieldnames=['frame', 'source_seconds', 'x', 'y', 'probability', 'svm_positive'])
        out.writeheader()
        out.writerows(ball_rows)
    court_frames = sum(row['is_court'] for row in rows)
    report['stages']['foreground_and_motion'] = {
        'status': 'ok' if court_frames else 'not_detected',
        'court_frames': court_frames,
        'foreground_candidates': foreground_count, 'motion_candidates': motion_count,
    }
    if report['stages']['ball_svm']['status'] == 'loaded':
        report['stages']['ball_svm'].update(
            status='failed' if errors else ('ok' if court_frames else 'skipped'), errors=errors,
            evaluated_candidates=len(ball_rows), positive_candidates=balls_count)
    return rows


def recognize_audio(args, output, report):
    if args.no_audio:
        report['stages']['sound_cnn'] = {'status': 'skipped', 'reason': '--no-audio supplied.'}
        return None
    model_dir = args.sound_model_dir
    if not model_dir and args.data_root:
        model_dir = os.path.join(args.data_root, 'cnn_sound')
    if not model_dir or not os.path.isfile(os.path.join(model_dir, 'weights', SOUND_MODEL)):
        report['stages']['sound_cnn'] = {'status': 'skipped', 'reason': 'No local sound-CNN weights supplied.'}
        return None
    try:
        import torch
        import dill
        import librosa
        import numpy as np
        import pandas as pd
        import torchaudio.transforms as transforms
        import warnings
        from ATMANpy.models.SoundCNN.nets import SoundModel_MN

        torch.set_num_threads(args.threads)
        with open(os.path.join(model_dir, 'params', SOUND_MODEL + '.pkl'), 'rb') as handle:
            params = dill.load(handle)
        audio = run_ffmpeg([
            '-ss', str(args.start), '-i', os.path.abspath(args.video), '-t', str(args.duration),
            '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '44100', '-f', 'f32le', 'pipe:1',
        ], args.threads, stdout=subprocess.PIPE).stdout
        waveform = np.frombuffer(audio, dtype='<f4').copy()
        if len(waveform) < params['hops'] * params['hop_length']:
            raise ValueError('Audio is shorter than the sound model receptive field (about one second).')
        model = SoundModel_MN(params['conv_filters'], params['conv_kernels'],
                              params['pool_kernels'], params['fc_neurons'],
                              frequencies=params['freqs'], hops=params['hops'],
                              initial_output_shares=[0.1, 0.1])
        weights = os.path.join(model_dir, 'weights', SOUND_MODEL)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            mel = transforms.MelSpectrogram(sample_rate=44100, n_fft=params['window_length'],
                                            hop_length=params['hop_length'], n_mels=params['mel_bands'])
            spectrum = mel(torch.from_numpy(waveform)).numpy()
        if params['to_db']:
            spectrum = librosa.power_to_db(spectrum, ref=params['db_reference'])
        if params['norm_value'] is not None:
            spectrum = spectrum / params['norm_value']
        with torch.inference_mode():
            probabilities = model(torch.from_numpy(spectrum[None, None]).float()).sigmoid()[0, :, 0].numpy()
        stride = int(np.prod(params['pool_kernels'])) * params['hop_length']
        center = ((2 * params['wing_size'] + 1) * params['hop_length'] - 1) / 2
        seconds = args.start + (np.arange(probabilities.shape[1]) * stride + center) / 44100
        table = pd.DataFrame({'source_seconds': seconds, 'hit_probability': probabilities[0],
                              'bounce_probability': probabilities[1]})
        table.to_csv(os.path.join(output, 'sound_probabilities.csv'), index=False)
        report['stages']['sound_cnn'] = {
            'status': 'ok', 'weights': os.path.abspath(weights), 'probability_samples': len(table),
            'warnings': [str(warning.message) for warning in caught],
            'max_hit_probability': float(probabilities[0].max()),
            'max_bounce_probability': float(probabilities[1].max()),
        }
        return table
    except Exception as error:
        message = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            message = error.stderr.decode('utf-8', errors='replace').strip()
        report['stages']['sound_cnn'] = {'status': 'failed', 'error': message}
        return None


def make_plot(rows, audio, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(2 if audio is not None else 1, 1, figsize=(10, 5), squeeze=False)
    table.plot(x='source_seconds', y=['foreground_candidates', 'motion_candidates', 'ball_svm_candidates'], ax=axes[0, 0])
    axes[0, 0].set_ylabel('Candidate count')
    axes[0, 0].set_title('ATMAN v1 — recognition attempt')
    if audio is not None:
        audio.plot(x='source_seconds', y=['hit_probability', 'bounce_probability'], ax=axes[1, 0])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel('CNN probability')
    axes[-1, 0].set_xlabel('Time in source video (seconds)')
    fig.tight_layout()
    fig.savefig(os.path.join(output, 'recognition_analysis.png'), dpi=130)
    plt.close(fig)


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        command.error(str(error))
    if shutil.which('ffmpeg') is None:
        command.error('ffmpeg must be installed and available on PATH.')
    for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[variable] = str(args.threads)
    output = os.path.abspath(args.output or os.path.join(
        'testing_outputs', 'recognition_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    os.makedirs(output, exist_ok=True)
    if os.listdir(output):
        command.error('Output directory is no longer empty.')
    started = time.monotonic()
    report = {
        'input_video': os.path.abspath(args.video), 'start_seconds': args.start,
        'requested_duration_seconds': args.duration, 'processing_resolution': [WIDTH, HEIGHT],
        'processing_fps': FPS, 'cpu_threads': args.threads, 'worker_processes': 0,
        'stages': {},
        'limitations': [
            'Research preview for a fixed, elevated view of a complete tennis court; camera cuts and zooms can fail.',
            'Input is resized to 640x360 and sampled at 25 fps, matching the historical Brest defaults.',
            'Foreground contours and SVM positives are candidates, not identified or temporally tracked players/balls.',
            'Background is estimated from at most 32 frames in this excerpt; stationary objects can disappear.',
            'The first and last eight frames have no motion/ball result because temporal context is missing.',
            'No shot classification, score interpretation, identity assignment or 3D trajectory reconstruction is performed.',
            'Sound outputs are independent hit/bounce probabilities; they are not validated event labels.',
        ],
    }
    exit_code = 1
    try:
        import cv2 as cv
        cv.setNumThreads(args.threads)
        normalized = normalize_video(args, output)
        cap = open_video(normalized, args.threads)
        try:
            count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            if count < 1:
                raise ValueError('The selected excerpt contains no decodable video frames.')
            report['decoded_frames'] = count
            report['processed_duration_seconds'] = count / FPS
            print(f'Calibrating from at most {args.calibration_samples} frames; processing {count} frames.', flush=True)
            court, _ = calibrate(cap, count, args.calibration_samples, report)
            back = background(cap, count) if court is not None else None
        finally:
            cap.release()
        svm = load_models(args, report)
        rows = process_frames(normalized, output, count, court, back, svm, args, report)
        audio = recognize_audio(args, output, report)
        make_plot(rows, audio, output)
        failed = any(stage.get('status') == 'failed' for stage in report['stages'].values())
        court_frames = report['stages']['foreground_and_motion']['court_frames']
        exit_code = 1 if failed else (0 if court_frames else 2)
    except Exception as error:
        message = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            message = error.stderr.decode('utf-8', errors='replace').strip()
        report['error'] = f'{type(error).__name__}: {message}'
        print(report['error'], file=sys.stderr)
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report['exit_code'] = exit_code
        write_json(os.path.join(output, 'report.json'), report)
    print(f'Recognition report: {os.path.join(output, "report.json")}', flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
