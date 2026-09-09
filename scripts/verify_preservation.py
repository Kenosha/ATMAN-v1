"""Compare the current demo harness against original and extracted ATMAN packages.

The two runs use the same saved features, interpreter and harness. This verifies
the exercised data queries, calculations and renderings; it does not rerun the
recognition pipeline or establish equivalence for every historical script.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A file passed to run_path does not install the new checkout on sys.path.
# Import ATMANpy first and verify every ATMANpy module again after the demo runs.
RUNNER = r'''
import json
import os
import runpy
import sys

selected_root, harness, result_dir = sys.argv[1:4]
package_root = os.path.realpath(os.path.join(selected_root, "ATMANpy"))
sys.path.insert(0, selected_root)
import ATMANpy

def inspect_origins():
    origins = {}
    for name, module in sorted(sys.modules.items()):
        if name == "ATMANpy" or name.startswith("ATMANpy."):
            location = getattr(module, "__file__", None)
            if location is None:
                paths = [os.path.realpath(path) for path in getattr(module, "__path__", [])]
                assert paths, f"No source location for {name}"
            else:
                paths = [os.path.realpath(location)]
            for path in paths:
                assert os.path.commonpath([path, package_root]) == package_root, (name, path, package_root)
            origins[name] = paths
    return origins

inspect_origins()
sys.argv = [harness] + sys.argv[4:]
try:
    runpy.run_path(harness, run_name="__main__")
finally:
    origins = inspect_origins()
    with open(os.path.join(result_dir, "import_origins.json"), "w", encoding="utf-8") as handle:
        json.dump({"selected_root": selected_root, "modules": origins}, handle, indent=2)
        handle.write("\n")
'''


def inside(path, directory):
    return os.path.commonpath([os.path.realpath(path), os.path.realpath(directory)]) == os.path.realpath(directory)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def artifact_paths(directory, suffix):
    paths = set()
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(suffix):
                paths.add(os.path.relpath(os.path.join(root, filename), directory))
    return paths


def compare_png(left, right):
    import cv2 as cv
    import numpy as np

    first, second = cv.imread(left, cv.IMREAD_UNCHANGED), cv.imread(right, cv.IMREAD_UNCHANGED)
    decodable = first is not None and second is not None
    equal = bool(decodable and first.dtype == second.dtype and np.array_equal(first, second))
    return {'equal': equal,
            'original_shape': list(first.shape) if first is not None else None,
            'new_shape': list(second.shape) if second is not None else None}


def open_video(path):
    import cv2 as cv

    if hasattr(cv, 'CAP_PROP_N_THREADS'):
        return cv.VideoCapture(path, cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, 1])
    return cv.VideoCapture(path)


def compare_mp4(left, right):
    import cv2 as cv
    import numpy as np

    first, second = open_video(left), open_video(right)
    counts = [0, 0]
    mismatch = None
    try:
        opened = first.isOpened() and second.isOpened()
        fps = [first.get(cv.CAP_PROP_FPS), second.get(cv.CAP_PROP_FPS)]
        index = 0
        while opened:
            first_ok, first_frame = first.read()
            second_ok, second_frame = second.read()
            counts[0] += int(first_ok)
            counts[1] += int(second_ok)
            if not first_ok and not second_ok:
                break
            if mismatch is None and (first_ok != second_ok or not np.array_equal(first_frame, second_frame)):
                mismatch = index
            index += 1
        equal = bool(opened and counts[0] > 0 and counts[0] == counts[1]
                     and mismatch is None and np.isclose(fps[0], fps[1], rtol=0, atol=1e-6))
        return {'equal': equal, 'decoded_frames': counts, 'fps': fps, 'first_different_frame': mismatch}
    finally:
        first.release()
        second.release()


def compare_gif(left, right):
    import numpy as np
    from PIL import Image

    with Image.open(left) as first, Image.open(right) as second:
        counts = [first.n_frames, second.n_frames]
        loops = [first.info.get('loop'), second.info.get('loop')]
        mismatch = None
        timings_equal = True
        for index in range(min(counts)):
            first.seek(index)
            second.seek(index)
            if first.info.get('duration') != second.info.get('duration'):
                timings_equal = False
            if mismatch is None and not np.array_equal(np.asarray(first.convert('RGBA')), np.asarray(second.convert('RGBA'))):
                mismatch = index
        return {'equal': bool(counts[0] > 0 and counts[0] == counts[1] and mismatch is None
                              and timings_equal and loops[0] == loops[1]),
                'decoded_frames': counts, 'first_different_frame': mismatch,
                'frame_timings_equal': timings_equal, 'loop_settings': loops}


def compare_artifacts(original, new, suffix, compare):
    first_paths, second_paths = artifact_paths(original, suffix), artifact_paths(new, suffix)
    results = {}
    for name in sorted(first_paths | second_paths):
        left, right = os.path.join(original, name), os.path.join(new, name)
        if name not in first_paths or name not in second_paths:
            results[name] = {'equal': False, 'error': 'Artifact is missing from one run.'}
            continue
        try:
            result = compare(left, right)
            result['original_sha256'] = sha256(left)
            result['new_sha256'] = sha256(right)
            result['bytes_equal'] = result['original_sha256'] == result['new_sha256']
            results[name] = result
        except Exception as error:
            results[name] = {'equal': False, 'error': f'{type(error).__name__}: {error}'}
    return {'equal': bool(results) and all(result['equal'] for result in results.values()), 'files': results}


def compare_bytes(left, right):
    return {'equal': sha256(left) == sha256(right)}


def run_demo(label, package_root, harness, args, output):
    result_dir = os.path.join(output, label)
    os.makedirs(result_dir)
    cache = os.path.join(output, '.cache', label)
    environment = os.environ.copy()
    environment.update({
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': package_root,
        'ATMAN_HOME': args.data_root, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'MPLBACKEND': 'Agg',
        'MPLCONFIGDIR': os.path.join(cache, 'matplotlib'), 'NUMBA_CACHE_DIR': os.path.join(cache, 'numba'),
        'XDG_CACHE_HOME': cache,
    })
    command = [sys.executable, '-B', '-c', RUNNER, package_root, harness, result_dir,
               '--data-root', args.data_root, '--output', result_dir,
               '--max-frames', str(args.max_frames)]
    if args.video:
        command.extend(['--video', args.video])
    if args.sample:
        command.append('--sample')
    log = os.path.join(output, label + '.log')
    print(f'Running the same demo with {label} package: {package_root}', flush=True)
    started = time.monotonic()
    with open(log, 'w', encoding='utf-8') as handle:
        process = subprocess.run(command, cwd=output, env=environment,
                                 stdout=handle, stderr=subprocess.STDOUT, check=False)
    result = {'package_root': package_root, 'exit_code': process.returncode,
              'elapsed_seconds': round(time.monotonic() - started, 3),
              'log': os.path.relpath(log, output), 'output': label}
    origin_path = os.path.join(result_dir, 'import_origins.json')
    if os.path.isfile(origin_path):
        origins = read_json(origin_path)
        result['verified_imported_modules'] = len(origins['modules'])
        result['import_origins'] = os.path.relpath(origin_path, output)
    else:
        result['verified_imported_modules'] = 0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dir', required=True, help='Original repository directory, read-only for this check.')
    parser.add_argument('--new-dir', default=REPO, help='Extracted repository directory; defaults to this checkout.')
    parser.add_argument('--data-root', help='Shared data root; defaults to the extracted repository.')
    parser.add_argument('--output', default=os.path.join(REPO, 'testing_outputs', 'preservation'),
                        help='New or empty result directory.')
    parser.add_argument('--video', help='Optional matching original local video for the broadcast overlay.')
    parser.add_argument('--max-frames', type=int, default=25, help='Rendered frame limit, 1 to 500 (default: 25).')
    parser.add_argument('--sample', action='store_true', help='Use examples/brest_sample.csv under the shared data root.')
    args = parser.parse_args(argv)
    args.original_dir, args.new_dir = os.path.abspath(args.original_dir), os.path.abspath(args.new_dir)
    args.data_root = os.path.abspath(args.data_root or args.new_dir)
    output = os.path.abspath(args.output)
    if not 1 <= args.max_frames <= 500:
        parser.error('--max-frames must be between 1 and 500.')
    for label, root in [('original', args.original_dir), ('new', args.new_dir)]:
        if not os.path.isdir(os.path.join(root, 'ATMANpy')):
            parser.error(f'The {label} directory must contain ATMANpy/.')
    if os.path.realpath(args.original_dir) == os.path.realpath(args.new_dir):
        parser.error('Original and new package roots must be different directories.')
    if inside(output, args.original_dir):
        parser.error('Outputs must be outside the original repository to preserve it unchanged.')
    if not os.path.isdir(args.data_root):
        parser.error('--data-root must be an existing directory.')
    if args.video:
        args.video = os.path.abspath(args.video)
        if not os.path.isfile(args.video):
            parser.error('--video must be an existing local file.')
    if args.sample and args.video:
        parser.error('--video requires the full feature store and its camera calibration; omit --sample.')
    harness = os.path.join(args.new_dir, 'ATMANpy', 'cli', 'demo.py')
    if not os.path.isfile(harness):
        parser.error(f'The current new demo harness is missing: {harness}')
    if os.path.exists(output) and (not os.path.isdir(output) or os.listdir(output)):
        parser.error('--output must be new or empty; existing results are preserved.')
    os.makedirs(output, exist_ok=True)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    started = time.monotonic()
    harness_hash = sha256(harness)
    report = {'passed': False, 'python': sys.executable, 'harness': harness,
              'harness_sha256': harness_hash, 'data_root': args.data_root,
              'sample': args.sample, 'video': args.video, 'max_frames': args.max_frames,
              'runs': {}, 'checks': {},
              'scope': 'Same current harness and saved inputs; comparisons cover exercised queries, calculations and renderings, not the full recognition pipeline.'}
    try:
        for label, root in [('baseline', args.original_dir), ('export', args.new_dir)]:
            report['runs'][label] = run_demo(label, root, harness, args, output)
        checks = report['checks']
        checks['both_runs_completed'] = all(run['exit_code'] == 0 for run in report['runs'].values())
        checks['import_origins_verified'] = all(run['verified_imported_modules'] > 0 for run in report['runs'].values())
        checks['same_harness_unchanged'] = sha256(harness) == harness_hash
        original, new = os.path.join(output, 'baseline'), os.path.join(output, 'export')
        if checks['both_runs_completed'] and checks['import_origins_verified']:
            original_report = read_json(os.path.join(original, 'report.json'))
            new_report = read_json(os.path.join(new, 'report.json'))
            checks['metrics_equal'] = original_report['metrics'] == new_report['metrics']
            checks['rendering_metadata_equal'] = original_report['rendering'] == new_report['rendering']
            checks['input_metadata_equal'] = all(original_report[key] == new_report[key] for key in ['match', 'source'])
            report['metrics'] = {'original': original_report['metrics'], 'new': new_report['metrics']}
            csv_hashes = [sha256(os.path.join(folder, 'shots.csv')) for folder in [original, new]]
            report['shots_csv_sha256'] = {'original': csv_hashes[0], 'new': csv_hashes[1]}
            checks['shots_csv_bytes_equal'] = csv_hashes[0] == csv_hashes[1]
            report['csv'] = compare_artifacts(original, new, '.csv', compare_bytes)
            report['png'] = compare_artifacts(original, new, '.png', compare_png)
            report['gif'] = compare_artifacts(original, new, '.gif', compare_gif)
            report['mp4'] = compare_artifacts(original, new, '.mp4', compare_mp4)
            checks['all_csv_bytes_equal'] = report['csv']['equal']
            checks['png_pixels_equal'] = report['png']['equal']
            checks['gif_frames_equal'] = report['gif']['equal']
            checks['mp4_frames_equal'] = report['mp4']['equal']
        report['passed'] = bool(checks) and all(checks.values())
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write_json(os.path.join(output, 'preservation.json'), report)
        status = 'PASS' if report['passed'] else 'FAIL'
        with open(os.path.join(output, 'README.md'), 'w', encoding='utf-8') as handle:
            handle.write(f'# Preservation check: {status}\n\n')
            handle.write('The same current demo harness ran sequentially against the original and extracted ATMANpy packages, using the same saved inputs and Python environment. Imported package locations are recorded for both runs.\n\n')
            for name, value in report['checks'].items():
                handle.write(f'- {"PASS" if value else "FAIL"}: {name.replace("_", " ")}\n')
            if report.get('error'):
                handle.write(f'\nError: {report["error"]}\n')
            handle.write('\nSee `preservation.json`, `baseline.log`, `export.log`, and each run\'s artifacts for details. PNG/GIF/MP4 comparisons use decoded pixels; container-byte hashes are also recorded.\n\n')
            handle.write('This verifies the exercised feature queries, numeric calculations and renderings. It does not rerun the recognition pipeline or prove equivalence for all historical scripts. The original repository is only imported/read; outputs and caches are kept here.\n')
    print(f'{"PASS" if report["passed"] else "FAIL"}: {os.path.join(output, "preservation.json")}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
