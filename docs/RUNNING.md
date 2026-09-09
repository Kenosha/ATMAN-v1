# Running ATMAN v1

## Environment

The local preparation uses Python 3.12. The [core constraints](../requirements/core-constraints.txt) record direct dependency versions available in that environment; they are not a complete dependency lock or a claim of compatibility with other Python versions.

Install from the repository root using the README command. FFmpeg and `ffprobe` must be on `PATH` for video export. A display server or GPU is not required for the sample demo.

NumPy is kept below 2 and OpenCV below 4.12 to preserve the historical numerical stack. Optional neural-network and audio packages are separate from the sample dependencies:

```bash
python -m pip install -e '.[recognition]'
python -m pip install -e '.[legacy]'
```

For the tested Linux CPU setup, install the matching CPU builds first, then the recognition extra:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cpu -c requirements/core-constraints.txt
python -m pip install -e '.[recognition]' -c requirements/recognition-constraints.txt
```

The wheel versions follow [PyTorch's 2.9.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v291). CPU inference is sufficient for the short preview. The recognition constraints record the additional versions exercised with the saved audio model.

`recognition` supplies the older CNN/audio modules. `legacy` additionally supplies the retained OCR, training, narration, and ReID integrations. Installing an extra does not supply trained weights or the historical data those scripts expect. Audio-recording scripts may also need the system PortAudio library, and scoreboard OCR needs the Tesseract executable.

## Stored feature demo

The sample CSV is enough to exercise the data model, filtering, graphs, and 3D rendering. To use the full local match instead, arrange the files beneath a data root as follows:

```text
ATMAN-v1/
├── features_stored/
│   └── 2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa/
│       └── driver_run_store/
├── video_files/
│   └── 2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa.mp4
└── testing_outputs/
```

The equivalent `video_files/<match-id>/match.mp4` layout is also used by historical ATMAN scripts. The stored frame numbers refer to the original full-length video: a shortened video cannot silently replace it for overlays.

```bash
python -m ATMANpy.cli.demo --data-root . --output testing_outputs/full_match
```

To add a broadcast overlay and a stored ball-detection snapshot, append:

```bash
--video video_files/2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa/match.mp4
```

The matching feature store supplies the camera calibration. The full demo writes all shots, first serves, and shots above 150 km/h to separate CSVs, alongside speed/spin graphs, serve-placement maps, and a virtual-court animation. `--max-frames 100` limits the animation to four seconds; `--metrics-only` skips rendering.

The full feature store contains intermediate detection data as well as final reconstructed points. Video is needed only for outputs that show the broadcast frames; graphs and virtual-court animations use the stored data. The demo does not retrain or rerun the original whole-match pipeline.

For historical scripts, set the data root before starting Python:

```bash
export ATMAN_HOME=/absolute/path/to/ATMAN-v1
```

Those scripts use `get_folder_struct()` to find subdirectories such as `features_stored`, `video_files`, and `results_video`.

## Recognition on local video

```bash
python -m ATMANpy.cli.recognize /absolute/path/to/match.mp4 \
  --start 120 --duration 4 --data-root . --output testing_outputs/recognition
python -m ATMANpy.cli.recognize --help
```

Choose a short section of active play with a stable elevated view of most of the court. Camera cuts, close-ups, obscured white court lines, moving cameras, and substantially different viewpoints can prevent useful recognition. The preview normalizes its local clip to 640×360 at 25 fps and uses one CPU thread by default, without a worker pool. Duration defaults to eight seconds and is capped at thirty seconds.

With the base installation, the preview attempts court calibration and matching, foreground/player candidates, and motion contours. It writes the normalized `input_excerpt.mp4`, a silent annotated `recognition.mp4`, `preview.png`, per-frame/candidate CSVs, `recognition_analysis.png`, and `report.json`. The report records skipped and failed stages. Exit code `0` means court recognition completed, `2` means no usable court was detected, and `1` means an execution or model failure. Player candidates are not identified or tracked players, and the preview does not infer complete shot classes, scoring, or 3D trajectories.

Supplying the saved ball SVM enables candidate classification. Supplying the audio model and installing the recognition extra enables hit/bounce probabilities; those are written to `sound_probabilities.csv`. Use `--ball-model` or `--sound-model-dir` to override the asset paths, and `--no-audio` to skip sound. A short smoke test checks that these components execute; it does not establish accuracy across matches.

The recorded local check used seconds 12–16 of the Nakashima–Sousa Brest match: all 12 calibration samples found the court, and all 100 processed frames matched it. The SVM evaluated 838 motion candidates and accepted 62; the sound model produced 67 hit/bounce probability samples. The annotated MP4 decoded all 100 frames. This took 70.17 seconds on one CPU thread, with about 965 MiB peak resident memory and no swaps. Reports and the resource log are in the local ignored `testing_outputs/recognition/` and `testing_outputs/recognition_resources.txt`.

Two compatibility warnings are retained in `report.json`: the SVM was saved with scikit-learn 1.0.2 and loaded with 1.7.2, and the original sound model's 128-band/1024-point mel settings produce an all-zero filter band. The run succeeded, but this does not prove equivalence to the training environment or detection accuracy. The preview also estimates its background from at most 32 excerpt frames, which can leave moving-object ghosts; the first and last eight frames lack the temporal context needed for motion/ball estimates. Non-16:9 footage is stretched to the historical input shape.

The original full pipeline uses learned ball and player-action/audio classifiers. Its configuration refers to these external assets:

| Asset directory beneath `ATMAN_HOME` | Purpose |
| --- | --- |
| `ball_svm/svm_brest/weights/` | Saved ball-candidate SVM |
| `cnn_sound/weights/` and `cnn_sound/params/` | Audio hit classifier and its parameters |
| `cnn_player/weights/` and `cnn_player/params/` | Player-action classifiers and their parameters |
| `trajectories_stored/` | Precomputed flight/bounce trajectory tables for full reconstruction |

The preview recognizes the original `svm_proba_dict_r75_p60_2023-08-15.pickle` SVM and `test_model_20240109_150057` audio weights (with matching `.pkl` parameters). Only load trusted model/parameter files: the historical formats use Python pickle/dill.

These files are not distributed with the repository. The historical configuration also contains manual player identification, score cleanup, and corrections; full reconstruction is a research workflow rather than a single portable model. Model availability and the tested recognition stages are recorded in the generated local test outputs.

## Complete historical pipeline

[Full-pipeline instructions](FULL_PIPELINE.md) cover the 26-stage workflow on a four-minute clip, including fresh ball/player inference, planar and colinear 3D reconstruction, human review, score OCR, and joined shot tables. The runner keeps its feature store under a separate data root and records stage logs, inputs, parameters, and peak memory.

This requires additional local weights, the trajectory bank, a prepared OCR runtime, and a review file for the chosen interval. The defaults use one CPU worker, small neural-network batches, and bounded trajectory scratch space while preserving the full search grid. The stored-feature demo and short recognition preview remain useful independently.

## Historical presentations

`ATMANpy/scripts/visualization_brest/` contains the website snippets, ATMAN walkthrough, serve-placement analysis, chipped-return analysis, and narration/assembly utilities. These scripts document how the finished videos were made; reproducing an entire narrated case study additionally needs the broader historical match collection, edited audio, and its original parameters.

Some early scripts under the root `scripts/` directory predate later refactors and still import the old, absent `ATMANpy.utils` module. Those imports were already broken in the original current checkout. They remain archived research code, rather than supported entry points.

The standalone demo uses one match and the same data/visualization utilities, so it can demonstrate those capabilities without reproducing the whole production workflow.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/verify_preservation.py --original-dir ../ATMAN --data-root . \
  --video video_files/2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa/match.mp4 \
  --max-frames 100 --output testing_outputs/recheck
```

The first command tests the included example, shot filtering, rendering, blank-video recognition, clip boundaries, tracking, classifier feature joins, bounded trajectory calculations, human-review adapters, OCR configuration, and report validation. All 54 tests passed in the local Python 3.12 environment. A fresh installation with only the core dependencies also passed: 53 tests succeeded and the optional full saved-match test was skipped because its private feature store was absent.

The second command needs the original checkout locally. It runs the **same new demonstration harness** twice, with the original package and the extracted package in separate Python processes. It verifies import locations and compares numeric results, CSV bytes, PNG pixels, GIF frames/timings, and every decoded MP4 frame. Use a new or empty output directory. These runs are sequential and do not run recognition or write into the original checkout.

The preparation check passed on 164 reconstructed points and 731 shots: all calculations, three CSVs, five PNGs, a 40-frame GIF, and both 100-frame videos matched exactly, including encoded file bytes. The pair of runs took about 101 seconds and peaked at 432 MiB resident memory. This proves preservation of the exercised functions and saved results; it does not validate every historical script or establish model accuracy.
