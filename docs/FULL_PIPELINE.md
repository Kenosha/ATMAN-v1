# Complete historical broadcast pipeline

The full runner processes a local excerpt through all 26 stages of the Brest workflow, including both 3D reconstruction passes and the final shot-data join. It writes newly inferred features into a separate data root; the original match's stored detections and solved trajectory tables are not inputs.

## Prepare and run

Install the recognition dependencies described in [RUNNING.md](RUNNING.md#environment), plus `fuzzywuzzy` for the final join and `pytesseract` for OCR. Provide the original ball SVM, sound model, both player-action models, and `ShorterTrajectories_g60_v41_s41_h46` trajectory bank beneath `--assets-root`, using the [documented asset directories](RUNNING.md#recognition-on-local-video). These large artifacts remain local and are not supplied by a public source checkout.

After installing the recognition extra, add the two Python packages used by the score stages and final join:

```bash
python -m pip install fuzzywuzzy==0.18.0 pytesseract==0.3.13
```

For OCR, install Tesseract with English language data on the system, or supply an extracted Linux bundle using `--ocr-runtime`. A portable bundle contains `root/usr/bin/tesseract`, its shared libraries, English `eng.traineddata`, and optional Python support under `python/`. The adapter prefers that bundle when present and otherwise finds `tesseract` on `PATH`. It creates the three tennis character-whitelist files locally and passes their paths to OCR; no system configuration edits are needed. An incomplete supplied bundle fails explicitly. The prepared local run records its bundle provenance in `ocr_runtime/provenance.json`; the Python wrapper alone does not supply the executable or language data.

Use a new run directory and an explicitly reviewed annotation file. For the same Brest match beginning at 00:00, the reviewed labels are in [examples/brest_first_4min/human_review.json](../examples/brest_first_4min/human_review.json); they apply only to its first four minutes at 25 fps. Omit `--ocr-runtime` when using system Tesseract:

```bash
python -m ATMANpy.cli.pipeline \
  --run-dir testing_outputs/full_pipeline_recheck \
  --prepare-video /absolute/path/to/match.mp4 \
  --assets-root /absolute/path/to/ATMAN-v1 \
  --human-review /absolute/path/to/human_review.json \
  --ocr-runtime /absolute/path/to/ocr_runtime \
  --all
```

Preparation defaults to the first 240 seconds at the historical 640×360/25-fps input shape. Its `config.json` records the source, local clip, frame interval, model settings, and data root. The runner sets `ATMAN_HOME` to `<run-dir>/data` before importing ATMAN. Model assets are available there, but the feature store starts empty. Do not substitute the original full-length match's feature store for this new one.

To continue a prepared run, or execute one stage whose prerequisites are already present:

```bash
python -m ATMANpy.cli.pipeline --run-dir testing_outputs/full_pipeline_recheck --all --resume
python -m ATMANpy.cli.pipeline --run-dir testing_outputs/full_pipeline_recheck --stage ball_interpolation_plane
```

After changing source inputs or earlier stage parameters, use a new run directory or deliberately rerun every affected downstream stage. Existing later features can otherwise describe an older run.

## Stages and human review

The runner executes these stages in the listed order:

| Steps | Stages |
| --- | --- |
| 1–5 | `court_lines`, `court_roi`, `segmentation_with_lines`, `hits_classify_sound`, `segmentation_with_sound` |
| 6–11 | `manual_player_names_and_handedness`, `manual_fix_tiebreaks`, `court_background`, `court_masks`, `player_candidates`, `player_tracking` |
| 12–17 | `ball_filtered_motion_mask`, `ball_contours`, `ball_inference_svm`, `hits_classify_video`, `segmentation_with_sound_and_vision`, `manual_remove_warm_ups` |
| 18–21 | `ball_inference_hit_candidates`, `ball_inference_particles`, `ball_interpolation_plane`, `manual_serve_identify` |
| 22–26 | `score_detection_testing`, `score_detection_cleanup`, `ball_interpolation_colinear`, `select_colinear_variant`, `join_traj_and_other_info` |

`human_review.json` records the source video ID and reviewed frame range; player identity/handedness intervals; tiebreak corrections; a warm-up policy; and serve references with historical classification values `first_serve`, `second_serve`, `let`, or `not a serve`. The current headless adapters cover the reviewed first four Brest minutes, including no warm-up and no tiebreak correction. They are not a general-purpose annotation interface for every match.

Identity decisions label newly detected court frames only within their reviewed identity intervals. Fresh play outside those intervals stops for review; no identity is invented for unreviewed changeover frames. Serve references are matched one-to-one to newly inferred serve events within 75 frames; their old timestamps are not emitted as new detections. Unmatched fresh events stop for additional review, while old references with no fresh detection are reported and never inserted. Score cleanup checks newly produced OCR values and stops on inconsistent game progression.

The colinear fitter produces alternative `fair` and `fault` tables. For this reviewed interval, the bridge selects the **newly computed fair hypothesis** under the `colinearTable` name expected by the historical join, retaining both alternatives. This is an explicit hypothesis choice, not a finding that every final shot landed in court.

## Resources and outputs

The prepared defaults use one worker and one numerical/FFmpeg thread, four clips per neural-network batch, 64 court-calibration samples, and a one-GB allowance for trajectory scratch arrays. The trajectory bank is read-only memory mapped. The full grid, 25 sampled planar candidates, candidate-minimum counts, and 10,000-particle filter remain unchanged. The scratch allowance is not a process-memory cap: model arrays, particle histories, and intermediate features use additional RAM.

Each stage writes its report and resource record under `stages/`, with console details in `logs/`. Failures stop the run and retain their reports. Look for freshly generated `ball_choicePt`, `trajTable`, both `colinearTable_*` variants, and the final `dataTable`/`dataTable_colinear` features in the new store.

After all stages complete, generate the front-to-back validation report:

```bash
python scripts/report_full_pipeline.py \
  --run-dir testing_outputs/full_pipeline_recheck --max-frames 250
```

The report validates stage completion, fresh classifier outputs, ball/player coverage, both reconstructed trajectory tables, and joined shot data. It writes filtered shot CSVs, five diagnostic images, a broadcast overlay, and a virtual-court video beneath `<run-dir>/report/`. Open `index.html` there. Every image and every video frame is decoded and checked; the default output directory must be new or empty. Projection residuals compare the fitted path with its own input detections and measure internal consistency, not independent accuracy.

Use `python -m ATMANpy.cli.demo --data-root <run-dir>/data --match <video_id> --store fresh --output <run-dir>/visualization` to analyze/render the joined results, taking `video_id` from `config.json`. Supplying the matching local excerpt with `--video` adds broadcast overlays; frame numbers belong to that excerpt.

The run records configuration and source hashes, but sampling in several historical functions remains stochastic. Matching those records does not guarantee identical inferred trajectories on every execution. Completing all stages demonstrates that the workflow executes; it does not establish detection accuracy or remove the original calibration and manual-review assumptions.

## Recorded four-minute check

The local test used the first 240 seconds of Nakashima–Sousa, Brest 2021: 6,000 decoded frames at 25 fps. All 26 stages completed with newly inferred features: 49 shot events including eight serves, 47 planar fitted shots, and 48 colinear fitted shots across eight play segments. Existing human identity, handedness, and serve-category decisions were replayed explicitly. The successful stages took about 54 minutes on one CPU worker and peaked at 2.58 GiB RSS; this excludes preparation, fixes, earlier attempts, and rendering.

The report passed 105 execution, data, and media checks; the regression suite passed 54 tests. It produced filtered shot CSVs, five diagnostic images, and two 193-frame (7.72-second) clips from the first reconstructed point. These clips illustrate results from the full four-minute input.

The two reconstruction methods do not agree equally closely with their input detections: planar median/p90 reprojection residuals were 3.05/14.44 pixels, versus 32.50/95.40 pixels for the selected colinear hypothesis at 640×360. This measures internal consistency, not independent accuracy. One colinear shot had only one actual ball observation and is flagged in `shot_observations.csv`; an empty final flight produced no reconstruction. The historical colinear observation guard counts some rows with inferred x/y and missing z as observed, so its output needs this additional support audit.

The ignored local report is `testing_outputs/full_pipeline_4min/report/index.html`. Weights and trajectory assets were recovered locally; they are still external requirements for reproducing the full run from a public source checkout.
