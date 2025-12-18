# Brest Legacy Swing Pipeline (3D CNN / TemporalCNN)

This note summarizes the older Brest pipeline under `scripts/matches/brest/` that (1) labeled tennis actions, (2) built clip datasets, (3) trained a small 3D CNN (“TemporalCNN”), and (4) ran sliding-window inference for segmentation/spotting.

## 1) Labeling (what was labeled, and how)

### 1.1 “Point events” (hit / serve / other) are recorded framewise

Core UI primitive: `ATMANpy.funcs.label_funcs.record_events(...)` records *sparse* event markers as strings (e.g. `"hit impact"`) into a `StreamFeature` that is saved to disk.

Event sub-states come from `ATMANpy/funcs/label_funcs.py:get_event_dict()`:

- `hit`: `begin loading` → `begin swinging` → `impact` → `end swing`
- `serve`: `begin bouncing` → `stop bouncing` → `ball leaves hand` → `begin swinging` → `impact` → `end swing`
- `other`: “begin …” labels for standing/walking/running/split step/etc + an explicit `end`

Scripts driving this stage:

- `scripts/matches/brest/labelling/player_motion_labelling.py`: chooses “what to label next” (video + `{top|bot}×{hit|serve}`) by counting already-recorded `"begin swinging"` markers; then runs `record_events(...)` over each `isCourt` segment.
- `scripts/matches/brest/labelling/player_motion_labelling_fixed_segment.py`: same as above but for a fixed `VIDNUM` and segment selection.
- `scripts/matches/brest/labelling/player_motion_labelling_redos.py`: redoes event labeling inside “redo” spans (see 1.3).

Saved feature names follow the pattern:

- `labelled_{player}_{event}_Flag` where `player ∈ {top, bot}` and `event ∈ {hit, serve, other}`

### 1.2 “Specification” adds categories (side/spin/type/hand)

After `hit` / `serve` are marked, the pipeline adds categorical attributes for each grouped event using `ATMANpy.funcs.label_funcs.classify_event(...)` (a looping UI that assigns a single category label for the whole shown segment).

Category dictionaries live in `ATMANpy/funcs/label_funcs.py:get_category_dict(event, category)`:

- `hit/side`: `forehand`, `backhand`
- `hit/spin`: `topspin`, `slice`, `flat`
- `hit/type`: `groundstroke`, `volley`, `overhead`, `other`
- `serve/side`: `left`, `right`
- (`hand` exists for both hit/serve in the legacy dict; not always used everywhere)

Script driving this stage:

- `scripts/matches/brest/labelling/player_motion_specification.py`: for each `(video, player, event)`, groups hits/serves (`group_events(...)`) and calls `classify_event(...)` to label “side”, then “spin”, then “type” (with small before/after margins for some categories).

Saved feature names follow the pattern:

- `labelled_{player}_{event}_{category}_Flag` (e.g. `labelled_bot_hit_spin_Flag`)

### 1.3 “Checking” and redo flows

- `scripts/matches/brest/labelling/player_motion_checking.py`:
  - Finds incomplete groups (`check_group(...)`) and re-runs `record_events(...)` around those spans.
  - Has an “uncaught events” mode: builds a `notLabelled` mask inside `isCourt`, then runs `classify_event(..., event='uncaught_check', category=VERSION, fps=100)` to tag spans as `ok` vs `redo` (via `get_category_dict('uncaught_check', ...)`).
- `scripts/matches/brest/labelling/player_motion_labelling_redos.py`: reads the `labelled_all_uncaught_check_{VERSION}_Flag` feature and re-labels inside frames tagged `"redo"`.

### 1.4 “Other” labeling is done in gaps between serves/hits

- `scripts/matches/brest/labelling/player_motion_filling.py`:
  - Expands sparse hit/serve markers into filled phase labels (`fill_in_group(...)` / `fill_in_group_2(...)` via `get_fill_dict(...)`).
  - Finds “non-event” spans (inside `isCourt` but outside filled hit/serve phases) and runs `record_events(... which_event='other', auto_ending_event='other end')`.

## 2) Feature creation (contours, masks, segmentation helpers)

The action model training/inference was not run on raw frames: it used derived masks and player contours/features produced earlier.

- `scripts/matches/brest/labelling/player_motion_segmentation.py` writes `isPlay` spans by taking serve start frames and building “between serve” segments (note: the file includes a prominent TODO about missing the last segment).
- `scripts/matches/brest/labelling/player_motion_feature_creation.py` is a large “feature factory” that (among other things):
  - Builds `foregroundMask`, `fine_motMask`, `shadowMask`, `burn_in_mask` style features.
  - Builds player contour tracks/features like `topPlayer` / `botPlayer` (and quality flags like `{player}Player_fine`).

Those masks/contours become prerequisites for clip creation (next section) and some inference paths.

## 3) Clip dataset creation (what is saved to disk)

All dataset builders write clips under:

- `video_files/labelled_data/<DATASET_FOLDER>/<vid>/clip_*.avi`
- `video_files/labelled_data/<DATASET_FOLDER>/<vid>/labels.csv`

There are (at least) two label-table “shapes” in these legacy scripts:

- **Older “separate hit/serve columns” shape** (primarily `cnn_clip_creation.py`): columns look like `hit_event`/`serve_event`-style (derived from `labelled_{PLAYER}_{hit|serve}_Flag`), and values can be phase strings like `"hit swinging"` / `"serve impact"`.
- **TemporalCNN “unified event/side/spin/type/hand” shape** (used by `ClipDataset` and the later pipelines): columns are `event`, `side`, `hand`, `spin`, `type` and values are category strings like `event="hit"`, `side="forehand"`, etc.

In both cases the index is `(clip, frame)` (multi-index), but only the unified shape plugs directly into `ATMANpy.models.TemporalCNN.data.ClipDataset`.

### 3.1 Plain clip creation (player-centered crops, no extra crop augmentation)

- `scripts/matches/brest/training/cnn_clip_creation.py`
  - Builds fixed-length clips by sliding a window across strong `isPlay` spans (restricted by `{player}Player_fine`).
  - Crops each clip around the enclosing region of the player contours inside the clip (`Contours(...).enclosing_regions().cover()`), then resizes to a fixed `H×W`.
  - Writes one `clip_{idx}.avi` per window (plus `labels.csv`).

### 3.2 “Impact-only” dataset (masked background, high temporal overlap)

- `scripts/matches/brest/training/cnn_clip_creation_impactonly.py`
  - Uses masked frames: union of `foregroundMask | fine_motMask` (same logic as `ATMANpy.models.TemporalCNN.data.get_masked_video_frames`).
  - Crops around player contours, then *shrinks-to-fit* inside `H×W` with padding (`resize(... how='any')` + `pad(...)`).
  - Often uses higher overlap (e.g. `T_STRIDE = 2` with `T_LENGTH = 32`) for dense temporal supervision.

### 3.3 Random-crop dataset (“randcropN” on disk, not just at train time)

- `scripts/matches/brest/training/cnn_clip_creation_crops.py`
  - Also masks background via `foregroundMask | fine_motMask`.
  - For each temporal window, saves multiple spatial variants: `clip_{idx}_{cropNum}.avi` with `CROPS = 5`.
  - For event windows: generates crops by resizing the player region with random height/width ratios but keeping the player in view.
  - For non-event windows: additionally jitters crop center (random shifts) to make “non-event” more robust.

## 4) Training (TemporalCNN)

Training entrypoint:

- `scripts/matches/brest/training/cnn_with_pytorch_from_disc.py`

### 4.1 Dataset / augmentation at training-time

Dataset:

- `ATMANpy/models/TemporalCNN/data.py:ClipDataset`

Key preprocessing inside `ClipDataset.__getitem__`:

- Reads clip from disk (`torchvision.io.read_video`)
- `CenterCrop(d_crop_size)` then `Resize(d_scale_size)`
- Random horizontal flip with `flip_prob`:
  - Applies `horizontal_flip` to the clip
  - Swaps the `hand` label `left ↔ right` in the label frame

Background masking is *not* applied inside `ClipDataset` by default; it is applied upstream during clip creation (see 3.2/3.3), and is also available for inference preprocessing (`get_masked_video_frames`).

### 4.2 Soft temporal targets (“proximity” already existed)

`ClipDataset.label_frame_to_vector(...)` produces a *soft* target vector:

- If a hit/serve is present in the window:
  - Computes `proximity = max(prox_func(impact_index, central_index))`
  - Assigns `p(other) = 1 - proximity`, `p(class) = proximity`
- Else: assigns `p(other) = 1`

This is exactly the “temporal centering” idea: off-center windows are trained to be partly “other”.

The proximity function is user-defined in `cnn_with_pytorch_from_disc.py` as `PROX_FUNC(...)` and injected via `data.set_proximity_function(PROX_FUNC)`.

### 4.3 Model architecture

Model:

- `ATMANpy/models/TemporalCNN/nets.py:SmallFullConvModel`

Highlights:

- 2+1D blocks (`TwoPlusOneBlock`) + residual blocks (`ResidualBlock`) (Tran et al style).
- Fully-convolutional logits head (`conv_fc_out = Conv3d(..., kernel_size=1)`), so inference can be convolutional over time/space.
- Initializes output biases using an assumed `share_other` (and a `share_right`) to roughly match class priors at init.

Training loop (in `cnn_with_pytorch_from_disc.py`):

- `CrossEntropyLoss` over the label dimension.
- `Adam` optimizer.
- Optional tensorboard logging + confusion matrix utilities.

## 5) Inference (sliding-window, convolutional)

Main inference script:

- `scripts/matches/brest/inference/segmentation_testing_convolutional.py`

Core approach:

- Use an `IterableDataset` that yields *long* samples (e.g. `frames_in_sample = 256`) with shape `C×T×H×W`.
- Run `SmallFullConvModel` once per sample and interpret the output as per-time-step logits (due to convolutional structure + temporal stride).

Key building blocks:

- `ATMANpy/models/TemporalCNN/data.py:BaselineInferenceDataset`:
  - Creates masked frames (`get_masked_video_frames(...)`), crops a fixed baseline ROI, preprocesses (`preprocess_cropped_frames`), then yields batched temporal chunks plus a vector of “central frame indices”.
- `ATMANpy/models/TemporalCNN/utils.py:infer_with_tbatch(...)`:
  - Applies softmax, concatenates probabilities across samples to get `prbArr` over time.
- `ATMANpy/models/TemporalCNN/utils.py:make_cat_frame(...)`:
  - Aggregates sublabels into coarse categories (e.g. sums all “forehand/*” labels).
  - If spatial dimensions exist, it selects (per time-step) the spatial location where `p(other)` is minimal, and reads category probs from that location (a “most event-like” location heuristic).

Practical consequence: the inference path is set up to do coarse spotting/segmentation with either a fixed ROI (serve spotting) or by scanning/choosing spatial positions based on the “other” channel.
