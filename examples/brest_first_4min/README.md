# First four Brest minutes

`human_review.json` contains the reviewed inputs for the first 240 seconds of
`2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa`, normalized to 25 fps
(frames 0 through 5999, inclusive). It records player names and handedness,
the absence of a warm-up or tiebreak correction, and eight reviewed serve
categories. Video, model weights, and automatically inferred features are
separate local inputs.

The identity intervals leave the changeover gap unlabelled. Serve frame
numbers locate the reviewed actions so their categories can be matched to
newly detected serves; the pipeline does not copy those times into its fresh
detections. These labels must not be reused for a different excerpt or match.

See [the full pipeline guide](../../docs/FULL_PIPELINE.md) for preparation,
asset requirements, and the headless review adapters.
