# Small Brest example

`brest_sample.csv` contains the first 12 reconstructed points (65 shots, 2,003 feature rows) from the saved Nakashima–Sousa Brest match. It includes trajectories, player positions, shot events, and score context used by ATMAN's existing `Points` and `Shots` classes.

Run `python -m ATMANpy.cli.demo --sample` from the repository root. No model weights, external feature archive, or match video are needed. FFmpeg is needed for animation export.

These are historical model estimates, with the corrections present in the original feature store. They are not measured ground truth. Speed and spin summaries use the original conversion methods; exported bounce coordinates retain ATMAN's internal court units.

The exact source tables, their hashes, and the selection rule are in [provenance.json](provenance.json). Expected results in `tests/fixtures/` were calculated with the original ATMAN checkout before comparison with this export. The licensing decision for publication also covers this example; see [licensing status](../docs/LICENSING.md).
