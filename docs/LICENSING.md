# Licensing status

A project-wide license for the original ATMAN code has not yet been selected. This is a public source repository for portfolio review; this file is not an open-source license grant. GitHub explains its default licensing behavior in [Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

The generic hidden Markov model implementation is now distributed separately as
[kenosha-hmm](https://github.com/Kenosha/kenosha-hmm), under the
[MIT licence](https://github.com/Kenosha/kenosha-hmm/blob/v0.1.0/LICENSE).
ATMAN imports that library through compatibility adapters; the library's MIT
grant covers the separate package. It does not license ATMAN's adapters,
tennis-specific postprocessing, data, weights or other source files. See the
[HMM integration guide](HMM.md).

The retained code includes these integrations, verified against imports and the local installed package metadata:

| Dependency | Local version | Upstream license | Use in this checkout |
| --- | --- | --- | --- |
| [Ultralytics](https://www.ultralytics.com/legal/agpl-3-0-software-license) | 8.3.228 | AGPLv3 | Player ReID adapter |
| [Pedalboard](https://github.com/spotify/pedalboard#license) | 0.9.19 | GPLv3 | Historical narration editing |
| [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy/blob/master/LICENSE.txt) | 0.18.0 | GPLv2 | Joining trajectory and player/score information |

This is a list of the copyleft integrations identified during extraction, not a complete inventory of every transitive dependency. Each dependency retains its own terms. The project-wide licensing decision remains open; making integrations optional does not change the terms of their dependencies.

The small example table contains estimated historical ATMAN output; its provenance accompanies the file. Full match videos, feature archives, recovered trained models, and narration recordings are not included in the Git repository. One small early SVM pickle already tracked under `ball_svm/` remains in the filtered history; it is a historical artifact, not the checkpoint used by the current recognition entry points. The three original website demonstrations are included as GIFs under `docs/media/`, with their provenance recorded there. They do not grant rights to underlying broadcast footage.
