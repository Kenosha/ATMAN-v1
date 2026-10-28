# Standalone HMM package and ATMAN adapters

The generic model initialization, single-model and embedded Baum–Welch training,
Viterbi scoring and grammar-constrained token passing are extracted into
[kenosha-hmm](https://github.com/Kenosha/kenosha-hmm) (import `kenosha_hmm`),
a separate MIT-licensed library. ATMAN's `hmm` extra pins its initial release
commit so the numerical implementation is reproducible.

## Install

Activate the ATMAN virtual environment and run from ATMAN-v1:

```bash
python -m pip install -e '.[hmm]' -c requirements/core-constraints.txt
```

Git must be available to fetch the pinned source dependency from GitHub. The
standalone library is not published on PyPI. For development, a sibling checkout
can be installed with `python -m pip install -e ../kenosha-hmm` after the command
above.

The `hmm` extra declares the dependency without adding it to users who only need
the public sample and video/trajectory tools. The package requires NumPy/pandas;
its standalone installation does not require ATMAN, OpenCV, PyTorch, weights or
match data. See the [package README](https://github.com/Kenosha/kenosha-hmm#readme) and
[synthetic example](https://github.com/Kenosha/kenosha-hmm/blob/v0.1.0/examples/synthetic.py) for its
array-oriented `fit`, `classify`, `decode` and `GaussianHMM.score` interface.

## Compatibility

`ATMANpy.models.HMM` forwards numerical algorithms to the installed package.
`ATMANpy.funcs.HMM` also resolves the original historical import paths. Both use
the same implementation; the algorithms are no longer copied into ATMAN.
`TokenPass.read_grammar(name)` still finds ATMAN's named bundled grammars, while
the standalone package accepts an explicit path. The existing CSV persistence
and labelled-data helpers remain in ATMAN's `utils` adapter. `HPost.py` and the
tennis-specific types are not moved or licensed as part of the generic package.

The original pandas-based numerical functions remain available from the standalone
modules for archived scripts. New callers can use the smaller array interface.
Model parameter layouts remain `A`, `avg`, `cov`, including non-emitting entry and
exit states; stored CSV formats and named ATMAN grammar files are unchanged.

## Verified corrections

Small independent tests uncovered and fixed two historical issues in the extracted
implementation: embedded re-estimation omitted final-frame entry/exit counts,
and recursive grammar backtraces could repeat the same terminal exit. Log sums
also use stable arithmetic, and importing the library no longer changes global
NumPy error handling. These are documented algorithm fixes, not a claim of exact
old-output equivalence for the affected cases. The extraction is tested with
synthetic sequences; it does not claim revalidation of historical match accuracy.

## Tests

The base test suite skips the four optional HMM integration checks when the
library is not installed. CI first checks that base installation, then installs
`.[hmm]` and runs those four checks. To run them locally after installing the extra:

```bash
python -m unittest discover -s tests -p test_hmm_package.py -v
```
