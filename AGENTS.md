# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

autoreject finds and repairs or rejects bad epochs in M/EEG data. `AutoReject` learns
per-channel peak-to-peak thresholds by cross-validation, then for each epoch either
interpolates the bad channels or drops the epoch, recording the decisions in a
`RejectLog`. `get_rejection_threshold` estimates a single global `reject` dict for
`mne.Epochs`, and `Ransac` detects bad channels. MNE-Python owns the data containers
(`Epochs`, `Info`) and the interpolation machinery; this package decides what to
interpolate or drop.

## Follow MNE-Python's conventions

This package is a subsidiary of MNE-Python. Unless something below says otherwise, follow
[MNE-Python's AGENTS.md](https://github.com/mne-tools/mne-python/blob/main/AGENTS.md)
(read it rather than guessing): keep changes small, naming, numpydoc style, imports,
deprecation policy, compact tests, license rules for adapted code, and in particular its
[policy on AI assistance](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md#policy-on-ai-assistance-in-contributions):

- Work test-first: write (or extend) a test that fails for the right reason, then make it
  pass. Promote anything a throwaway script caught into a real test.
- Do not open pull requests, push, or commit unless explicitly asked; the human submitting
  the change must review, understand, and disclose AI use in the PR description.
- Keep changes minimal and scoped to the request; mention, don't silently fix, unrelated
  problems you notice.
- Do not paste agent-written text into issues, PRs, or comments.
- When presenting benchmarks or results for a given metric, include a minimal
  reproducible example that reviewers can run locally, where feasible. For example:

<details><summary>Minimally Working Example</summary>

```python
import timeit

import mne
import numpy as np
from autoreject import AutoReject

data_path = mne.datasets.testing.data_path()
raw = mne.io.read_raw_fif(
    data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif", verbose="error"
)
raw.pick("eeg", exclude="bads").load_data()
epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose="error")

times = timeit.repeat(
    lambda: AutoReject(cv=3, random_state=0, verbose=False).fit(epochs),
    repeat=5,
    number=1,
)
print(f"{np.mean(times):.3f} ± {np.std(times):.3f} s per fit")
```
</details>

What does *not* carry over from MNE-Python:

- There are no towncrier fragments. The changelog is `doc/whats_new.rst`: add a bullet
  under the right heading (Changelog or Bug) of the unreleased version, in the
  existing format (``- Description, by `Name`_ in :github:`#123` ``, no trailing
  period), and add first-time contributors to the name links at the bottom of that file.
- There are no lazy `__init__.pyi` stubs: `autoreject/__init__.py` imports its public
  names directly, and new public API must also be added to `doc/api.rst`.
- There is no `docdict` and no `@verbose` decorator. `verbose` is a plain parameter
  (mostly a bool, with `'tqdm'`, `'tqdm_notebook'`, and `'progressbar'` accepted where
  progress bars are shown) that is passed down explicitly; document it in each
  docstring like the existing ones.
- Style is checked with flake8 only (`flake8 autoreject`, configured in
  `pyproject.toml`, max line length 88). There is no ruff, ruff-format, or pre-commit,
  so match the formatting of the surrounding code by hand rather than reformatting it.
- The default branch is `main`, and there is no PR template.

## Things to know

- Results matter, not just speed: thresholds, `n_interpolate_`, `consensus_`, and the
  `RejectLog` labels are what users rely on. A change that alters them for a fixed
  `random_state` must be called out as such in `doc/whats_new.rst`; a refactor or
  speedup should be shown to leave them unchanged.
- `AutoReject` is a scikit-learn `BaseEstimator` whose fitted attributes end in `_`.
  Fitted objects round-trip through `save`/`read_auto_reject` (HDF5 via `h5io`) using
  the `_INIT_PARAMS` and `_FIT_PARAMS` tuples in `autoreject/autoreject.py`, so a new
  constructor argument or fitted attribute needs to be added there and covered by the
  I/O tests.
- Threshold estimation is random (cross-validation splits, Bayesian optimization).
  Tests should pass an explicit `random_state`, because with `random_state=None` the
  number of rejected epochs varies from run to run and across platforms.
- `autoreject/utils.py` and `autoreject/ransac.py` rely on private MNE-Python helpers
  (`_make_interpolation_matrix`, `_do_interp_dots`, `_setup_dots`, `_picks_to_idx`,
  `_fit_sphere`, ...), so they can break when MNE-Python's `main` changes. Supported
  MNE-Python versions start at the minimum in `pyproject.toml`, and CI tests both the
  current stable release and `main`. Code that needs a newer MNE-Python gates on
  `mne.utils.check_version`, or on the feature itself (e.g., a parameter in
  `inspect.signature`) when it must also work with unreleased MNE-Python.
- matplotlib (and the MNE plotting it calls) is imported inside the functions that use
  it, not at module level.

## Tests and docs

`pytest autoreject` runs the unit tests. Most tests need the MNE testing dataset
(`python -c 'import mne; mne.datasets.testing.data_path(verbose=True)'`) and are
decorated with `@testing.requires_testing_data`; new tests that use it must be decorated
too. Warnings are errors (`filterwarnings` lines added in `autoreject/conftest.py`):
catch warnings a test expects with `pytest.warns`, and add a narrowly scoped `ignore`
line there only for third-party noise. CI runs the suite on Linux, macOS, and Windows
against MNE-Python stable and `main`, plus a flake8 job.

The gallery examples in `examples/` are not unit tests: they run when the docs (Sphinx +
sphinx-gallery) are built on CircleCI with `make -C doc html`, and some download data
from OpenNeuro. Run `flake8 autoreject` before handing work back.
