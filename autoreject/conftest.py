# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import gc
import os
import warnings
import pytest

import mne
from mne.viz import use_browser_backend


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    config.addinivalue_line('markers', 'pgtest')

    # Fixtures
    for fixture in ('matplotlib_config', 'close_all', 'check_verbose'):
        config.addinivalue_line('usefixtures', fixture)

    # Warnings as errors
    warning_lines = r"""
    error::
    ignore:Matplotlib is currently using agg.*:UserWarning
    ignore:`np.MachAr` is deprecated.*:DeprecationWarning
    """
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


# Have to be careful with autouse=True, but this is just an int comparison
# so it shouldn't really add appreciable overhead
@pytest.fixture(autouse=True)
def check_verbose(request):
    """Set to the default logging level to ensure it's tested properly."""
    starting_level = mne.utils.logger.level
    yield
    # ensures that no tests break the global state
    try:
        assert mne.utils.logger.level == starting_level
    except AssertionError:  # pragma: no cover
        pytest.fail('.'.join([request.module.__name__,
                              request.function.__name__]) +
                    ' modifies logger.level')


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing, and we have ~2500 tests, so ~2 ms max
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    from matplotlib import cbook
    # Allow for easy interactive debugging with a call like:
    #
    #     $ MNE_MPL_TESTING_BACKEND=Qt5Agg pytest mne/viz/tests/test_raw.py -k annotation -x --pdb  # noqa: E501
    #
    try:
        want = os.environ['MNE_MPL_TESTING_BACKEND']
    except KeyError:
        want = 'agg'  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings('ignore')
        matplotlib.use(want, force=True)
    import matplotlib.pyplot as plt
    assert plt.get_backend() == want
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.raise_window'] = False

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None, signals=None):
            super(CallbackRegistryReraise, self).__init__(exception_handler)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture
def garbage_collect():
    """Garbage collect on exit."""
    yield
    gc.collect()


@pytest.fixture(params=[
    'matplotlib',
    pytest.param('qt', marks=pytest.mark.pgtest),
])
def browser_backend(request, garbage_collect, monkeypatch):
    """Parametrizes the name of the browser backend."""
    backend_name = request.param
    if backend_name == 'qt':
        pytest.importorskip('mne_qt_browser')
    with use_browser_backend(backend_name) as backend:
        backend._close_all()
        monkeypatch.setenv('MNE_BROWSE_RAW_SIZE', '10,10')
        yield backend
        backend._close_all()
        if backend_name == 'qt':
            # This shouldn't be necessary, but let's make sure nothing is stale
            import mne_qt_browser
            mne_qt_browser._browser_instances.clear()
