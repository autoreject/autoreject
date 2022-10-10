# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


pytest_plugins = ['mne.conftest']


def pytest_configure(config):
    # Fixtures
    for fixture in ('matplotlib_config', 'close_all', 'check_verbose',
                    'qt_config', 'protect_config'):
        config.addinivalue_line('usefixtures', fixture)
