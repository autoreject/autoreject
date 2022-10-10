# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


def pytest_configure(config):
    """Configure pytest options."""
    # Fixtures
    for fixture in ('matplotlib_config', 'close_all'):
        config.addinivalue_line('usefixtures', fixture)
