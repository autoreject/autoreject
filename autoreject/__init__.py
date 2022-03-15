"""Automated rejection and repair of epochs in M/EEG."""
__version__ = '0.3.1'

from .autoreject import _GlobalAutoReject, _AutoReject, AutoReject
from .autoreject import RejectLog, read_auto_reject
from .autoreject import compute_thresholds, validation_curve, get_rejection_threshold
from .ransac import Ransac
from .utils import set_matplotlib_defaults
