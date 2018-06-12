__version__ = '0.2.dev0'

from .autoreject import _GlobalAutoReject, _AutoReject, AutoReject
from .autoreject import RejectLog
from .autoreject import compute_thresholds, validation_curve, get_rejection_threshold
from .ransac import Ransac
from .utils import set_matplotlib_defaults
