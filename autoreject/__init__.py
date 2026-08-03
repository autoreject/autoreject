"""Automated rejection and repair of epochs in M/EEG."""

try:
    from importlib.metadata import version

    __version__ = version("autoreject")
except Exception:
    __version__ = "0.0.0"

from .autoreject import _GlobalAutoReject, _AutoReject, AutoReject
from .autoreject import RejectLog, read_auto_reject, read_reject_log
from .autoreject import compute_thresholds, validation_curve, get_rejection_threshold
from .ransac import Ransac
from .utils import set_matplotlib_defaults
from .backends import (
    detect_hardware, get_backend, get_backend_names,
    DeviceArray, is_device_array
)

# GPU interpolation functions (optional - only available with torch backend)
try:
    from .gpu_interpolation import (
        gpu_make_interpolation_matrix,
        gpu_do_interp_dots,
        gpu_interpolate_bads_eeg,
        gpu_clean_by_interp,
    )
except ImportError:
    pass  # torch not installed
