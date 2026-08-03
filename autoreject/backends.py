"""Backend abstraction for compute operations.

This module provides a unified interface for array operations across different
compute backends (NumPy, PyTorch). It enables:

- Automatic hardware detection (CPU, CUDA, MPS)
- Graceful fallback when optional dependencies are not installed
- User-configurable backend selection via environment variable
- **Data-on-GPU architecture** for avoiding CPU↔GPU transfer overhead

Usage
-----
>>> from autoreject.backends import get_backend, detect_hardware
>>>
>>> # Auto-detect best available backend
>>> backend = get_backend()
>>> result = backend.ptp(data, axis=-1)
>>>
>>> # Keep data on GPU (avoid transfers)
>>> gpu_data = backend.to_device(data)
>>> result = backend.ptp(gpu_data, axis=-1, keep_on_device=True)
>>> # ... more operations on result ...
>>> final = backend.to_numpy(result)  # Transfer only at the end
>>>
>>> # Force a specific backend
>>> backend = get_backend(prefer='torch')
>>>
>>> # Check available hardware
>>> hw = detect_hardware()
>>> print(hw)  # {'cpu': True, 'cuda': False, 'mps': True, ...}

Environment Variables
---------------------
AUTOREJECT_BACKEND : str
    Override automatic backend selection. Valid values:
    'numpy', 'torch'
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import os
import warnings
from contextlib import contextmanager
from functools import lru_cache

import numpy as np


__all__ = [
    "detect_hardware",
    "get_backend",
    "get_backend_names",
    "DeviceArray",
    "is_device_array",
    "force_cpu_backend",
    "use_backend",
]


# =================================================
# DeviceArray - Wrapper for GPU-resident arrays
# =================================================


class DeviceArray:
    """Wrapper for arrays that may reside on CPU or GPU.

    This class provides a unified interface for tracking array location
    and enables the data-on-GPU architecture where arrays stay on the
    accelerator throughout a computation pipeline.

    Parameters
    ----------
    data : array-like
        The underlying array (numpy or torch.Tensor)
    backend : BaseBackend
        The backend that created this array.
    device : str
        Device location ('cpu', 'cuda', 'mps', etc.)

    Attributes
    ----------
    data : array-like
        The underlying array.
    backend : BaseBackend
        The backend managing this array.
    device : str
        Current device location.
    shape : tuple
        Array shape.
    dtype : dtype
        Array data type.

    Examples
    --------
    >>> backend = get_backend(prefer='torch')
    >>> gpu_arr = backend.to_device(numpy_data)
    >>> isinstance(gpu_arr, DeviceArray)
    True
    >>> gpu_arr.device
    'mps'
    >>> result = backend.ptp(gpu_arr, keep_on_device=True)
    >>> numpy_result = backend.to_numpy(result)
    """

    def __init__(self, data, backend, device):
        self._data = data
        self._backend = backend
        self._device = device

    @property
    def data(self):
        """Return the underlying array."""
        return self._data

    @property
    def backend(self):
        """Return the backend managing this array."""
        return self._backend

    @property
    def device(self):
        """Return the device location."""
        return self._device

    @property
    def shape(self):
        """Return the array shape."""
        return self._data.shape

    @property
    def dtype(self):
        """Return the array dtype."""
        if hasattr(self._data, "dtype"):
            return self._data.dtype
        return type(self._data)

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return len(self.shape)

    def numpy(self):
        """Convert to NumPy array (transfers from GPU if needed)."""
        return self._backend.to_numpy(self._data)

    def __repr__(self):
        return (
            f"DeviceArray(shape={self.shape}, dtype={self.dtype}, "
            f"device='{self._device}', backend='{self._backend.name}')"
        )

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        """Support indexing (returns DeviceArray if on GPU)."""
        result = self._data[idx]
        if self._device != "cpu":
            return DeviceArray(result, self._backend, self._device)
        return result


def is_device_array(arr):
    """Check if an array is a DeviceArray.

    Parameters
    ----------
    arr : array-like
        Array to check.

    Returns
    -------
    bool
        True if arr is a DeviceArray.
    """
    return isinstance(arr, DeviceArray)


def _unwrap_device_array(arr):
    """Unwrap DeviceArray to get underlying data.

    Parameters
    ----------
    arr : array-like or DeviceArray
        Input array.

    Returns
    -------
    data : array-like
        The underlying array data.
    """
    if isinstance(arr, DeviceArray):
        return arr._data
    return arr


# ====================
# Hardware Detection
# ====================


@lru_cache(maxsize=1)
def detect_hardware():
    """Detect available hardware acceleration.

    Returns
    -------
    available : dict
        Dictionary with keys for each hardware type and boolean values
        indicating availability. Keys include:
        - 'cpu': Always True
        - 'cuda': True if NVIDIA GPU available via PyTorch
        - 'mps': True if Apple Silicon GPU available via PyTorch
        - 'cuda_device': Name of CUDA device (if available)
        - 'mps_device': 'Apple Silicon' (if available)

    Examples
    --------
    >>> hw = detect_hardware()
    >>> if hw.get('mps'):
    ...     print("Apple Silicon GPU available")
    """
    available = {"cpu": True}

    # Check for NVIDIA GPU via PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            available["cuda"] = True
            available["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except Exception:
        pass  # CUDA initialization errors

    # Check for Apple Silicon GPU via PyTorch MPS
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available["mps"] = True
            available["mps_device"] = "Apple Silicon"
    except ImportError:
        pass
    except Exception:
        pass

    return available


def get_backend_names():
    """Get list of available backend names.

    Returns
    -------
    names : list of str
        List of backend names that can be used with get_backend().
    """
    names = ["numpy"]  # Always available

    try:
        import torch  # noqa: F401

        names.append("torch")
    except ImportError:
        pass

    return names


# ===================
# Backend Selection
# ===================

_BACKEND_CACHE = {}


def get_backend(prefer=None):
    """Get the best available compute backend.

    Parameters
    ----------
    prefer : str | None
        Preferred backend: 'numpy', 'torch', or None (auto).
        Can also be set via AUTOREJECT_BACKEND environment variable.
        If the preferred backend is not available, falls back to the next
        best option.

    Returns
    -------
    backend : object
        A backend instance (NumpyBackend or TorchBackend) with methods
        for array operations.

    Notes
    -----
    Backend selection priority (when prefer=None):
    1. If CUDA GPU available: PyTorch > NumPy
    2. If MPS (Apple Silicon) available: PyTorch > NumPy
    3. CPU only: NumPy

    Examples
    --------
    >>> backend = get_backend()
    >>> print(f"Using {backend.name} on {backend.device}")

    >>> # Force NumPy backend
    >>> backend = get_backend(prefer='numpy')
    """
    # Check environment variable
    prefer = prefer or os.environ.get("AUTOREJECT_BACKEND", None)

    # Normalize preference
    if prefer is not None:
        prefer = prefer.lower().strip()

    # Check cache
    cache_key = prefer or "auto"
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]

    # Try to load preferred backend
    if prefer is not None:
        backend = _try_load_backend(prefer)
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend
        warnings.warn(
            f"Preferred backend '{prefer}' not available, "
            "falling back to auto-detection.",
            RuntimeWarning,
        )

    # Auto-detect best backend
    hw = detect_hardware()

    # Priority 1: GPU acceleration
    if hw.get("cuda"):
        backend = _try_load_backend("torch")
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend

    # Priority 2: Apple Silicon MPS
    if hw.get("mps"):
        backend = _try_load_backend("torch")
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend

    # Fallback: NumPy (always available)
    backend = NumpyBackend()
    _BACKEND_CACHE[cache_key] = backend
    return backend


@contextmanager
def force_cpu_backend():
    """Context manager to temporarily force the CPU (NumPy) backend.

    This is used for MPS fallback to CPU when float64 precision is required.
    The context manager saves the current backend cache state, forces NumPy,
    and restores the original state on exit.

    Examples
    --------
    >>> with force_cpu_backend():
    ...     backend = get_backend()
    ...     assert backend.name == 'numpy'
    >>> # Original backend is restored after the context
    """
    # Save current cache and environment
    saved_cache = _BACKEND_CACHE.copy()
    saved_env = os.environ.get("AUTOREJECT_BACKEND", None)

    try:
        # Force NumPy backend
        _BACKEND_CACHE.clear()
        os.environ["AUTOREJECT_BACKEND"] = "numpy"
        yield
    finally:
        # Restore original state
        _BACKEND_CACHE.clear()
        _BACKEND_CACHE.update(saved_cache)
        if saved_env is not None:
            os.environ["AUTOREJECT_BACKEND"] = saved_env
        elif "AUTOREJECT_BACKEND" in os.environ:
            del os.environ["AUTOREJECT_BACKEND"]


@contextmanager
def use_backend(backend_name):
    """Context manager to temporarily use a specific backend.

    This provides an MNE-style interface for backend selection, similar to
    MNE's ``use_3d_backend()``.

    Parameters
    ----------
    backend_name : str
        Backend to use: 'numpy' or 'torch'.

    Examples
    --------
    >>> from autoreject.backends import use_backend
    >>> with use_backend('torch'):
    ...     # GPU-accelerated operations
    ...     ar = AutoReject()
    ...     ar.fit(epochs)
    >>> # Back to default backend

    See Also
    --------
    force_cpu_backend : Force CPU (NumPy) backend temporarily.
    get_backend : Get the current backend.
    """
    # Save current cache and environment
    saved_cache = _BACKEND_CACHE.copy()
    saved_env = os.environ.get("AUTOREJECT_BACKEND", None)

    try:
        # Set requested backend
        _BACKEND_CACHE.clear()
        os.environ["AUTOREJECT_BACKEND"] = backend_name
        yield
    finally:
        # Restore original state
        _BACKEND_CACHE.clear()
        _BACKEND_CACHE.update(saved_cache)
        if saved_env is not None:
            os.environ["AUTOREJECT_BACKEND"] = saved_env
        elif "AUTOREJECT_BACKEND" in os.environ:
            del os.environ["AUTOREJECT_BACKEND"]


def _try_load_backend(name):
    """Try to load a specific backend.

    Parameters
    ----------
    name : str
        Backend name: 'numpy' or 'torch'.

    Returns
    -------
    backend : Backend | None
        Backend instance if successful, None otherwise.
    """
    try:
        if name == "numpy":
            return NumpyBackend()
        elif name == "torch":
            return TorchBackend()
        else:
            warnings.warn(f"Unknown backend: {name}", RuntimeWarning)
            return None
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"Failed to initialize {name} backend: {e}", RuntimeWarning)
        return None


def clear_backend_cache():
    """Clear the backend cache.

    Useful for testing or when hardware configuration changes.
    """
    _BACKEND_CACHE.clear()
    detect_hardware.cache_clear()


# ====================
# Base Backend Class
# ====================


class BaseBackend:
    """Abstract base class for compute backends.

    All backends must implement these methods with identical signatures
    to ensure interchangeability.

    Attributes
    ----------
    name : str
        Backend name ('numpy', 'torch').
    device : str
        Device description ('cpu', 'cuda:0', 'mps', etc.).
    supports_gpu : bool
        Whether this backend supports GPU acceleration.
    """

    name = "base"
    device = "cpu"
    supports_gpu = False

    def to_device(self, data):
        """Transfer data to the backend's device.

        For GPU backends, this transfers data to GPU memory.
        For CPU backends, this is a no-op that returns a DeviceArray wrapper.

        Parameters
        ----------
        data : array-like or DeviceArray
            Input array.

        Returns
        -------
        result : DeviceArray
            Array wrapped in DeviceArray, potentially on GPU.

        Examples
        --------
        >>> backend = get_backend(prefer='torch')
        >>> gpu_data = backend.to_device(numpy_array)
        >>> gpu_data.device
        'mps'
        """
        raise NotImplementedError

    def ptp(self, data, axis=-1, keep_on_device=False):
        """Compute peak-to-peak (max - min) along an axis.

        Parameters
        ----------
        data : array-like or DeviceArray
            Input array.
        axis : int
            Axis along which to compute ptp. Default: -1.
        keep_on_device : bool
            If True, return DeviceArray staying on device.
            If False (default), return NumPy array.

        Returns
        -------
        result : ndarray or DeviceArray
            Peak-to-peak values.
        """
        raise NotImplementedError

    def median(self, data, axis=None, keep_on_device=False):
        """Compute median along an axis.

        Parameters
        ----------
        data : array-like or DeviceArray
            Input array.
        axis : int | None
            Axis along which to compute median. Default: None (all elements).
        keep_on_device : bool
            If True, return DeviceArray staying on device.
            If False (default), return NumPy array.

        Returns
        -------
        result : ndarray or DeviceArray or scalar
            Median values.
        """
        raise NotImplementedError

    def correlation(self, x, y, keep_on_device=False):
        """Compute correlation between two arrays.

        Parameters
        ----------
        x : array-like or DeviceArray, shape (n_times, n_channels)
            First array.
        y : array-like or DeviceArray, shape (n_times, n_channels)
            Second array.
        keep_on_device : bool
            If True, return DeviceArray staying on device.
            If False (default), return NumPy array.

        Returns
        -------
        corr : ndarray or DeviceArray, shape (n_channels,)
            Correlation coefficients.
        """
        raise NotImplementedError

    def matmul(self, a, b, keep_on_device=False):
        """Matrix multiplication.

        Parameters
        ----------
        a : array-like or DeviceArray
            First matrix.
        b : array-like or DeviceArray
            Second matrix.
        keep_on_device : bool
            If True, return DeviceArray staying on device.
            If False (default), return NumPy array.

        Returns
        -------
        result : ndarray or DeviceArray
            Matrix product.
        """
        raise NotImplementedError

    def to_numpy(self, arr):
        """Convert array to NumPy ndarray.

        Parameters
        ----------
        arr : array-like or DeviceArray
            Input array (may be on GPU or in different format).

        Returns
        -------
        result : ndarray
            NumPy array on CPU.
        """
        raise NotImplementedError

    def is_on_device(self, arr):
        """Check if array is on this backend's device.

        Parameters
        ----------
        arr : array-like or DeviceArray
            Array to check.

        Returns
        -------
        bool
            True if array is on device (GPU for GPU backends).
        """
        if isinstance(arr, DeviceArray):
            return arr.device == self.device
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"


# ============================
# NumPy Backend (Baseline)
# ============================


class NumpyBackend(BaseBackend):
    """NumPy backend (baseline, always available).

    This is the reference implementation. All operations use standard
    NumPy functions without parallelization.
    """

    name = "numpy"
    device = "cpu"
    supports_gpu = False

    def to_device(self, data):
        """Wrap data in DeviceArray (no transfer, already on CPU)."""
        data = _unwrap_device_array(data)
        arr = np.asarray(data)
        return DeviceArray(arr, self, "cpu")

    def ptp(self, data, axis=-1, keep_on_device=False):
        """Compute peak-to-peak using np.ptp."""
        data = _unwrap_device_array(data)
        result = np.ptp(data, axis=axis)
        if keep_on_device:
            return DeviceArray(result, self, "cpu")
        return result

    def median(self, data, axis=None, keep_on_device=False):
        """Compute median using np.median."""
        data = _unwrap_device_array(data)
        result = np.median(data, axis=axis)
        if keep_on_device:
            return DeviceArray(result, self, "cpu")
        return result

    def correlation(self, x, y, keep_on_device=False):
        """Compute correlation between arrays.

        Uses the formula: corr = sum(x*y) / (||x|| * ||y||)
        """
        x = _unwrap_device_array(x)
        y = _unwrap_device_array(y)
        num = np.sum(x * y, axis=0)
        denom = np.sqrt(np.sum(x**2, axis=0)) * np.sqrt(np.sum(y**2, axis=0))
        result = num / denom
        if keep_on_device:
            return DeviceArray(result, self, "cpu")
        return result

    def matmul(self, a, b, keep_on_device=False):
        """Matrix multiplication using np.matmul."""
        a = _unwrap_device_array(a)
        b = _unwrap_device_array(b)
        result = np.matmul(a, b)
        if keep_on_device:
            return DeviceArray(result, self, "cpu")
        return result

    def to_numpy(self, arr):
        """Return array as-is (already NumPy)."""
        arr = _unwrap_device_array(arr)
        return np.asarray(arr)


# ==============================
# PyTorch Backend (CUDA/MPS)
# ==============================


class TorchBackend(BaseBackend):
    """PyTorch backend with CUDA/MPS GPU support.

    Automatically selects the best available device:
    - CUDA (NVIDIA GPUs)
    - MPS (Apple Silicon)
    - CPU (fallback)

    Notes
    -----
    Use `to_device()` to transfer data to GPU once, then use `keep_on_device=True`
    in operations to avoid CPU↔GPU transfers. Only call `to_numpy()` at the end.

    MPS (Apple Silicon) only supports float32, so data is automatically
    converted when using MPS device.

    Examples
    --------
    >>> backend = get_backend(prefer='torch')
    >>> # Transfer data to GPU once
    >>> gpu_data = backend.to_device(epochs_data)
    >>> # All operations stay on GPU
    >>> ptp_result = backend.ptp(gpu_data, keep_on_device=True)
    >>> median_result = backend.median(gpu_data, axis=0, keep_on_device=True)
    >>> # Transfer back only at the end
    >>> final = backend.to_numpy(median_result)
    """

    name = "torch"
    supports_gpu = True

    def __init__(self):
        """Initialize PyTorch backend."""
        import torch

        self._torch = torch

        # Select device
        if torch.cuda.is_available():
            self.device = "cuda"
            self._device = torch.device("cuda")
            self._dtype = torch.float64  # CUDA supports float64
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self._device = torch.device("mps")
            self._dtype = torch.float32  # MPS only supports float32
        else:
            self.device = "cpu"
            self._device = torch.device("cpu")
            self._dtype = torch.float64  # CPU supports float64

    def _to_tensor(self, data):
        """Convert data to PyTorch tensor on device.

        Handles DeviceArray, numpy arrays, and existing tensors.
        """
        # Unwrap DeviceArray
        if isinstance(data, DeviceArray):
            data = data._data

        # Already a tensor on the right device?
        if isinstance(data, self._torch.Tensor):
            if data.device == self._device and data.dtype == self._dtype:
                return data
            return data.to(device=self._device, dtype=self._dtype)

        # Convert from numpy
        arr = np.asarray(data)
        # Convert to appropriate dtype for device
        if self._dtype == self._torch.float32:
            arr = arr.astype(np.float32)
        return self._torch.from_numpy(arr).to(self._device)

    def _maybe_wrap(self, tensor, keep_on_device):
        """Wrap result in DeviceArray or convert to numpy."""
        if keep_on_device:
            return DeviceArray(tensor, self, self.device)
        return tensor.cpu().numpy()

    def to_device(self, data):
        """Transfer data to GPU.

        Parameters
        ----------
        data : array-like or DeviceArray
            Input data.

        Returns
        -------
        DeviceArray
            Data on GPU wrapped in DeviceArray.
        """
        tensor = self._to_tensor(data)
        return DeviceArray(tensor, self, self.device)

    def ptp(self, data, axis=-1, keep_on_device=False):
        """Compute peak-to-peak using PyTorch."""
        t = self._to_tensor(data)
        result = t.max(dim=axis).values - t.min(dim=axis).values
        return self._maybe_wrap(result, keep_on_device)

    def median(self, data, axis=None, keep_on_device=False):
        """Compute median using PyTorch.

        Note: torch.median behaves differently from numpy.median for arrays
        with an even number of elements:
        - numpy returns the average of the two middle values
        - torch returns the lower of the two middle values

        This implementation matches numpy.median behavior for consistency
        by using torch.sort and computing the average of the two middle values
        for even-length arrays. This approach:
        - Stays entirely on GPU (no CPU fallback needed)
        - Has the same performance as torch.median
        - Is numerically identical to numpy.median
        """
        t = self._to_tensor(data)

        if axis is None:
            # Flatten and compute median
            t_flat = t.flatten()
            n = t_flat.shape[0]
            sorted_t, _ = self._torch.sort(t_flat)

            if n % 2 == 1:
                # Odd: return middle element
                result = sorted_t[n // 2]
            else:
                # Even: return average of two middle elements
                result = (sorted_t[n // 2 - 1] + sorted_t[n // 2]) / 2
        else:
            # Sort along the specified axis
            n = t.shape[axis]
            sorted_t, _ = self._torch.sort(t, dim=axis)

            if n % 2 == 1:
                # Odd: return middle element
                idx = n // 2
                result = self._torch.select(sorted_t, axis, idx)
            else:
                # Even: return average of two middle elements
                idx1, idx2 = n // 2 - 1, n // 2
                v1 = self._torch.select(sorted_t, axis, idx1)
                v2 = self._torch.select(sorted_t, axis, idx2)
                result = (v1 + v2) / 2

        return self._maybe_wrap(result, keep_on_device)

    def correlation(self, x, y, keep_on_device=False):
        """Compute correlation using PyTorch."""
        tx = self._to_tensor(x)
        ty = self._to_tensor(y)

        num = (tx * ty).sum(dim=0)
        denom = tx.pow(2).sum(dim=0).sqrt() * ty.pow(2).sum(dim=0).sqrt()
        result = num / denom

        return self._maybe_wrap(result, keep_on_device)

    def matmul(self, a, b, keep_on_device=False):
        """Matrix multiplication using PyTorch."""
        ta = self._to_tensor(a)
        tb = self._to_tensor(b)
        result = self._torch.matmul(ta, tb)
        return self._maybe_wrap(result, keep_on_device)

    def to_numpy(self, arr):
        """Convert tensor to NumPy array."""
        # Unwrap DeviceArray
        if isinstance(arr, DeviceArray):
            arr = arr._data

        if isinstance(arr, self._torch.Tensor):
            return arr.cpu().numpy()
        return np.asarray(arr)

    def is_on_device(self, arr):
        """Check if array is on this backend's GPU."""
        if isinstance(arr, DeviceArray):
            return arr.device == self.device
        if isinstance(arr, self._torch.Tensor):
            return arr.device.type == self._device.type
        return False

    # =========================================================================
    # Extended GPU operations for data-on-GPU architecture
    # =========================================================================

    def zeros(self, shape, keep_on_device=True):
        """Create zeros array on device."""
        result = self._torch.zeros(shape, dtype=self._dtype, device=self._device)
        return self._maybe_wrap(result, keep_on_device)

    def ones(self, shape, keep_on_device=True):
        """Create ones array on device."""
        result = self._torch.ones(shape, dtype=self._dtype, device=self._device)
        return self._maybe_wrap(result, keep_on_device)

    def empty(self, shape, keep_on_device=True):
        """Create uninitialized array on device."""
        result = self._torch.empty(shape, dtype=self._dtype, device=self._device)
        return self._maybe_wrap(result, keep_on_device)

    def sqrt(self, data, keep_on_device=False):
        """Compute square root."""
        t = self._to_tensor(data)
        result = self._torch.sqrt(t)
        return self._maybe_wrap(result, keep_on_device)

    def sum(self, data, axis=None, keep_on_device=False):
        """Compute sum along axis."""
        t = self._to_tensor(data)
        if axis is None:
            result = t.sum()
        else:
            result = t.sum(dim=axis)
        return self._maybe_wrap(result, keep_on_device)

    def mean(self, data, axis=None, keep_on_device=False):
        """Compute mean along axis."""
        t = self._to_tensor(data)
        if axis is None:
            result = t.mean()
        else:
            result = t.mean(dim=axis)
        return self._maybe_wrap(result, keep_on_device)

    def max(self, data, axis=None, keep_on_device=False):
        """Compute max along axis."""
        t = self._to_tensor(data)
        if axis is None:
            result = t.max()
        else:
            result = t.max(dim=axis).values
        return self._maybe_wrap(result, keep_on_device)

    def min(self, data, axis=None, keep_on_device=False):
        """Compute min along axis."""
        t = self._to_tensor(data)
        if axis is None:
            result = t.min()
        else:
            result = t.min(dim=axis).values
        return self._maybe_wrap(result, keep_on_device)

    def abs(self, data, keep_on_device=False):
        """Compute absolute value."""
        t = self._to_tensor(data)
        result = self._torch.abs(t)
        return self._maybe_wrap(result, keep_on_device)

    def where(self, condition, x, y, keep_on_device=False):
        """Element-wise selection based on condition."""
        cond = self._to_tensor(condition)
        tx = self._to_tensor(x)
        ty = self._to_tensor(y)
        result = self._torch.where(cond.bool(), tx, ty)
        return self._maybe_wrap(result, keep_on_device)

    def concatenate(self, arrays, axis=0, keep_on_device=False):
        """Concatenate arrays along axis."""
        tensors = [self._to_tensor(a) for a in arrays]
        result = self._torch.cat(tensors, dim=axis)
        return self._maybe_wrap(result, keep_on_device)

    def stack(self, arrays, axis=0, keep_on_device=False):
        """Stack arrays along new axis."""
        tensors = [self._to_tensor(a) for a in arrays]
        result = self._torch.stack(tensors, dim=axis)
        return self._maybe_wrap(result, keep_on_device)

    def argsort(self, data, axis=-1, descending=False, keep_on_device=False):
        """Return indices that would sort the array."""
        t = self._to_tensor(data)
        result = t.argsort(dim=axis, descending=descending)
        return self._maybe_wrap(result, keep_on_device)

    def copy(self, data, keep_on_device=True):
        """Create a copy of the array on device."""
        t = self._to_tensor(data)
        result = t.clone()
        return self._maybe_wrap(result, keep_on_device)
