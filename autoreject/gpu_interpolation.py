"""GPU-accelerated spherical spline interpolation for EEG channels.

This module provides PyTorch implementations of MNE's spherical spline
interpolation functions, enabling the entire interpolation pipeline to
run on GPU without CPU<->GPU transfers.

The implementation is based on:
    Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
    Spherical splines for scalp potential and current density mapping.
    Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import numpy as np
from .backends import get_backend, is_device_array, DeviceArray

__all__ = [
    "gpu_make_interpolation_matrix",
    "gpu_do_interp_dots",
    "gpu_interpolate_bads_eeg",
    "gpu_interpolate_bad_epochs",
    "gpu_batch_interpolate_all_n_interp",
    "legval_torch",
    "is_cuda_device",
]


def is_cuda_device(device):
    """Check if device is CUDA (not MPS or CPU).

    This is used to determine compute strategy:
    - CUDA: float64 on device for all operations
    - MPS: float32 on device (variance test showed this is acceptable)
    - CPU: float64 everywhere

    Parameters
    ----------
    device : str or torch.device
        Device to check.

    Returns
    -------
    bool
        True if device is CUDA.
    """
    import torch

    if device is None:
        return False

    if isinstance(device, torch.device):
        return device.type == "cuda"

    device_str = str(device).lower()
    return device_str.startswith("cuda")


def legval_torch(x, c):
    """Evaluate Legendre polynomial using Clenshaw's algorithm.

    PyTorch implementation of numpy.polynomial.legendre.legval.
    Follows the exact same algorithm as NumPy.

    Parameters
    ----------
    x : torch.Tensor
        Evaluation points.
    c : list or torch.Tensor
        Legendre coefficients ordered from low to high degree.

    Returns
    -------
    torch.Tensor
        Polynomial evaluation at x.
    """
    import torch

    # Ensure x is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Convert coefficients to list for indexing
    if isinstance(c, torch.Tensor):
        c_list = c.tolist()
    else:
        c_list = list(c)

    n = len(c_list)

    if n == 1:
        return torch.full_like(x, c_list[0])
    elif n == 2:
        return c_list[0] + c_list[1] * x
    else:
        # Clenshaw recurrence - match NumPy exactly
        # nd starts as len(c) and decrements BEFORE each use
        nd = n
        c0 = torch.full_like(x, c_list[-2])
        c1 = torch.full_like(x, c_list[-1])

        for i in range(3, n + 1):
            tmp = c0.clone()
            nd = nd - 1  # Decrement BEFORE use (this is key!)
            # Clenshaw recurrence for Legendre
            c0 = c_list[-i] - c1 * ((nd - 1) / nd)
            c1 = tmp + c1 * x * ((2 * nd - 1) / nd)

        return c0 + c1 * x


def _calc_g_torch(cosang, stiffness=4, n_legendre_terms=50):
    """Calculate spherical spline G function on GPU.

    PyTorch implementation of MNE's _calc_g.

    Parameters
    ----------
    cosang : torch.Tensor, shape (n, m)
        Cosine of angles between pairs of points on a spherical surface.
    stiffness : float
        Stiffness of the spline.
    n_legendre_terms : int
        Number of Legendre terms to evaluate.

    Returns
    -------
    torch.Tensor, shape (n, m)
        The G matrix.
    """

    # Compute Legendre coefficients
    # factors[n] = (2n+1) / (n^stiffness * (n+1)^stiffness * 4*pi)
    factors = [0.0]  # c[0] = 0
    for n in range(1, n_legendre_terms + 1):
        factor = (2 * n + 1) / (n**stiffness * (n + 1) ** stiffness * 4 * np.pi)
        factors.append(factor)

    return legval_torch(cosang, factors)


def _normalize_vectors_torch(pos):
    """Normalize position vectors to unit sphere.

    Parameters
    ----------
    pos : torch.Tensor, shape (n, 3)
        Position vectors.

    Returns
    -------
    torch.Tensor, shape (n, 3)
        Normalized position vectors.
    """
    import torch

    norms = torch.norm(pos, dim=1, keepdim=True)
    return pos / norms


def gpu_make_interpolation_matrix(pos_from, pos_to, alpha=1e-5, device=None):
    """Compute interpolation matrix based on spherical splines on GPU.

    PyTorch implementation of MNE's _make_interpolation_matrix.
    Uses float64 internally for numerical stability (like MNE), then
    converts to float32 for GPU compute efficiency.

    Parameters
    ----------
    pos_from : np.ndarray or torch.Tensor, shape (n_good, 3)
        The positions to interpolate from (good sensors).
    pos_to : np.ndarray or torch.Tensor, shape (n_bad, 3)
        The positions to interpolate to (bad sensors).
    alpha : float
        Regularization parameter. Defaults to 1e-5.
    device : str or torch.device, optional
        Device to run on. If None, uses 'mps' if available, else 'cpu'.

    Returns
    -------
    DeviceArray
        The interpolation matrix that maps good signals to bad signal locations.
        Shape: (n_bad, n_good)
    """
    import torch

    backend = get_backend()
    if backend.name != "torch":
        raise RuntimeError("gpu_make_interpolation_matrix requires torch backend")

    if device is None:
        device = backend.device

    # Determine compute strategy based on device type
    # CUDA: float64 on device for all operations
    # MPS: float32 on device (MPS only supports float32,
    #      variance test showed this is acceptable)
    # CPU: float64 everywhere
    use_cuda = is_cuda_device(device)
    is_mps = str(device).lower() == "mps"

    if use_cuda:
        # CUDA: everything in float64 on device
        compute_device = device
        compute_dtype = torch.float64
        output_dtype = torch.float64
    elif is_mps:
        # MPS: compute in float64 on CPU (for pinv precision), output float32 on device
        compute_device = "cpu"
        compute_dtype = torch.float64
        output_dtype = torch.float32  # MPS only supports float32
    else:
        # CPU: float64 everywhere
        compute_device = "cpu"
        compute_dtype = torch.float64
        output_dtype = torch.float64

    # Convert to torch tensors (float64 for precision)
    if isinstance(pos_from, np.ndarray):
        pos_from = torch.tensor(pos_from, dtype=compute_dtype, device=compute_device)
    else:
        pos_from = pos_from.clone().to(device=compute_device, dtype=compute_dtype)

    if isinstance(pos_to, np.ndarray):
        pos_to = torch.tensor(pos_to, dtype=compute_dtype, device=compute_device)
    else:
        pos_to = pos_to.clone().to(device=compute_device, dtype=compute_dtype)

    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]

    # Normalize sensor positions to unit sphere
    pos_from = _normalize_vectors_torch(pos_from)
    pos_to = _normalize_vectors_torch(pos_to)

    # Cosine angles between source positions (dot product of unit vectors)
    cosang_from = pos_from @ pos_from.T  # (n_from, n_from)
    cosang_to_from = pos_to @ pos_from.T  # (n_to, n_from)

    # Compute G matrices
    G_from = _calc_g_torch(cosang_from)
    G_to_from = _calc_g_torch(cosang_to_from)

    # Add regularization
    if alpha is not None:
        G_from = G_from + alpha * torch.eye(
            n_from, device=compute_device, dtype=compute_dtype
        )

    # Build the C matrix and compute pseudo-inverse
    # C = [[G_from, ones], [ones.T, 0]]
    ones_col = torch.ones((n_from, 1), device=compute_device, dtype=compute_dtype)
    ones_row = torch.ones((1, n_from), device=compute_device, dtype=compute_dtype)
    zero = torch.zeros((1, 1), device=compute_device, dtype=compute_dtype)

    C = torch.cat(
        [torch.cat([G_from, ones_col], dim=1), torch.cat([ones_row, zero], dim=1)],
        dim=0,
    )

    # Pseudo-inverse (computed in float64 for numerical stability)
    C_inv = torch.linalg.pinv(C)

    # Compute interpolation matrix
    # interpolation = [G_to_from, ones] @ C_inv[:, :-1]
    ones_to = torch.ones((n_to, 1), device=compute_device, dtype=compute_dtype)
    interpolation = torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]

    assert interpolation.shape == (n_to, n_from)

    # Move to target device with appropriate dtype
    # CUDA: keep float64 on device
    # MPS: convert to float32 and move to device
    #      (variance test showed this is acceptable)
    # CPU: keep float64
    interpolation = interpolation.to(device=device, dtype=output_dtype)

    return DeviceArray(interpolation, backend, str(device))


def gpu_do_interp_dots(data, interpolation, goods_idx, bads_idx, keep_on_device=True):
    """Apply interpolation matrix to data on GPU.

    Uses batch matrix multiplication (bmm) for optimal GPU performance.

    Parameters
    ----------
    data : np.ndarray or DeviceArray, shape (..., n_channels, n_times)
        The data to interpolate.
    interpolation : DeviceArray or torch.Tensor, shape (n_bad, n_good)
        The interpolation matrix.
    goods_idx : np.ndarray of bool, shape (n_channels,)
        Boolean mask for good channels.
    bads_idx : np.ndarray of bool, shape (n_channels,)
        Boolean mask for bad channels.
    keep_on_device : bool
        If True, returns DeviceArray. If False, returns numpy array.

    Returns
    -------
    DeviceArray or np.ndarray
        Interpolated data with bad channels replaced.
    """
    import torch

    backend = get_backend()
    if backend.name != "torch":
        raise RuntimeError("gpu_do_interp_dots requires torch backend")

    device = backend.device

    # Get interpolation matrix tensor
    # Use hasattr check for robustness with module reloads
    if hasattr(interpolation, "data") and hasattr(interpolation, "_backend"):
        interp_tensor = interpolation.data
    elif is_device_array(interpolation):
        interp_tensor = interpolation.data
    else:
        interp_tensor = interpolation

    # Convert data to tensor if needed
    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    elif hasattr(data, "data") and hasattr(data, "_backend"):
        data_tensor = data.data.clone()
    elif is_device_array(data):
        data_tensor = data.data.clone()
    else:
        data_tensor = data.clone()

    # Convert indices to tensors
    goods_idx_t = torch.tensor(goods_idx, dtype=torch.bool, device=device)
    bads_idx_t = torch.tensor(bads_idx, dtype=torch.bool, device=device)

    # Get good channel data
    good_data = data_tensor[..., goods_idx_t, :]

    # Ensure interp_tensor is on the same device as data
    if interp_tensor.device != data_tensor.device:
        interp_tensor = interp_tensor.to(data_tensor.device)

    # Match dtype if needed
    if interp_tensor.dtype != good_data.dtype:
        interp_tensor = interp_tensor.to(good_data.dtype)

    # Matmul directly on device (GPU accelerated)
    if data_tensor.ndim == 2:
        # (n_channels, n_times) - single epoch/evoked
        interpolated = interp_tensor @ good_data
    elif data_tensor.ndim == 3:
        # (n_epochs, n_channels, n_times)
        # interp_tensor: (n_bad, n_good)
        # good_data: (n_epochs, n_good, n_times)
        n_epochs = good_data.shape[0]
        interp_tensor.shape[0]

        # Expand interp to (n_epochs, n_bad, n_good) for bmm
        interp_expanded = interp_tensor.unsqueeze(0).expand(n_epochs, -1, -1)
        # bmm: (n_epochs, n_bad, n_good) @ (n_epochs, n_good, n_times)
        #   -> (n_epochs, n_bad, n_times)
        interpolated = torch.bmm(interp_expanded, good_data)
    else:
        raise ValueError(f"Unsupported data dimensions: {data_tensor.ndim}")

    # Replace bad channel data
    data_tensor[..., bads_idx_t, :] = interpolated

    if keep_on_device:
        return DeviceArray(data_tensor, backend, str(device))
    else:
        return data_tensor.cpu().numpy()


def gpu_interpolate_bads_eeg(inst, picks=None, keep_on_device=True):
    """Interpolate bad EEG channels using GPU.

    GPU-accelerated version of MNE's _interpolate_bads_eeg.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    picks : array-like of int or None
        Channels to consider for interpolation.
    keep_on_device : bool
        If True, returns DeviceArray. If False, modifies inst in place.

    Returns
    -------
    DeviceArray or None
        If keep_on_device=True, returns interpolated data as DeviceArray.
        If False, modifies inst in place and returns None.
    """
    import torch
    from mne import pick_types
    from .utils import _handle_picks

    backend = get_backend()
    if backend.name != "torch":
        raise RuntimeError("gpu_interpolate_bads_eeg requires torch backend")

    device = backend.device

    if picks is None:
        picks = pick_types(inst.info, meg=False, eeg=True, exclude=[])
    else:
        picks = _handle_picks(inst.info, picks)

    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)
    bads_idx[picks] = [inst.ch_names[ch] in inst.info["bads"] for ch in picks]

    if len(picks) == 0 or bads_idx.sum() == 0:
        if keep_on_device:
            return DeviceArray(
                torch.tensor(inst._data, dtype=torch.float32, device=device),
                backend,
                str(device),
            )
        return None

    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = inst._get_channel_positions(picks)

    # Make sure only good EEG are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]
    pos_good = pos[goods_idx_pos]
    pos_bad = pos[bads_idx_pos]

    # Compute interpolation matrix on GPU
    interpolation = gpu_make_interpolation_matrix(pos_good, pos_bad, device=device)

    # Apply interpolation
    result = gpu_do_interp_dots(
        inst._data, interpolation, goods_idx, bads_idx, keep_on_device=keep_on_device
    )

    if keep_on_device:
        return result
    else:
        inst._data = result if isinstance(result, np.ndarray) else result.cpu().numpy()
        return None


# Global cache for LOOCV interpolation matrices
# Key: tuple of channel positions hash, Value: (interpolation_matrices, good_picks_list)
_LOOCV_INTERP_CACHE = {}


def _get_loocv_interp_matrices(
    pos, picks, device, compute_device, compute_dtype, data_dtype
):
    """Get or compute cached LOOCV interpolation matrices.

    For LOOCV (leave-one-out cross-validation), each channel is interpolated
    from all other channels. The interpolation matrices depend ONLY on channel
    geometry, not on the data. So we compute them once and cache them.

    Parameters
    ----------
    pos : np.ndarray, shape (n_picks, 3)
        Normalized channel positions.
    picks : np.ndarray
        Channel indices.
    device : torch.device
        Target device for data operations.
    compute_device : torch.device
        Device for matrix computation (CPU for MPS, device for CUDA).
    compute_dtype : torch.dtype
        Dtype for matrix computation (float64).
    data_dtype : torch.dtype
        Dtype for data operations.

    Returns
    -------
    interp_matrices : torch.Tensor, shape (n_picks, n_picks)
        Interpolation weight matrix. interp_matrices[i, j] = weight for
        interpolating channel i from channel j. Diagonal is 0.
    """
    import torch

    # Create cache key from positions (hash of flattened array)
    cache_key = (pos.tobytes(), str(device), str(data_dtype))

    if cache_key in _LOOCV_INTERP_CACHE:
        return _LOOCV_INTERP_CACHE[cache_key]

    n_picks = len(picks)

    # Normalize positions and compute full G matrix
    pos_t = torch.tensor(pos, dtype=compute_dtype, device=compute_device)
    norms = torch.norm(pos_t, dim=1, keepdim=True)
    pos_t = pos_t / norms

    cosang_all = pos_t @ pos_t.T
    G_all = _calc_g_torch(cosang_all)

    # Build full interpolation matrix (n_picks, n_picks)
    # interp_matrices[i, j] = weight for interpolating channel i from channel j
    interp_matrices = torch.zeros((n_picks, n_picks), dtype=data_dtype, device=device)

    # For each channel being interpolated
    for bad_idx in range(n_picks):
        # Good channels = all except bad_idx
        good_mask = torch.ones(n_picks, dtype=torch.bool, device=compute_device)
        good_mask[bad_idx] = False
        good_idx = torch.where(good_mask)[0]

        # Extract G submatrices
        G_from = G_all[good_idx][:, good_idx]
        G_to_from = G_all[bad_idx:bad_idx + 1, good_idx]  # (1, n_good)

        # Add regularization
        n_from = len(good_idx)
        G_from_reg = G_from + 1e-5 * torch.eye(
            n_from, device=compute_device, dtype=compute_dtype
        )

        # Build C matrix and compute pseudo-inverse
        ones_col = torch.ones((n_from, 1), device=compute_device, dtype=compute_dtype)
        ones_row = torch.ones((1, n_from), device=compute_device, dtype=compute_dtype)
        zero = torch.zeros((1, 1), device=compute_device, dtype=compute_dtype)

        C = torch.cat(
            [
                torch.cat([G_from_reg, ones_col], dim=1),
                torch.cat([ones_row, zero], dim=1),
            ],
            dim=0,
        )

        C_inv = torch.linalg.pinv(C)

        # Interpolation weights for this channel: (1, n_good)
        ones_to = torch.ones((1, 1), device=compute_device, dtype=compute_dtype)
        weights = torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]
        weights = weights.to(device=device, dtype=data_dtype)

        # Store in matrix: interp_matrices[bad_idx, good_idx] = weights
        interp_matrices[bad_idx, good_idx.to(device)] = weights.squeeze(0)

    _LOOCV_INTERP_CACHE[cache_key] = interp_matrices
    return interp_matrices


def gpu_clean_by_interp(inst, picks=None, device=None, verbose=True):
    """Clean epochs/evoked by LOOCV interpolation on GPU.

    GPU-accelerated version of clean_by_interp that keeps data on GPU
    throughout the entire interpolation process.

    OPTIMIZED: Pre-computes and caches ALL interpolation matrices once,
    then applies them in a single batch matrix multiply.

    Parameters
    ----------
    inst : mne.Evoked or mne.Epochs
        The evoked or epochs object.
    picks : array-like or None
        Channels to include for interpolation.
    device : str or torch.device, optional
        Device to run on.
    verbose : bool
        Whether to show progress.

    Returns
    -------
    DeviceArray
        Interpolated data staying on GPU.
    """
    import torch
    from .utils import _handle_picks, _get_epochs_type

    backend = get_backend()
    if backend.name != "torch":
        raise RuntimeError("gpu_clean_by_interp requires torch backend")

    if device is None:
        device = backend.device

    picks = _handle_picks(info=inst.info, picks=picks)
    BaseEpochs = _get_epochs_type()

    # Determine compute strategy
    use_cuda = is_cuda_device(device)

    if use_cuda:
        compute_device = device
        compute_dtype = torch.float64
        data_dtype = torch.float64
    else:
        compute_device = "cpu"
        compute_dtype = torch.float64
        data_dtype = torch.float32 if device == "mps" else torch.float64

    # Get positions for interpolation
    pos_all = inst._get_channel_positions(picks)

    # Get cached interpolation matrices (or compute if not cached)
    if verbose:
        print(
            "  Computing/loading LOOCV interpolation matrices...", end=" ", flush=True
        )
    interp_matrices = _get_loocv_interp_matrices(
        pos_all, picks, device, compute_device, compute_dtype, data_dtype
    )
    if verbose:
        print("done.")

    # Transfer data to GPU
    data_gpu = torch.tensor(inst._data, dtype=data_dtype, device=device)

    # Apply interpolation in ONE batch operation
    # For epochs: data_gpu is (n_epochs, n_channels, n_times)
    # For evoked: data_gpu is (n_channels, n_times)

    if isinstance(inst, BaseEpochs):
        # Epochs: (n_epochs, n_channels, n_times)
        # Extract picked channels
        data_picks = data_gpu[:, picks, :]  # (n_epochs, n_picks, n_times)

        # Apply interpolation: result[e, i, t] = sum_j(interp[i,j] * data[e,j,t])
        # interp_matrices: (n_picks, n_picks), data_picks: (n_epochs, n_picks, n_times)
        # Use einsum: 'ij,ejt->eit'
        result_picks = torch.einsum("ij,ejt->eit", interp_matrices, data_picks)

        # Put back into full data
        result_gpu = data_gpu.clone()
        result_gpu[:, picks, :] = result_picks
    else:
        # Evoked: (n_channels, n_times)
        data_picks = data_gpu[picks, :]  # (n_picks, n_times)

        # Apply interpolation: result[i, t] = sum_j(interp[i,j] * data[j,t])
        result_picks = interp_matrices @ data_picks  # (n_picks, n_times)

        result_gpu = data_gpu.clone()
        result_gpu[picks, :] = result_picks

    return DeviceArray(result_gpu, backend, str(device))


def benchmark_interpolation_gpu(n_epochs=100, n_channels=64, n_times=1000, n_iters=3):
    """Benchmark GPU vs CPU interpolation.

    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    n_times : int
        Number of time points.
    n_iters : int
        Number of iterations for timing.

    Returns
    -------
    dict
        Benchmark results.
    """
    import time
    import torch
    from mne.channels.interpolation import _make_interpolation_matrix as cpu_make_interp

    backend = get_backend()
    if backend.name != "torch":
        print("Benchmark requires torch backend")
        return None

    device = backend.device
    print(f"Benchmarking on device: {device}")

    # Create random sensor positions on unit sphere
    np.random.seed(42)
    theta = np.random.uniform(0, np.pi, n_channels)
    phi = np.random.uniform(0, 2 * np.pi, n_channels)
    pos = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    # Simulate 5 bad channels
    n_bad = 5
    pos_good = pos[n_bad:]
    pos_bad = pos[:n_bad]

    # Create fake data
    data = np.random.randn(n_epochs, n_channels, n_times).astype(np.float32)

    # Benchmark CPU
    times_cpu = []
    for _ in range(n_iters):
        start = time.perf_counter()
        interp_cpu = cpu_make_interp(pos_good, pos_bad)
        # Simulate applying to epochs
        for e in range(n_epochs):
            _ = interp_cpu @ data[e, n_bad:, :]
        times_cpu.append(time.perf_counter() - start)

    cpu_time = np.median(times_cpu) * 1000

    # Benchmark GPU
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

    times_gpu = []
    for _ in range(n_iters):
        start = time.perf_counter()

        interp_gpu = gpu_make_interpolation_matrix(pos_good, pos_bad, device=device)

        # Transfer data once
        data_gpu = torch.tensor(data, dtype=torch.float32, device=device)

        # Apply interpolation - batched
        interp_tensor = interp_gpu.data
        good_data = data_gpu[:, n_bad:, :]
        torch.einsum("bg,egt->ebt", interp_tensor, good_data)

        # Force synchronization
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        times_gpu.append(time.perf_counter() - start)

    gpu_time = np.median(times_gpu) * 1000

    print(
        f"\nInterpolation Benchmark ({n_epochs} epochs, "
        f"{n_channels} channels, {n_times} times):"
    )
    print(f"  CPU: {cpu_time:.1f} ms")
    print(f"  GPU: {gpu_time:.1f} ms")
    print(f"  Speedup: {cpu_time / gpu_time:.1f}x")

    return {"cpu_ms": cpu_time, "gpu_ms": gpu_time, "speedup": cpu_time / gpu_time}


def gpu_batch_interpolate_all_n_interp(
    epochs, labels_list, picks, pos, device=None, verbose=True
):
    """Batch interpolation for ALL n_interpolate values at once.

    This is the key optimization: instead of interpolating sequentially
    for each n_interpolate value, we do them ALL in parallel on GPU.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to interpolate.
    labels_list : list of np.ndarray
        List of label arrays, one per n_interpolate value.
        Each array has shape (n_epochs, n_channels).
        Label values: 0=good, 1=bad, 2=to interpolate.
    picks : np.ndarray
        Channel indices that were picked.
    pos : np.ndarray, shape (n_picks, 3)
        3D positions of picked channels (will be normalized internally).
    device : str or torch.device, optional
        Device to run on.
    verbose : bool
        Whether to show progress.

    Returns
    -------
    list of torch.Tensor
        List of interpolated data tensors (one per n_interpolate value),
        each with shape (n_epochs, len(picks), n_times).
    """
    import torch
    from .utils import _pbar, _GDKW

    backend = get_backend()
    if backend.name != "torch":
        raise RuntimeError("gpu_batch_interpolate_all_n_interp requires torch backend")

    if device is None:
        device = backend.device

    len(labels_list)
    n_epochs = len(epochs)
    n_picks = len(picks)
    picks = np.asarray(picks)

    # Determine compute strategy based on device type
    # CUDA: float64 on device for everything (bit-exact with CPU)
    # MPS: float64 on CPU for pinv, float32 for data matmul
    use_cuda = is_cuda_device(device)

    if use_cuda:
        # CUDA: everything in float64 on device
        compute_device = device
        compute_dtype = torch.float64
        data_dtype = torch.float64
    else:
        # MPS or CPU: compute on CPU in float64, data in float32 for MPS
        compute_device = "cpu"
        compute_dtype = torch.float64
        data_dtype = torch.float32 if device == "mps" else torch.float64

    # Get original data and transfer to GPU once
    X_full = epochs.get_data(**_GDKW)
    X_full.shape[2]

    X_gpu = torch.tensor(X_full, dtype=data_dtype, device=device)

    # Normalize positions to unit sphere (in float64 for precision)
    pos_t = torch.tensor(pos, dtype=compute_dtype, device=compute_device)
    norms = torch.norm(pos_t, dim=1, keepdim=True)
    pos_t = pos_t / norms

    # Pre-compute full G matrix for all positions (n_picks x n_picks) in float64
    cosang_all = pos_t @ pos_t.T
    G_all = _calc_g_torch(cosang_all)

    # Prepare all interp_channels lists (convert labels to channel indices)
    all_interp_ch_indices = []

    for labels in labels_list:
        interp_channels = []
        for epoch_idx in range(n_epochs):
            # Find channels marked for interpolation (label == 2)
            to_interp_mask = labels[epoch_idx, picks] == 2
            ch_indices_in_picks = np.where(to_interp_mask)[0].tolist()
            interp_channels.append(ch_indices_in_picks)
        all_interp_ch_indices.append(interp_channels)

    # Cache interpolation matrices (shared across n_interp values if same pattern)
    interp_cache = {}

    # Process all n_interp values
    results = []

    desc = "Batch interpolating all n_interp values"
    for interp_idx, interp_channels in enumerate(
        _pbar(all_interp_ch_indices, desc=desc, verbose=verbose)
    ):
        # Clone the GPU data for this n_interp value
        data_gpu = X_gpu[:, picks, :].clone()

        for epoch_idx, bad_ch_indices in enumerate(interp_channels):
            if len(bad_ch_indices) == 0:
                continue

            # Create cache key from bad channel pattern
            cache_key = tuple(sorted(bad_ch_indices))

            if cache_key not in interp_cache:
                # Create masks for good/bad channels
                goods_mask = np.ones(n_picks, dtype=bool)
                for bad_idx in bad_ch_indices:
                    goods_mask[bad_idx] = False
                bads_mask = ~goods_mask

                # Get indices of good and bad channels within picks
                good_idx_in_picks = np.where(goods_mask)[0]
                bad_idx_in_picks = np.where(bads_mask)[0]

                # Create torch index tensors for operations on G matrix
                good_idx_t = torch.tensor(
                    good_idx_in_picks, device=compute_device, dtype=torch.long
                )
                bad_idx_t = torch.tensor(
                    bad_idx_in_picks, device=compute_device, dtype=torch.long
                )

                # Extract submatrices from pre-computed G using advanced indexing
                G_from = G_all[good_idx_t][:, good_idx_t]
                G_to_from = G_all[bad_idx_t][:, good_idx_t]

                # Add regularization
                n_from = len(good_idx_in_picks)
                G_from_reg = G_from + 1e-5 * torch.eye(
                    n_from, device=compute_device, dtype=compute_dtype
                )

                # Build C matrix and compute pseudo-inverse
                ones_col = torch.ones(
                    (n_from, 1), device=compute_device, dtype=compute_dtype
                )
                ones_row = torch.ones(
                    (1, n_from), device=compute_device, dtype=compute_dtype
                )
                zero = torch.zeros((1, 1), device=compute_device, dtype=compute_dtype)

                C = torch.cat(
                    [
                        torch.cat([G_from_reg, ones_col], dim=1),
                        torch.cat([ones_row, zero], dim=1),
                    ],
                    dim=0,
                )

                C_inv = torch.linalg.pinv(C)

                # Compute interpolation matrix (n_bad, n_good)
                n_bad = len(bad_idx_in_picks)
                ones_to = torch.ones(
                    (n_bad, 1), device=compute_device, dtype=compute_dtype
                )
                interpolation = torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]

                # Move to target device with appropriate dtype
                interpolation = interpolation.to(device=device, dtype=data_dtype)

                interp_cache[cache_key] = (
                    interpolation,
                    good_idx_in_picks,
                    bad_idx_in_picks,
                )

            interpolation, good_idx, bad_idx = interp_cache[cache_key]

            # Apply interpolation for this epoch
            # (n_bad, n_good) @ (n_good, n_times) -> (n_bad, n_times)
            good_data = data_gpu[epoch_idx, good_idx, :]
            interpolated = interpolation @ good_data
            data_gpu[epoch_idx, bad_idx, :] = interpolated

        results.append(data_gpu)

    return results


def gpu_interpolate_bad_epochs(data, interp_channels, picks, pos, device=None):
    """GPU-accelerated interpolation for epochs with per-epoch bad channels.

    This is a GPU version of _interpolate_bad_epochs that keeps all data
    on GPU throughout the interpolation process.

    Parameters
    ----------
    data : np.ndarray, shape (n_epochs, n_channels_total, n_times)
        The epoch data to interpolate.
    interp_channels : list of list of int
        For each epoch, list of channel INDICES (within picks) to interpolate.
    picks : np.ndarray
        Channel indices that were picked.
    pos : np.ndarray, shape (n_picks, 3)
        3D positions of picked channels, normalized to unit vectors.
    device : str or torch.device, optional
        Device to run on.

    Returns
    -------
    torch.Tensor
        Interpolated data on GPU, shape (n_epochs, n_channels_total, n_times).
    """
    import torch

    backend = get_backend()
    if backend.name != "torch":
        raise RuntimeError("gpu_interpolate_bad_epochs requires torch backend")

    if device is None:
        device = backend.device

    n_epochs, n_channels_total, n_times = data.shape
    n_picks = len(picks)
    picks = np.asarray(picks)  # Ensure picks is numpy array

    # Determine compute strategy based on device type
    # CUDA: float64 on device for everything (bit-exact with CPU)
    # MPS: float64 on CPU for pinv, float32 for data matmul
    use_cuda = is_cuda_device(device)

    if use_cuda:
        # CUDA: everything in float64 on device
        compute_device = device
        compute_dtype = torch.float64
        data_dtype = torch.float64
    else:
        # MPS or CPU: compute on CPU in float64, data in float32 for MPS
        compute_device = "cpu"
        compute_dtype = torch.float64
        data_dtype = torch.float32 if device == "mps" else torch.float64

    # Transfer data to GPU once with appropriate dtype
    if isinstance(data, torch.Tensor):
        data_gpu = data.clone().to(dtype=data_dtype)
    else:
        data_gpu = torch.tensor(data, dtype=data_dtype, device=device)

    # Pre-compute positions tensor
    pos_t = torch.tensor(pos, dtype=compute_dtype, device=compute_device)

    # Pre-compute full G matrix for all positions (n_picks x n_picks)
    cosang_all = pos_t @ pos_t.T
    G_all = _calc_g_torch(cosang_all)

    # Cache interpolation matrices to avoid recomputing for same bad channel patterns
    interp_cache = {}

    for epoch_idx, bad_ch_indices in enumerate(interp_channels):
        if len(bad_ch_indices) == 0:
            continue

        # Create cache key from bad channel pattern
        cache_key = tuple(sorted(bad_ch_indices))

        if cache_key not in interp_cache:
            # Create masks for good/bad channels
            goods_mask = np.ones(n_picks, dtype=bool)
            for bad_idx in bad_ch_indices:
                goods_mask[bad_idx] = False
            bads_mask = ~goods_mask

            # Get indices of good and bad channels within picks
            good_idx_in_picks = np.where(goods_mask)[0]
            bad_idx_in_picks = np.where(bads_mask)[0]

            # Create torch index tensors for operations on G matrix
            good_idx_t = torch.tensor(
                good_idx_in_picks, device=compute_device, dtype=torch.long
            )
            bad_idx_t = torch.tensor(
                bad_idx_in_picks, device=compute_device, dtype=torch.long
            )

            # Extract submatrices from pre-computed G using advanced indexing
            G_from = G_all[good_idx_t][:, good_idx_t]
            G_to_from = G_all[bad_idx_t][:, good_idx_t]

            # Add regularization
            n_from = len(good_idx_in_picks)
            G_from_reg = G_from + 1e-5 * torch.eye(
                n_from, device=compute_device, dtype=compute_dtype
            )

            # Build C matrix and compute pseudo-inverse
            ones_col = torch.ones(
                (n_from, 1), device=compute_device, dtype=compute_dtype
            )
            ones_row = torch.ones(
                (1, n_from), device=compute_device, dtype=compute_dtype
            )
            zero = torch.zeros((1, 1), device=compute_device, dtype=compute_dtype)

            C = torch.cat(
                [
                    torch.cat([G_from_reg, ones_col], dim=1),
                    torch.cat([ones_row, zero], dim=1),
                ],
                dim=0,
            )

            C_inv = torch.linalg.pinv(C)

            # Compute interpolation matrix (n_bad, n_good)
            n_bad = len(bad_idx_in_picks)
            ones_to = torch.ones((n_bad, 1), device=compute_device, dtype=compute_dtype)
            interpolation = torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]

            # Move to target device with appropriate dtype
            interpolation = interpolation.to(device=device, dtype=data_dtype)

            # Store original picks for good and bad channels (in full data indexing)
            good_picks = picks[goods_mask]
            bad_picks = picks[bads_mask]

            interp_cache[cache_key] = (interpolation, good_picks, bad_picks)

        interpolation, good_picks, bad_picks = interp_cache[cache_key]

        # Apply interpolation for this epoch
        # (n_bad, n_good) @ (n_good, n_times) -> (n_bad, n_times)
        good_data = data_gpu[epoch_idx, good_picks, :]
        interpolated = interpolation @ good_data
        data_gpu[epoch_idx, bad_picks, :] = interpolated

    return data_gpu


if __name__ == "__main__":
    # Run benchmark when executed directly
    import os

    os.environ["AUTOREJECT_BACKEND"] = "torch"

    # Reload backend
    from . import backends

    backends._backend = None

    benchmark_interpolation_gpu()
