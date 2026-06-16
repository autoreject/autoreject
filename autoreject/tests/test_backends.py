"""Tests for compute backends.

These tests verify that all backends produce correct and consistent results.
They also test the fallback mechanisms and environment variable configuration.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from autoreject.backends import (
    detect_hardware,
    clear_backend_cache,
    get_backend,
    get_backend_names,
    NumpyBackend,
    TorchBackend,
    DeviceArray,
    is_device_array,
    use_backend,
)


# Mark all tests in this module as backend tests
pytestmark = pytest.mark.backends


class TestDetectHardware:
    """Tests for hardware detection."""

    def test_detect_hardware_returns_dict(self):
        """Test that detect_hardware returns a dictionary."""

        # Clear cache to get fresh results
        clear_backend_cache()

        hw = detect_hardware()
        assert isinstance(hw, dict)

    def test_cpu_always_available(self):
        """Test that CPU is always detected as available."""

        clear_backend_cache()
        hw = detect_hardware()
        assert hw.get('cpu') is True

    def test_detect_hardware_cached(self):
        """Test that detect_hardware results are cached."""

        clear_backend_cache()
        hw1 = detect_hardware()
        hw2 = detect_hardware()

        # Should return the same object (cached)
        assert hw1 is hw2


class TestGetBackend:
    """Tests for backend selection."""

    def test_get_backend_returns_backend(self):
        """Test that get_backend returns a valid backend object."""

        clear_backend_cache()
        backend = get_backend()

        # Should have required attributes
        assert hasattr(backend, 'name')
        assert hasattr(backend, 'device')
        assert hasattr(backend, 'ptp')
        assert hasattr(backend, 'median')
        assert hasattr(backend, 'correlation')
        assert hasattr(backend, 'to_numpy')

    def test_get_numpy_backend_explicit(self):
        """Test that we can explicitly request NumPy backend."""

        clear_backend_cache()
        backend = get_backend(prefer='numpy')

        assert isinstance(backend, NumpyBackend)
        assert backend.name == 'numpy'
        assert backend.device == 'cpu'

    def test_get_backend_env_var(self):
        """Test that AUTOREJECT_BACKEND environment variable is respected."""

        # Use context manager to set backend
        with use_backend('numpy'):
            backend = get_backend()
            assert isinstance(backend, NumpyBackend)

    def test_get_backend_invalid_prefer_warns(self):
        """Test that invalid prefer value triggers warnings."""
        import warnings

        clear_backend_cache()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = get_backend(prefer='invalid_backend')

            # Should have warnings about invalid/unknown backend
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) >= 1, "Expected at least one RuntimeWarning"

        # Should fall back to auto-detection
        assert backend is not None

    def test_get_backend_cached(self):
        """Test that backends are cached."""

        clear_backend_cache()

        backend1 = get_backend(prefer='numpy')
        backend2 = get_backend(prefer='numpy')

        # Should return the same object (cached)
        assert backend1 is backend2

    def test_clear_backend_cache(self):
        """Test that cache clearing works."""

        backend1 = get_backend(prefer='numpy')
        clear_backend_cache()
        backend2 = get_backend(prefer='numpy')

        # Should be different objects after cache clear
        assert backend1 is not backend2


class TestNumpyBackend:
    """Tests for NumPy backend operations."""

    @pytest.fixture
    def backend(self):
        """Get NumPy backend."""
        return NumpyBackend()

    def test_ptp_1d(self, backend):
        """Test ptp on 1D array."""
        data = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        result = backend.ptp(data, axis=0)
        expected = 6.0  # 7 - 1
        assert_allclose(result, expected)

    def test_ptp_2d(self, backend):
        """Test ptp on 2D array."""
        data = np.array([
            [1.0, 5.0, 3.0],
            [2.0, 8.0, 4.0]
        ])
        result = backend.ptp(data, axis=-1)
        expected = np.array([4.0, 6.0])  # [5-1, 8-2]
        assert_allclose(result, expected)

    def test_ptp_3d(self, backend):
        """Test ptp on 3D array (epochs x channels x times)."""
        np.random.seed(42)
        data = np.random.randn(10, 5, 100)

        result = backend.ptp(data, axis=-1)
        expected = np.ptp(data, axis=-1)

        assert_allclose(result, expected)
        assert result.shape == (10, 5)

    def test_median_all(self, backend):
        """Test median over all elements."""
        data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = backend.median(data)
        assert_allclose(result, 5.0)

    def test_median_axis(self, backend):
        """Test median along an axis."""
        data = np.array([
            [1.0, 5.0, 3.0],
            [2.0, 8.0, 4.0]
        ])
        result = backend.median(data, axis=-1)
        expected = np.array([3.0, 4.0])
        assert_allclose(result, expected)

    def test_correlation(self, backend):
        """Test correlation computation."""
        np.random.seed(42)

        # Create correlated data
        x = np.random.randn(100, 5)
        y = x + np.random.randn(100, 5) * 0.1  # Highly correlated

        corr = backend.correlation(x, y)

        # Should be close to 1 for each channel
        assert_allclose(corr, np.ones(5), atol=0.1)

    def test_correlation_perfect(self, backend):
        """Test correlation with perfectly correlated data."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = x * 2  # Perfectly correlated

        corr = backend.correlation(x, y)
        assert_allclose(corr, np.ones(2), atol=1e-10)

    def test_to_numpy(self, backend):
        """Test conversion to numpy."""
        data = [1.0, 2.0, 3.0]
        result = backend.to_numpy(data)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_repr(self, backend):
        """Test string representation."""
        assert 'NumpyBackend' in repr(backend)
        assert 'cpu' in repr(backend)


class TestTorchBackend:
    """Tests for PyTorch backend operations."""

    @pytest.fixture
    def backend(self):
        """Get PyTorch backend, skip if not available."""
        pytest.importorskip('torch')
        return TorchBackend()

    @pytest.fixture
    def numpy_backend(self):
        """Get NumPy backend for comparison."""
        return NumpyBackend()

    def test_ptp_matches_numpy(self, backend, numpy_backend):
        """Test that PyTorch ptp matches NumPy."""
        np.random.seed(42)
        data = np.random.randn(10, 5, 100)

        result = backend.ptp(data, axis=-1)
        expected = numpy_backend.ptp(data, axis=-1)

        assert_allclose(result, expected, rtol=1e-5)

    def test_median_matches_numpy(self, backend, numpy_backend):
        """Test that PyTorch median matches NumPy."""
        np.random.seed(42)
        data = np.random.randn(10, 5)

        result = backend.median(data, axis=-1)
        expected = numpy_backend.median(data, axis=-1)

        assert_allclose(result, expected, rtol=1e-5)

    def test_correlation_matches_numpy(self, backend, numpy_backend):
        """Test that PyTorch correlation matches NumPy."""
        np.random.seed(42)
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5)

        result = backend.correlation(x, y)
        expected = numpy_backend.correlation(x, y)

        assert_allclose(result, expected, rtol=1e-5)

    def test_to_numpy_from_tensor(self, backend):
        """Test conversion from PyTorch tensor to NumPy."""
        import torch

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = backend.to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_device_detection(self, backend):
        """Test that device is properly detected."""
        assert backend.device in ('cpu', 'cuda', 'mps')


class TestBackendConsistency:
    """Tests ensuring all backends produce consistent results."""

    @pytest.fixture
    def all_backends(self):
        """Get all available backends."""

        clear_backend_cache()

        backends = []
        for name in get_backend_names():
            try:
                backend = get_backend(prefer=name)
                backends.append(backend)
                clear_backend_cache()
            except Exception:
                pass

        return backends

    def test_ptp_consistency(self, all_backends):
        """Test that all backends produce the same ptp results."""
        if len(all_backends) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        np.random.seed(42)
        data = np.random.randn(10, 5, 100)

        results = [b.ptp(data, axis=-1) for b in all_backends]

        for i in range(1, len(results)):
            assert_allclose(
                results[0], results[i],
                rtol=1e-5,
                err_msg=f"{all_backends[0].name} vs {all_backends[i].name}"
            )

    def test_median_consistency(self, all_backends):
        """Test that all backends produce the same median results."""
        if len(all_backends) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        np.random.seed(42)
        data = np.random.randn(10, 5)

        results = [b.median(data, axis=-1) for b in all_backends]

        for i in range(1, len(results)):
            assert_allclose(
                results[0], results[i],
                rtol=1e-5,
                err_msg=f"{all_backends[0].name} vs {all_backends[i].name}"
            )

    def test_correlation_consistency(self, all_backends):
        """Test that all backends produce the same correlation results."""
        if len(all_backends) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        np.random.seed(42)
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5)

        results = [b.correlation(x, y) for b in all_backends]

        for i in range(1, len(results)):
            assert_allclose(
                results[0], results[i],
                rtol=1e-5,
                err_msg=f"{all_backends[0].name} vs {all_backends[i].name}"
            )


class TestGetBackendNames:
    """Tests for get_backend_names function."""

    def test_returns_list(self):
        """Test that get_backend_names returns a list."""

        names = get_backend_names()
        assert isinstance(names, list)

    def test_numpy_always_included(self):
        """Test that 'numpy' is always in the list."""

        names = get_backend_names()
        assert 'numpy' in names

    def test_names_are_strings(self):
        """Test that all names are strings."""

        names = get_backend_names()
        assert all(isinstance(n, str) for n in names)


# =============================================================================
# Tests for DeviceArray and keep_on_device API (Phase 6)
# =============================================================================

class TestDeviceArray:
    """Tests for DeviceArray wrapper class."""

    def test_device_array_creation(self):
        """Test creating a DeviceArray."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5)

        arr = backend.to_device(data)
        assert isinstance(arr, DeviceArray)
        assert arr.shape == data.shape
        assert arr.device == 'cpu'

    def test_device_array_numpy_conversion(self):
        """Test converting DeviceArray back to numpy."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5)

        arr = backend.to_device(data)
        result = arr.numpy()

        assert isinstance(result, np.ndarray)
        assert_allclose(result, data)

    def test_device_array_properties(self):
        """Test DeviceArray properties."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5, 3)

        arr = backend.to_device(data)

        assert arr.shape == (10, 5, 3)
        assert arr.ndim == 3
        assert len(arr) == 10
        assert arr.backend is backend

    def test_is_device_array(self):
        """Test is_device_array helper."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5)

        arr = backend.to_device(data)

        assert is_device_array(arr) is True
        assert is_device_array(data) is False
        assert is_device_array([1, 2, 3]) is False


class TestKeepOnDevice:
    """Tests for keep_on_device parameter."""

    def test_ptp_keep_on_device_false(self):
        """Test ptp returns numpy when keep_on_device=False."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5, 100)

        result = backend.ptp(data, axis=-1, keep_on_device=False)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, DeviceArray)

    def test_ptp_keep_on_device_true(self):
        """Test ptp returns DeviceArray when keep_on_device=True."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5, 100)

        result = backend.ptp(data, axis=-1, keep_on_device=True)
        assert isinstance(result, DeviceArray)

    def test_median_keep_on_device(self):
        """Test median with keep_on_device."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5)

        result_numpy = backend.median(data, axis=0, keep_on_device=False)
        result_device = backend.median(data, axis=0, keep_on_device=True)

        assert isinstance(result_numpy, np.ndarray)
        assert isinstance(result_device, DeviceArray)
        assert_allclose(result_numpy, result_device.numpy())

    def test_correlation_keep_on_device(self):
        """Test correlation with keep_on_device."""

        backend = get_backend(prefer='numpy')
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5)

        result_numpy = backend.correlation(x, y, keep_on_device=False)
        result_device = backend.correlation(x, y, keep_on_device=True)

        assert isinstance(result_numpy, np.ndarray)
        assert isinstance(result_device, DeviceArray)
        assert_allclose(result_numpy, result_device.numpy())

    def test_matmul_keep_on_device(self):
        """Test matmul with keep_on_device."""

        backend = get_backend(prefer='numpy')
        a = np.random.randn(10, 5)
        b = np.random.randn(5, 3)

        result_numpy = backend.matmul(a, b, keep_on_device=False)
        result_device = backend.matmul(a, b, keep_on_device=True)

        assert isinstance(result_numpy, np.ndarray)
        assert isinstance(result_device, DeviceArray)
        assert_allclose(result_numpy, result_device.numpy())


class TestDeviceArrayChaining:
    """Tests for chaining operations on DeviceArrays."""

    def test_operations_on_device_array(self):
        """Test that operations can accept DeviceArray as input."""

        backend = get_backend(prefer='numpy')
        data = np.random.randn(10, 5, 100)

        # Transfer to device
        gpu_data = backend.to_device(data)

        # Chain operations
        result1 = backend.ptp(gpu_data, axis=-1, keep_on_device=True)
        result2 = backend.median(result1, axis=0, keep_on_device=True)

        # Both intermediate results should be DeviceArrays
        assert isinstance(result1, DeviceArray)
        assert isinstance(result2, DeviceArray)

        # Final conversion to numpy
        final = backend.to_numpy(result2)
        assert isinstance(final, np.ndarray)

    def test_mixed_input_types(self):
        """Test that backends handle mixed numpy and DeviceArray inputs."""

        backend = get_backend(prefer='numpy')
        x_np = np.random.randn(100, 5)
        y_device = backend.to_device(np.random.randn(100, 5))

        # Should work with mixed types
        result = backend.correlation(x_np, y_device, keep_on_device=False)
        assert isinstance(result, np.ndarray)


@pytest.mark.skipif(
    not pytest.importorskip('torch', reason="PyTorch not installed"),
    reason="PyTorch not installed"
)
class TestTorchDeviceArray:
    """Tests for DeviceArray with PyTorch backend."""

    def test_torch_to_device(self):
        """Test transferring data to GPU with torch backend."""

        backend = get_backend(prefer='torch')
        data = np.random.randn(10, 5, 100)

        gpu_data = backend.to_device(data)

        assert isinstance(gpu_data, DeviceArray)
        # Device should be cuda, mps, or cpu depending on hardware
        assert gpu_data.device in ('cuda', 'mps', 'cpu')

    def test_torch_operations_stay_on_gpu(self):
        """Test that operations with keep_on_device=True stay on GPU."""
        import torch

        backend = get_backend(prefer='torch')
        data = np.random.randn(10, 5, 100)

        gpu_data = backend.to_device(data)
        result = backend.ptp(gpu_data, axis=-1, keep_on_device=True)

        # Result should be a DeviceArray containing a torch.Tensor
        assert isinstance(result, DeviceArray)
        assert isinstance(result.data, torch.Tensor)
        # Tensor should be on the same device as the backend
        assert result.data.device.type == backend._device.type

    def test_torch_chain_without_transfer(self):
        """Test chaining operations without CPU<->GPU transfers."""
        import torch

        backend = get_backend(prefer='torch')
        data = np.random.randn(50, 32, 200)

        # Single transfer to GPU
        gpu_data = backend.to_device(data)

        # Chain of operations - all stay on GPU
        ptp_result = backend.ptp(gpu_data, axis=-1, keep_on_device=True)
        med_result = backend.median(ptp_result, axis=0, keep_on_device=True)

        # Verify all intermediate results are torch tensors on device
        assert isinstance(ptp_result.data, torch.Tensor)
        assert isinstance(med_result.data, torch.Tensor)
        assert ptp_result.data.device.type == backend._device.type
        assert med_result.data.device.type == backend._device.type

        # Single transfer back to CPU
        final = backend.to_numpy(med_result)
        assert isinstance(final, np.ndarray)

    def test_torch_extended_operations(self):
        """Test extended operations available in TorchBackend."""

        backend = get_backend(prefer='torch')
        data = np.random.randn(10, 5)

        gpu_data = backend.to_device(data)

        # Test extended operations
        zeros = backend.zeros((5, 3))
        assert isinstance(zeros, DeviceArray)

        result_sum = backend.sum(gpu_data, axis=0, keep_on_device=True)
        assert isinstance(result_sum, DeviceArray)

        result_mean = backend.mean(gpu_data, axis=0, keep_on_device=True)
        assert isinstance(result_mean, DeviceArray)

        result_sqrt = backend.sqrt(
            backend.abs(gpu_data, keep_on_device=True), keep_on_device=True
        )
        assert isinstance(result_sqrt, DeviceArray)


class TestMultiBackendConsistency:
    """Test that different backends produce consistent results."""

    def test_autoreject_consistency_across_backends(self):
        """Test AutoReject produces consistent results with numpy vs torch backends.

        This test verifies that the full AutoReject pipeline produces similar
        results regardless of backend. On macOS with MPS, we use float32
        (vs float64 on CPU/CUDA) which can lead to small numerical differences.
        We use a relaxed tolerance to account for these precision differences.
        """
        import mne
        from autoreject import AutoReject

        # Skip if torch backend not available
        if 'torch' not in get_backend_names():
            pytest.skip("PyTorch backend not available")

        # Create small synthetic dataset with real channel names
        n_epochs = 20
        n_times = 100
        sfreq = 100

        # Use actual channel names from standard_1020 montage
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        n_channels = len(ch_names)
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names, sfreq, ch_types=ch_types)

        # Add montage for interpolation
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

        # Synthetic data with artifacts
        rng = np.random.RandomState(42)
        data = rng.randn(n_epochs, n_channels, n_times) * 1e-6
        data[5, 2, :] += 50e-6  # Large artifact on epoch 5, channel 2
        data[10, :, :] *= 3     # Noisy epoch 10

        # Create events
        events = np.column_stack([
            np.arange(0, n_epochs * n_times, n_times),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int)
        ])
        epochs = mne.EpochsArray(data, info, events, tmin=0, verbose=False)

        # Run AutoReject with numpy backend
        with use_backend('numpy'):
            ar_numpy = AutoReject(
                n_interpolate=[1, 2],
                consensus=np.linspace(0, 1.0, 3),
                cv=3,
                random_state=42,
                verbose=False
            )
            ar_numpy.fit(epochs)
            threshes_numpy = ar_numpy.threshes_
            consensus_numpy = ar_numpy.consensus_
            n_interp_numpy = ar_numpy.n_interpolate_

        # Run AutoReject with torch backend
        with use_backend('torch'):
            ar_torch = AutoReject(
                n_interpolate=[1, 2],
                consensus=np.linspace(0, 1.0, 3),
                cv=3,
                random_state=42,
                verbose=False
            )
            ar_torch.fit(epochs)
            threshes_torch = ar_torch.threshes_
            consensus_torch = ar_torch.consensus_
            n_interp_torch = ar_torch.n_interpolate_

        # Compare results - should be very similar
        # Use relaxed tolerance for float32 (MPS) vs float64 (CPU/CUDA) differences
        assert threshes_numpy.keys() == threshes_torch.keys(), \
            "Channel types differ between backends"

        for ch_name in threshes_numpy.keys():
            # Thresholds should be very close (within 0.1% relative tolerance)
            # This accounts for float32 (MPS) vs float64 (CPU/CUDA) differences
            # In practice, results are often identical to machine precision
            assert_allclose(
                threshes_numpy[ch_name],
                threshes_torch[ch_name],
                rtol=0.001,  # 0.1% tolerance
                err_msg=f"Thresholds differ for channel {ch_name}"
            )

        # Consensus and n_interpolate should be identical
        # (same CV logic, discrete values)
        assert consensus_numpy == consensus_torch, (
            f"Consensus differs: {consensus_numpy} vs {consensus_torch}"
        )
        assert n_interp_numpy == n_interp_torch, \
            f"n_interpolate differs: {n_interp_numpy} vs {n_interp_torch}"

    def test_reject_log_consistency_across_backends(self):
        """Test that reject_log works correctly with both backends.

        This test verifies that reject_log structure and plot() method work
        correctly with both numpy and torch backends, and that the results
        are identical between backends.
        """
        import mne
        from autoreject import AutoReject

        # Skip if torch backend not available
        if 'torch' not in get_backend_names():
            pytest.skip("PyTorch backend not available")

        # Create small synthetic dataset
        n_epochs = 20
        n_times = 100
        sfreq = 100
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        n_channels = len(ch_names)
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names, sfreq, ch_types=ch_types)

        # Add montage for interpolation
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

        # Synthetic data with artifacts
        rng = np.random.RandomState(42)
        data = rng.randn(n_epochs, n_channels, n_times) * 1e-6
        data[5, 2, :] += 50e-6  # Large artifact
        data[10, :, :] *= 3     # Noisy epoch

        # Create events
        events = np.column_stack([
            np.arange(0, n_epochs * n_times, n_times),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int)
        ])
        epochs = mne.EpochsArray(data, info, events, tmin=0, verbose=False)

        # Test with numpy backend
        with use_backend('numpy'):
            ar_numpy = AutoReject(
                n_interpolate=[1, 2],
                consensus=np.linspace(0, 1.0, 3),
                cv=3,
                random_state=42,
                verbose=False
            )
            ar_numpy.fit(epochs)
            reject_log_numpy = ar_numpy.get_reject_log(epochs)

            # Verify reject_log structure
            assert hasattr(reject_log_numpy, 'labels')
            assert hasattr(reject_log_numpy, 'bad_epochs')
            assert hasattr(reject_log_numpy, 'ch_names')
            assert reject_log_numpy.labels.shape == (n_epochs, n_channels)
            assert len(reject_log_numpy.bad_epochs) == n_epochs
            assert reject_log_numpy.ch_names == ch_names

            # Test plot (just ensure it runs without error)
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                fig = reject_log_numpy.plot('horizontal')
                import matplotlib.pyplot as plt
                plt.close(fig)
            except ImportError:
                pass  # matplotlib not available

        # Test with torch backend
        with use_backend('torch'):
            ar_torch = AutoReject(
                n_interpolate=[1, 2],
                consensus=np.linspace(0, 1.0, 3),
                cv=3,
                random_state=42,
                verbose=False
            )
            ar_torch.fit(epochs)
            reject_log_torch = ar_torch.get_reject_log(epochs)

            # Verify reject_log structure
            assert hasattr(reject_log_torch, 'labels')
            assert hasattr(reject_log_torch, 'bad_epochs')
            assert hasattr(reject_log_torch, 'ch_names')
            assert reject_log_torch.labels.shape == (n_epochs, n_channels)
            assert len(reject_log_torch.bad_epochs) == n_epochs
            assert reject_log_torch.ch_names == ch_names

            # Test plot
            try:
                import matplotlib
                matplotlib.use('Agg')
                fig = reject_log_torch.plot('horizontal')
                import matplotlib.pyplot as plt
                plt.close(fig)
            except ImportError:
                pass

        # Compare reject_logs between backends
        # Labels should be identical (0=good, 1=bad, 2=interpolated)
        assert_array_equal(
            reject_log_numpy.labels,
            reject_log_torch.labels,
            err_msg="reject_log.labels differ between backends"
        )

        # Bad epochs should be identical
        assert_array_equal(
            reject_log_numpy.bad_epochs,
            reject_log_torch.bad_epochs,
            err_msg="reject_log.bad_epochs differ between backends"
        )
