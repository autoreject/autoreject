"""Tests for GPU interpolation module."""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import numpy as np
import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from autoreject.backends import use_backend, is_device_array  # noqa: E402
from autoreject.gpu_interpolation import (  # noqa: E402
    legval_torch,
    _calc_g_torch,
    _normalize_vectors_torch,
    gpu_make_interpolation_matrix,
    gpu_do_interp_dots,
)


@pytest.fixture(autouse=True)
def setup_torch_backend():
    """Force torch backend for all tests in this module."""
    with use_backend('torch'):
        yield


class TestLegvalTorch:
    """Tests for legval_torch function."""

    def test_legval_matches_numpy(self):
        """Test that legval_torch matches numpy's legval."""
        from numpy.polynomial.legendre import legval as np_legval

        x = np.array([[0.5, 0.8], [-0.3, 0.9]])
        c = [1.0, 2.0, 3.0, 4.0, 5.0]

        np_result = np_legval(x, c)
        torch_result = legval_torch(
            torch.tensor(x, dtype=torch.float32), c
        ).cpu().numpy()

        np.testing.assert_allclose(np_result, torch_result, rtol=1e-5)

    def test_legval_single_coeff(self):
        """Test with single coefficient."""
        x = torch.tensor([0.5, 1.0, -0.5])
        c = [2.5]
        result = legval_torch(x, c)
        np.testing.assert_allclose(result.cpu().numpy(), [2.5, 2.5, 2.5])

    def test_legval_two_coeffs(self):
        """Test with two coefficients."""
        x = torch.tensor([0.5])
        c = [1.0, 2.0]  # 1 + 2*x
        result = legval_torch(x, c)
        np.testing.assert_allclose(result.cpu().numpy(), [2.0])


class TestCalcGTorch:
    """Tests for _calc_g_torch function."""

    def test_calc_g_matches_mne(self):
        """Test that _calc_g_torch matches MNE's _calc_g."""
        from mne.channels.interpolation import _calc_g as mne_calc_g

        np.random.seed(42)
        cosang = np.random.randn(10, 10).astype(np.float32)
        cosang = np.clip(cosang, -1, 1)

        mne_result = mne_calc_g(cosang)
        torch_result = _calc_g_torch(torch.tensor(cosang)).cpu().numpy()

        np.testing.assert_allclose(mne_result, torch_result, rtol=1e-5)

    def test_calc_g_symmetric(self):
        """Test that G is symmetric for symmetric input."""
        cosang = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        G = _calc_g_torch(cosang)
        np.testing.assert_allclose(G.cpu().numpy(), G.T.cpu().numpy())


class TestNormalizeVectors:
    """Tests for _normalize_vectors_torch function."""

    def test_normalizes_to_unit_length(self):
        """Test that vectors are normalized to unit length."""
        pos = torch.tensor([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]])
        normalized = _normalize_vectors_torch(pos)
        norms = torch.norm(normalized, dim=1)
        np.testing.assert_allclose(norms.cpu().numpy(), [1.0, 1.0], rtol=1e-6)


class TestGpuMakeInterpolationMatrix:
    """Tests for gpu_make_interpolation_matrix function."""

    def test_matches_mne(self):
        """Test that GPU interpolation matrix matches MNE's."""
        from mne.channels.interpolation import _make_interpolation_matrix

        np.random.seed(42)
        n_good, n_bad = 30, 5
        theta = np.random.uniform(0, np.pi, n_good + n_bad)
        phi = np.random.uniform(0, 2 * np.pi, n_good + n_bad)
        pos = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        pos_good = pos[:n_good]
        pos_bad = pos[n_good:]

        mne_interp = _make_interpolation_matrix(pos_good, pos_bad)
        gpu_interp = gpu_make_interpolation_matrix(pos_good, pos_bad)

        assert is_device_array(gpu_interp)
        np.testing.assert_allclose(
            mne_interp.astype(np.float32),
            gpu_interp.data.cpu().numpy(),
            rtol=1e-4
        )

    def test_correct_shape(self):
        """Test that output has correct shape."""
        pos_good = np.random.randn(20, 3)
        pos_bad = np.random.randn(3, 3)

        result = gpu_make_interpolation_matrix(pos_good, pos_bad)

        assert result.data.shape == (3, 20)


class TestGPUDoInterpDots:
    """Tests for gpu_do_interp_dots function."""

    def test_interpolation_2d(self):
        """Test interpolation on 2D data (evoked-like)."""
        np.random.seed(42)
        n_channels, n_times = 64, 500
        n_bad = 5

        # Setup positions
        pos = np.random.randn(n_channels, 3)
        bads_idx = np.zeros(n_channels, dtype=bool)
        bads_idx[:n_bad] = True
        goods_idx = ~bads_idx

        # Create interpolation matrix
        interp = gpu_make_interpolation_matrix(pos[goods_idx], pos[bads_idx])

        # Create data
        data = np.random.randn(n_channels, n_times).astype(np.float32)

        # Apply interpolation
        result = gpu_do_interp_dots(
            data, interp, goods_idx, bads_idx, keep_on_device=False
        )

        assert result.shape == data.shape
        # Check that good channels are unchanged
        np.testing.assert_allclose(result[goods_idx], data[goods_idx])

    def test_interpolation_3d(self):
        """Test interpolation on 3D data (epochs-like)."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 10, 64, 500
        n_bad = 5

        # Setup positions
        pos = np.random.randn(n_channels, 3)
        bads_idx = np.zeros(n_channels, dtype=bool)
        bads_idx[:n_bad] = True
        goods_idx = ~bads_idx

        # Create interpolation matrix
        interp = gpu_make_interpolation_matrix(pos[goods_idx], pos[bads_idx])

        # Create data
        data = np.random.randn(n_epochs, n_channels, n_times).astype(np.float32)

        # Apply interpolation
        result = gpu_do_interp_dots(
            data, interp, goods_idx, bads_idx, keep_on_device=False
        )

        assert result.shape == data.shape
        # Check that good channels are unchanged
        np.testing.assert_allclose(result[:, goods_idx, :], data[:, goods_idx, :])

    def test_matches_mne_interpolation(self):
        """Test that GPU interpolation matches MNE's interpolation."""
        from mne.channels.interpolation import _make_interpolation_matrix

        np.random.seed(42)
        n_epochs, n_channels, n_times = 5, 32, 100
        n_bad = 3

        # Setup positions
        pos = np.random.randn(n_channels, 3)
        bads_idx = np.zeros(n_channels, dtype=bool)
        bads_idx[:n_bad] = True
        goods_idx = ~bads_idx

        # Create data
        data = np.random.randn(n_epochs, n_channels, n_times).astype(np.float32)

        # MNE interpolation
        mne_interp = _make_interpolation_matrix(pos[goods_idx], pos[bads_idx])
        data_mne = data.copy()
        for e in range(n_epochs):
            data_mne[e, bads_idx, :] = mne_interp @ data_mne[e, goods_idx, :]

        # GPU interpolation
        gpu_interp = gpu_make_interpolation_matrix(pos[goods_idx], pos[bads_idx])
        data_gpu = gpu_do_interp_dots(
            data, gpu_interp, goods_idx, bads_idx, keep_on_device=False
        )

        np.testing.assert_allclose(data_mne, data_gpu, rtol=1e-3, atol=1e-6)

    def test_keep_on_device(self):
        """Test that keep_on_device returns DeviceArray."""
        np.random.seed(42)
        n_channels, n_times = 32, 100
        n_bad = 3

        pos = np.random.randn(n_channels, 3)
        bads_idx = np.zeros(n_channels, dtype=bool)
        bads_idx[:n_bad] = True
        goods_idx = ~bads_idx

        interp = gpu_make_interpolation_matrix(pos[goods_idx], pos[bads_idx])
        data = np.random.randn(n_channels, n_times).astype(np.float32)

        result = gpu_do_interp_dots(
            data, interp, goods_idx, bads_idx, keep_on_device=True
        )

        assert is_device_array(result)
        assert result.backend.name == 'torch'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
