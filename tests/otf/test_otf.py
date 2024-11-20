import unittest.mock as mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Callable

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pybsm import otf
from pybsm.simulation import Scenario, Sensor
from tests.test_utils import CustomFloatSnapshotExtension

try:
    import cv2

    is_usable = True
except ImportError:
    is_usable = False


@pytest.fixture()
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(lambda: CustomFloatSnapshotExtension())


@pytest.mark.skipif(
    not is_usable,
    reason="OpenCV not found. Please install 'pybsm[graphics]' or `pybsm[headless]`.",
)
class TestOTFHelper:
    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2"),
        [
            (0.0, np.array([]), np.array([])),
            (0.0, np.array([]), np.array([1.0])),
            (1.0, np.array([]), np.array([])),
        ],
    )
    def test_coherence_diameter_value_error(
        self,
        lambda0: float,
        z_path: np.ndarray,
        cn2: np.ndarray,
    ) -> None:
        """Cover cases where ValueError occurs."""
        with pytest.raises(ValueError):  # noqa: PT011 - This just raises a ValueError
            otf.coherence_diameter(lambda0, z_path, cn2)

    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2"),
        [
            (0.0, np.array([1.0]), np.array([])),
            (0.0, np.array([1.0]), np.array([0.0])),
            (0.0, np.array([1.0]), np.array([1.0])),
        ],
    )
    def test_coherence_diameter_zero_division(
        self,
        lambda0: float,
        z_path: np.ndarray,
        cn2: np.ndarray,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.coherence_diameter(lambda0, z_path, cn2)

    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2"),
        [
            (1.0, np.array([1.0]), np.array([0.0])),
            (1.0, np.array([1.0, 2.0]), np.array([0.0])),
            (1.0, np.array([1.0]), np.array([1.0])),
            (1.0, np.array([2.0]), np.array([1.0])),
        ],
    )
    def test_coherence_diameter_infinite(
        self,
        lambda0: float,
        z_path: np.ndarray,
        cn2: np.ndarray,
    ) -> None:
        """Cover cases where infinite output occurs."""
        output = otf.coherence_diameter(lambda0, z_path, cn2)
        assert np.isinf(output)

    @pytest.mark.parametrize(
        ("lambda0", "z_path", "cn2"),
        [
            (1.0, np.array([1.0, 2.0]), np.array([1.0])),
            (2.0, np.array([1.0, 2.0]), np.array([1.0])),
            (1.0, np.array([1.0, 2.0]), np.array([2.0])),
            (1.0, np.array([1.0, 2.0, 3.0]), np.array([1.0])),
        ],
    )
    def test_coherence_diameter(
        self,
        lambda0: float,
        z_path: np.ndarray,
        cn2: np.ndarray,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test coherence_diameter with normal inputs and expected outputs."""
        output = otf.coherence_diameter(lambda0, z_path, cn2)
        snapshot_custom.assert_match(output)

    @pytest.mark.parametrize(
        ("h", "v", "cn2_at_1m"),
        [
            (np.array([]), 0.0, 0.0),
            (np.array([]), 1.0, 1.0),
        ],
    )
    def test_hufnagel_valley_turbulence_profile_empty_array(
        self,
        h: np.ndarray,
        v: float,
        cn2_at_1m: float,
    ) -> None:
        """Test hufnagel_valley_turbulence_profile with empty input."""
        output = otf.hufnagel_valley_turbulence_profile(h, v, cn2_at_1m)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("h", "v", "cn2_at_1m"),
        [
            (np.array([1.0]), 1.0, 0.0),
            (np.array([1.0]), 0.0, 1.0),
            (np.array([0.0]), 1.0, 1.0),
            (np.array([1.0]), 1.0, 1.0),
            (np.array([-1.0]), -1.0, -1.0),
            (np.array([1.0, 1.0]), 1.0, 0.0),
        ],
    )
    def test_hufnagel_valley_turbulence_profile(
        self,
        h: np.ndarray,
        v: float,
        cn2_at_1m: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test hufnagel_valley_turbulence_profile with normal inputs and expected outputs."""
        output = otf.hufnagel_valley_turbulence_profile(h, v, cn2_at_1m)
        snapshot_custom.assert_match(output)

    @pytest.mark.parametrize(
        ("wavelengths", "weights", "my_function"),
        [
            (np.array([]), np.array([]), lambda wavelengths: wavelengths),
            (np.array([]), np.array([0.0]), lambda wavelengths: wavelengths),
            (np.array([0.0]), np.array([]), lambda wavelengths: wavelengths),
            (np.array([1.0, 2.0]), np.array([1.0]), lambda wavelengths: wavelengths),
        ],
    )
    def test_weighted_by_wavelength_index_error(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        my_function: Callable,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.weighted_by_wavelength(wavelengths, weights, my_function)

    @pytest.mark.parametrize(
        ("wavelengths", "weights", "my_function"),
        [
            (np.array([0.0]), np.array([0.0]), lambda wavelengths: wavelengths),
            (np.array([1.0]), np.array([0.0]), lambda wavelengths: wavelengths),
            (
                np.array([1.0, 1.0]),
                np.array([0.0, 0.0]),
                lambda wavelengths: wavelengths,
            ),
        ],
    )
    def test_weighted_by_wavelength_nan(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        my_function: Callable,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.weighted_by_wavelength(wavelengths, weights, my_function)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("wavelengths", "weights", "my_function"),
        [
            (
                np.array([0.0]),
                np.array([1.0]),
                lambda wavelengths: wavelengths,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                lambda wavelengths: wavelengths,
            ),
            (
                np.array([1.0]),
                np.array([1.0, 2.0]),
                lambda wavelengths: wavelengths,
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                lambda wavelengths: wavelengths,
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                lambda wavelengths: wavelengths * 2,
            ),
        ],
    )
    def test_weighted_by_wavelength(
        self,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        my_function: Callable,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test weighted_by_wavelength with normal inputs and expected outputs."""
        output = otf.weighted_by_wavelength(wavelengths, weights, my_function)
        snapshot_custom.assert_match(output)

    @pytest.mark.parametrize(
        ("D", "R", "R0"),
        [
            (10.0, 1.0, 1.0),
            (0.0, 2.0, 1.0),
            (
                10.0,
                1.0,
                2.0,
            ),
            (
                10.0,
                2.0,
                1.0,
            ),
        ],
    )
    def test_object_domain_radii(
        self,
        D: float,  # noqa: N803
        R: float,  # noqa: N803
        R0: float,  # noqa: N803
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test object_domain_defocus_radii with normal inputs and expected outputs."""
        output = otf.object_domain_defocus_radii(D, R, R0)
        assert output == snapshot

    @pytest.mark.parametrize(
        ("D", "R", "R0", "expected"),
        [
            (1.0, 1.0, 0.0, pytest.raises(ZeroDivisionError)),
            (1.0, 0.0, 1.0, pytest.raises(ZeroDivisionError)),
            (1.0, 1.0, 1.0, does_not_raise()),
        ],
    )
    def test_object_domain_radii_zero_division(
        self,
        D: float,  # noqa: N803
        R: float,  # noqa: N803
        R0: float,  # noqa: N803
        expected: AbstractContextManager,
    ) -> None:
        with expected:
            otf.object_domain_defocus_radii(D, R, R0)

    @pytest.mark.parametrize(
        ("jd", "w_x", "w_y"),
        [
            (10.0, 0.0, 1.0),
            (10.0, 2.0, 0.0),
            (0.0, 2.0, 1.0),
            (
                1.0,
                1.0,
                2.0,
            ),
            (
                1.0,
                2.0,
                1.0,
            ),
        ],
    )
    def test_dark_current_from_density(
        self,
        jd: float,
        w_x: float,
        w_y: float,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test dark_current_from_density with normal inputs and expected outputs."""
        output = otf.dark_current_from_density(jd, w_x, w_y)
        assert output == snapshot

    @pytest.mark.parametrize(
        ("D", "dz", "f"),
        [
            (10.0, 0.0, 1.0),
            (0.0, 2.0, 1.0),
            (
                10.0,
                1.0,
                2.0,
            ),
            (
                10.0,
                2.0,
                1.0,
            ),
        ],
    )
    def test_image_domain_defocus_radii(
        self,
        D: float,  # noqa: N803
        dz: float,
        f: float,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test image_domain_defocus_radii with normal inputs and expected outputs."""
        output = otf.image_domain_defocus_radii(D, dz, f)
        assert output == snapshot

    @pytest.mark.parametrize(
        ("D", "dz", "f", "expected"),
        [(1.0, 1.0, 0.0, pytest.raises(ZeroDivisionError)), (1.0, 1.0, 1.0, does_not_raise())],
    )
    def test_image_domain_defocus_radii_zero_division(
        self,
        D: float,  # noqa: N803
        dz: float,
        f: float,
        expected: AbstractContextManager,
    ) -> None:
        with expected:
            otf.image_domain_defocus_radii(D, dz, f)


@pytest.mark.skipif(
    not is_usable,
    reason="OpenCV not found. Please install 'pybsm[graphics]' or `pybsm[headless]`.",
)
class TestResample2D:
    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out"),
        [
            (np.array([]), 1.0, 1.0),
            (np.array([1.0]), 1.0, 1.0),
        ],
    )
    def test_resample_2d_index_error(
        self,
        img_in: np.ndarray,
        dx_in: float,
        dx_out: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.resample_2D(img_in, dx_in, dx_out)

    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out"),
        [
            (np.ones((5, 5)), 0.0, 0.0),
            (np.ones((5, 5)), 1.0, 0.0),
        ],
    )
    def test_resample_2d_zero_division(
        self,
        img_in: np.ndarray,
        dx_in: float,
        dx_out: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.resample_2D(img_in, dx_in, dx_out)

    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out"),
        [
            (np.ones((5, 5)), 0.0, 1.0),
        ],
    )
    def test_resample_2d_cv2_error(
        self,
        img_in: np.ndarray,
        dx_in: float,
        dx_out: float,
    ) -> None:
        """Cover cases where cv2.error occurs."""
        with pytest.raises(cv2.error):
            otf.resample_2D(img_in, dx_in, dx_out)

    @pytest.mark.parametrize(
        ("img_in", "dx_in", "dx_out"),
        [
            (np.ones((5, 5)), 1.0, 1.0),
        ],
    )
    def test_resample_2d(
        self,
        img_in: np.ndarray,
        dx_in: float,
        dx_out: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test resample_2D with normal inputs and expected outputs."""
        output = otf.resample_2D(img_in, dx_in, dx_out)
        snapshot_custom.assert_match(output)


@pytest.mark.skipif(
    not is_usable,
    reason="OpenCV not found. Please install 'pybsm[graphics]' or `pybsm[headless]`.",
)
class TestApplyOTFToImage:
    @pytest.mark.parametrize(
        ("ref_img", "ref_gsd", "ref_range", "otf_value", "df", "ifov"),
        [
            (np.array([]), 0.0, 0.0, np.array([]), 0.0, 0.0),
            (np.array([]), 1.0, 0.0, np.array([]), 0.0, 0.0),
            (np.array([]), 0.0, 0.0, np.array([]), 1.0, 0.0),
            (np.array([]), 0.0, 0.0, np.array([]), 0.0, 1.0),
            (np.array([]), 1.0, 0.0, np.array([]), 1.0, 0.0),
            (np.array([]), 1.0, 0.0, np.array([]), 0.0, 1.0),
            (np.array([]), 1.0, 0.0, np.array([]), 1.0, 1.0),
            (np.ones((100, 100)), 0.0, 1.0, np.ones((100, 100)), 0.0, 0.0),
            (np.ones((100, 100)), 1.0, 1.0, np.ones((100, 100)), 1.0, 0.0),
        ],
    )
    def test_apply_otf_to_image_zero_division(
        self,
        ref_img: np.ndarray,
        ref_gsd: float,
        ref_range: float,
        otf_value: np.ndarray,
        df: float,
        ifov: float,
    ) -> None:
        """Cover cases where ZeroDivisionError occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.apply_otf_to_image(ref_img, ref_gsd, ref_range, otf_value, df, ifov)

    @pytest.mark.parametrize(
        ("ref_img", "ref_gsd", "ref_range", "otf_value", "df", "ifov"),
        [
            (np.array([]), 0.0, 1.0, np.array([]), 0.0, 0.0),
            (np.array([1.0]), 0.0, 1.0, np.array([]), 0.0, 0.0),
            (np.array([]), 0.0, 1.0, np.array([1.0]), 0.0, 0.0),
        ],
    )
    def test_apply_otf_to_image_index_error(
        self,
        ref_img: np.ndarray,
        ref_gsd: float,
        ref_range: float,
        otf_value: np.ndarray,
        df: float,
        ifov: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.apply_otf_to_image(ref_img, ref_gsd, ref_range, otf_value, df, ifov)

    @pytest.mark.parametrize(
        ("ref_img", "ref_gsd", "ref_range", "otf_value", "df", "ifov"),
        [
            (
                np.ones((100, 100)),
                1.0,
                1.0,
                np.ones((100, 100)),
                1.0,
                1.0,
            ),
        ],
    )
    def test_apply_otf_to_image(
        self,
        ref_img: np.ndarray,
        ref_gsd: float,
        ref_range: float,
        otf_value: np.ndarray,
        df: float,
        ifov: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test apply_otf_to_image with normal inputs and expected outputs."""
        output = otf.apply_otf_to_image(
            ref_img,
            ref_gsd,
            ref_range,
            otf_value,
            df,
            ifov,
        )
        snapshot_custom.assert_match(output)


@pytest.mark.skipif(
    not is_usable,
    reason="OpenCV not found. Please install 'pybsm[graphics]' or `pybsm[headless]`.",
)
class TestOTFToPSF:
    @pytest.mark.parametrize(
        ("otf_value", "df", "dx_out"),
        [
            (np.array([]), 0.0, 0.0),
            (np.array([0.0]), 0.0, 0.0),
        ],
    )
    def test_otf_to_psf_index_error(
        self,
        otf_value: np.ndarray,
        df: float,
        dx_out: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.otf_to_psf(otf_value, df, dx_out)

    @pytest.mark.parametrize(
        ("otf_value", "df", "dx_out"),
        [
            (np.ones((10, 10)), 0.0, 0.0),
            (np.ones((10, 10)), 0.0, 1.0),
            (np.ones((10, 10)), 1.0, 0.0),
        ],
    )
    def test_otf_to_psf_zero_division(
        self,
        otf_value: np.ndarray,
        df: float,
        dx_out: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.otf_to_psf(otf_value, df, dx_out)

    @pytest.mark.parametrize(
        ("otf_value", "df", "dx_out"),
        [
            (np.ones((15, 15)), 1.0, 1.0),
            (np.ones((100, 100)), 1.0, 1.0),
        ],
    )
    def test_otf_to_psf(
        self,
        otf_value: np.ndarray,
        df: float,
        dx_out: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test otf_to_psf with normal inputs and expected outputs."""
        output = otf.otf_to_psf(otf_value, df, dx_out)
        snapshot_custom.assert_match(output)


class TestCTEOTF:
    @pytest.mark.parametrize(
        ("u", "v", "p_x", "p_y", "cte_n_x", "cte_n_y", "phases_n", "cte_eff", "f"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 0, 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p_x: float,
        p_y: float,
        cte_n_x: float,
        cte_n_y: float,
        phases_n: int,
        cte_eff: float,
        f: float,
    ) -> None:
        """Test cte_OTF with empty input."""
        output = otf.cte_OTF(u, v, p_x, p_y, cte_n_x, cte_n_y, phases_n, cte_eff, f)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "p_x", "p_y", "cte_n_x", "cte_n_y", "phases_n", "cte_eff", "f"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 0, 0.0, 0.0),
            (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 1.0, 1.0, 1.0, 1, 1.0, 0.0),
        ],
    )
    def test_otf_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p_x: float,
        p_y: float,
        cte_n_x: float,
        cte_n_y: float,
        phases_n: int,
        cte_eff: float,
        f: float,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.cte_OTF(u, v, p_x, p_y, cte_n_x, cte_n_y, phases_n, cte_eff, f)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "p_x", "p_y", "cte_n_x", "cte_n_y", "phases_n", "cte_eff", "f"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 1.0, 1, 1.0, 1.0),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                0.0,
                0.0,
                0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                1.0,
                2.0,
                2,
                2.0,
                2.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1,
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1,
                1.0,
                1.0,
            ),
        ],
    )
    def test_otf(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p_x: float,
        p_y: float,
        cte_n_x: float,
        cte_n_y: float,
        phases_n: int,
        cte_eff: float,
        f: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test cte_OTF with normal inputs and expected outputs."""
        output = otf.cte_OTF(u, v, p_x, p_y, cte_n_x, cte_n_y, phases_n, cte_eff, f)
        snapshot_custom.assert_match(output)


class TestDefocusOTF:
    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y"),
        [
            (np.array([]), np.array([]), 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0),
            (np.array([]), np.array([1.0]), 1.0, 1.0),
        ],
    )
    def test_otf_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
    ) -> None:
        """Test defocus_OTF with empty input."""
        output = otf.defocus_OTF(u, v, w_x, w_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                2.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
            ),
        ],
    )
    def test_otf(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test defocus_OTF with normal inputs and expected outputs."""
        output = otf.defocus_OTF(u, v, w_x, w_y)
        snapshot_custom.assert_match(output)


class TestDetectorOTFWithAggregation:
    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "p_x", "p_y", "f", "n"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 1),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 1),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0, 0.0, 1),
            (np.array([]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 1.0, 1),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        p_x: float,
        p_y: float,
        f: float,
        n: int,
    ) -> None:
        """Test detector_OTF_with_aggregation with empty input."""
        output = otf.detector_OTF_with_aggregation(u, v, w_x, w_y, p_x, p_y, f, n)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "p_x", "p_y", "f", "n"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0, 0.0, 1),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 0.0, 1),
            (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 1.0, 1.0, 0.0, 0.0, 1),
        ],
    )
    def test_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        p_x: float,
        p_y: float,
        f: float,
        n: int,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.detector_OTF_with_aggregation(u, v, w_x, w_y, p_x, p_y, f, n)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "p_x", "p_y", "f", "n"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 1.0, 1.0, 1),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        p_x: float,
        p_y: float,
        f: float,
        n: int,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test detector_OTF_with_aggregation with normal inputs and expected outputs."""
        output = otf.detector_OTF_with_aggregation(u, v, w_x, w_y, p_x, p_y, f, n)
        snapshot_custom.assert_match(output)


class TestDiffusionOTF:
    @pytest.mark.parametrize(
        ("u", "v", "alpha", "ald", "al0", "f"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                2.0,
                2.0,
                1.0,
            ),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, 0.0),
            (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.0, 1.0, 1.0, 0.0),
        ],
    )
    def test_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        alpha: np.ndarray,
        ald: float,
        al0: float,
        f: float,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.diffusion_OTF(u, v, alpha, ald, al0, f)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "alpha", "ald", "al0", "f"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 1.0, 1.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 1.0, 1.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 1.0, 1.0),
            (np.array([]), np.array([1.0]), 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        alpha: np.ndarray,
        ald: float,
        al0: float,
        f: float,
    ) -> None:
        """Test diffusion_OTF with empty input."""
        output = otf.diffusion_OTF(u, v, alpha, ald, al0, f)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "alpha", "ald", "al0", "f"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                0.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                2.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        alpha: np.ndarray,
        ald: float,
        al0: float,
        f: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test cte_OTF with normal inputs and expected outputs."""
        output = otf.diffusion_OTF(u, v, alpha, ald, al0, f)
        snapshot_custom.assert_match(output)


class TestGaussianOTF:
    @pytest.mark.parametrize(
        ("u", "v", "blur_size_x", "blur_size_y"),
        [
            (np.array([]), np.array([]), 1.0, 1.0),
            (np.array([1.0]), np.array([]), 1.0, 1.0),
            (np.array([]), np.array([1.0]), 2.0, 1.0),
            (np.array([]), np.array([1.0]), 1.0, 2.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        blur_size_x: float,
        blur_size_y: float,
    ) -> None:
        """Test gaussian_OTF with empty input."""
        output = otf.gaussian_OTF(u, v, blur_size_x, blur_size_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "blur_size_x", "blur_size_y"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                2.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                2.0,
                2.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        blur_size_x: float,
        blur_size_y: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test cte_OTF with normal inputs and expected outputs."""
        output = otf.gaussian_OTF(u, v, blur_size_x, blur_size_y)
        snapshot_custom.assert_match(output)


class TestTdiOTF:
    @pytest.mark.parametrize(
        ("u_or_v", "w", "n_tdi", "phases_n", "beta", "f"),
        [
            (
                np.array([1.0]),
                0.0,
                2.0,
                1,
                1.0,
                0.0,
            ),
            (np.array([1.0]), 1.0, 1.0, 1.0, 0, 1.0),
            (np.array([1.0, 2.0]), 1.0, 1.0, 1.0, 1, 0.0),
        ],
    )
    def test_nan(
        self,
        u_or_v: np.ndarray,
        w: float,
        n_tdi: float,
        phases_n: int,
        beta: float,
        f: float,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.tdi_OTF(u_or_v, w, n_tdi, phases_n, beta, f)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u_or_v", "w", "n_tdi", "phases_n", "beta", "f"),
        [
            (np.array([]), 1.0, 1.0, 1.0, 1, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u_or_v: np.ndarray,
        w: float,
        n_tdi: float,
        phases_n: int,
        beta: float,
        f: float,
    ) -> None:
        """Test tdi_OTF with empty input."""
        output = otf.tdi_OTF(u_or_v, w, n_tdi, phases_n, beta, f)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u_or_v", "w", "n_tdi", "phases_n", "beta", "f"),
        [
            (
                np.array([1.0]),
                1000.0,
                0.5,
                1.0,
                30,
                1000.0,
            ),
            (
                np.array([1.0]),
                20.0,
                20.0,
                10.0,
                1,
                20.0,
            ),
            (
                np.array([1.0, 1.0]),
                10.0,
                10.0,
                10.0,
                10,
                10.0,
            ),
        ],
    )
    def test(
        self,
        u_or_v: np.ndarray,
        w: float,
        n_tdi: float,
        phases_n: int,
        beta: float,
        f: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test cte_OTF with normal inputs and expected outputs."""
        output = otf.tdi_OTF(u_or_v, w, n_tdi, phases_n, beta, f)
        snapshot_custom.assert_match(output)


class TestWavefrontOTF2:
    @pytest.mark.parametrize(
        ("u", "v", "cutoff", "w_rms"),
        [
            (np.array([]), np.array([]), 1.0, 1.0),
            (np.array([1.0]), np.array([]), 1.0, 1.0),
            (np.array([]), np.array([1.0]), 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        cutoff: float,
        w_rms: float,
    ) -> None:
        """Test wavefront_OTF_2 with empty input."""
        output = otf.wavefront_OTF_2(u, v, cutoff, w_rms)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "cutoff", "w_rms"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                0.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        cutoff: float,
        w_rms: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test cte_OTF with normal inputs and expected outputs."""
        output = otf.wavefront_OTF_2(u, v, cutoff, w_rms)
        snapshot_custom.assert_match(output)


class TestSliceOTF:
    @pytest.mark.parametrize(
        ("otf_input", "ang"),
        [
            (np.array([[1.0, 1.0], [1.0, 1.0]]), 0.0),
            (np.array([[1.0, 0.0], [1.0, 0.0]]), 10.0),
            (np.array([[1.0, 0.0], [1.0, 0.0]]), 180.0),
            (np.array([[1.0, 1.0], [1.0, 1.0]]), 360.0),
        ],
    )
    def test(
        self,
        otf_input: np.ndarray,
        ang: float,
        snapshot_custom: float,
    ) -> None:
        """Test slice_otf with normal inputs and expected outputs."""
        output = otf.slice_otf(otf_input, ang)
        snapshot_custom.assert_match(output)


class TestPolychromaticTurbulenceOTF:
    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test_zero_division(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.polychromatic_turbulence_OTF(
                u,
                v,
                wavelengths,
                weights,
                altitude,
                slant_range,
                D,
                ha_wind_speed,
                cn2_at_1m,
                int_time,
                aircraft_speed,
            )

    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test_index_error(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.polychromatic_turbulence_OTF(
                u,
                v,
                wavelengths,
                weights,
                altitude,
                slant_range,
                D,
                ha_wind_speed,
                cn2_at_1m,
                int_time,
                aircraft_speed,
            )

    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
        ),
        [
            (
                np.array([]),
                np.array([]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([]),
                np.array([]),
                np.array([2.0]),
                np.array([2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([]),
                np.array([]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test_first_array_empty(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test polychromatic_turbulence_OTF with empty input."""
        output = otf.polychromatic_turbulence_OTF(
            u,
            v,
            wavelengths,
            weights,
            altitude,
            slant_range,
            D,
            ha_wind_speed,
            cn2_at_1m,
            int_time,
            aircraft_speed,
        )
        assert output[0].size == 0
        snapshot_custom.assert_match(output[1])

    @pytest.mark.parametrize(
        (
            "u",
            "v",
            "wavelengths",
            "weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "aircraft_speed",
        ),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray,
        altitude: float,
        slant_range: float,
        D: float,  # noqa: N803
        ha_wind_speed: float,
        cn2_at_1m: float,
        int_time: float,
        aircraft_speed: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test polychromatic_turbulence_OTF with normal inputs and expected outputs."""
        output = otf.polychromatic_turbulence_OTF(
            u,
            v,
            wavelengths,
            weights,
            altitude,
            slant_range,
            D,
            ha_wind_speed,
            cn2_at_1m,
            int_time,
            aircraft_speed,
        )
        snapshot_custom.assert_match(output)


class TestDetectorOTF:
    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "f"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 1.0, 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        f: float,
    ) -> None:
        """Test detector_OTF with empty input."""
        output = otf.detector_OTF(u, v, w_x, w_y, f)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "f"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0),
            (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 1.0, 0.0),
        ],
    )
    def test_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        f: float,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.detector_OTF(u, v, w_x, w_y, f)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "w_x", "w_y", "f"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                0.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w_x: float,
        w_y: float,
        f: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test detector_OTF with normal inputs and expected outputs."""
        output = otf.detector_OTF(u, v, w_x, w_y, f)
        snapshot_custom.assert_match(output)


class TestDriftOTF:
    @pytest.mark.parametrize(
        ("u", "v", "a_x", "a_y"),
        [
            (np.array([]), np.array([]), 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        a_x: float,
        a_y: float,
    ) -> None:
        """Test drift_OTF with empty input."""
        output = otf.drift_OTF(u, v, a_x, a_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "a_x", "a_y"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.07),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.07),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.03),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        a_x: float,
        a_y: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test drift_OTF with normal inputs and expected outputs."""
        output = otf.drift_OTF(u, v, a_x, a_y)
        snapshot_custom.assert_match(output)


class TestJitterOTF:
    @pytest.mark.parametrize(
        ("u", "v", "s_x", "s_y"),
        [
            (np.array([]), np.array([]), 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        s_x: float,
        s_y: float,
    ) -> None:
        """Test jitter_OTF with empty input."""
        output = otf.jitter_OTF(u, v, s_x, s_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "s_x", "s_y"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                0.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                0.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        s_x: float,
        s_y: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test jitter_OTF with normal inputs and expected outputs."""
        output = otf.jitter_OTF(u, v, s_x, s_y)
        snapshot_custom.assert_match(output)


class TestCommonOTFs:
    def check_otf(
        self,
        otf: otf.OTF,
        ap_OTF: np.ndarray,  # noqa: N803
        turb_OTF: np.ndarray,  # noqa: N803
        r0_band: np.ndarray,
        det_OTF: np.ndarray,  # noqa: N803
        jit_OTF: np.ndarray,  # noqa: N803
        drft_OTF: np.ndarray,  # noqa: N803
        wav_OTF: np.ndarray,  # noqa: N803
        filter_OTF: np.ndarray,  # noqa: N803
        system_OTF: np.ndarray,  # noqa: N803
    ) -> None:
        """Internal function to check if OTF object's attributes match expected values."""
        assert np.isclose(otf.ap_OTF, ap_OTF).all()
        assert np.isclose(otf.turb_OTF, turb_OTF).all()
        assert np.isclose(otf.r0_band, r0_band).all()
        assert np.isclose(otf.det_OTF, det_OTF, atol=5e-34).all()
        assert np.isclose(otf.jit_OTF, jit_OTF).all()
        assert np.isclose(otf.drft_OTF, drft_OTF).all()
        assert np.isclose(otf.wav_OTF, wav_OTF).all()
        assert np.isclose(otf.filter_OTF, filter_OTF).all()
        assert np.isclose(otf.system_OTF, system_OTF).all()

    @pytest.mark.parametrize(
        (
            "sensor",
            "scenario",
            "uu",
            "vv",
            "mtf_wavelengths",
            "mtf_weights",
            "slant_range",
            "int_time",
        ),
        [
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                0.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
            ),
        ],
    )
    def test_common_otfs_zero_division(
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
        slant_range: float,
        int_time: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.common_OTFs(
                sensor,
                scenario,
                uu,
                vv,
                mtf_wavelengths,
                mtf_weights,
                slant_range,
                int_time,
            )

    @pytest.mark.parametrize(
        (
            "sensor",
            "scenario",
            "uu",
            "vv",
            "mtf_wavelengths",
            "mtf_weights",
            "slant_range",
            "int_time",
        ),
        [
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([1.0]),
                np.array([]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([]),
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                1.0,
                1.0,
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([]),
                np.array([1.0]),
                1.0,
                1.0,
            ),
        ],
    )
    def test_common_otfs_index_error(
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
        slant_range: float,
        int_time: float,
    ) -> None:
        """Cover cases where IndexError occurs."""
        with pytest.raises(IndexError):
            otf.common_OTFs(
                sensor,
                scenario,
                uu,
                vv,
                mtf_wavelengths,
                mtf_weights,
                slant_range,
                int_time,
            )

    @pytest.mark.parametrize(
        (
            "sensor",
            "scenario",
            "uu",
            "vv",
            "mtf_wavelengths",
            "mtf_weights",
            "slant_range",
            "int_time",
            "expected",
        ),
        [
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                {
                    "ap_OTF": np.array([0.0]),
                    "turb_OTF": np.array([1.0]),
                    "r0_band": np.array([60457834.264253505]),
                    "det_OTF": np.array([1.51957436e-33]),
                    "jit_OTF": np.array([1.0]),
                    "drft_OTF": np.array([1.0]),
                    "wav_OTF": np.array([1.0]),
                    "filter_OTF": np.array([1.0]),
                    "system_OTF": np.array([0.0]),
                },
            ),
            (
                Sensor("test_scene", 1.0, 1.0, 1.0, np.array([0.0, 1.0])),
                Scenario("test_scenario", 1, 1.0, 1.0),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                {
                    "ap_OTF": np.array([0.0, 0.0]),
                    "turb_OTF": np.array([1.0, 1.0]),
                    "r0_band": np.array([60457834.264253505, 60457834.264253505]),
                    "det_OTF": np.array([1.51957436e-33, 1.51957436e-33]),
                    "jit_OTF": np.array([1.0, 1.0]),
                    "drft_OTF": np.array([1.0, 1.0]),
                    "wav_OTF": np.array([1.0, 1.0]),
                    "filter_OTF": np.array([1.0, 1.0]),
                    "system_OTF": np.array([0.0, 0.0]),
                },
            ),
        ],
    )
    def test_common_otfs(
        self,
        sensor: Sensor,
        scenario: Scenario,
        uu: np.ndarray,
        vv: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
        slant_range: float,
        int_time: float,
        expected: dict[str, np.ndarray],
    ) -> None:
        """Test common_OTFs with normal inputs and expected outputs."""
        output = otf.common_OTFs(
            sensor,
            scenario,
            uu,
            vv,
            mtf_wavelengths,
            mtf_weights,
            slant_range,
            int_time,
        )
        self.check_otf(output, **expected)


class TestTurbulenceOTF:
    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
    ) -> None:
        """Test turbulence_OTF with empty input."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0, 1.0),
        ],
    )
    def test_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
    ) -> None:
        """Test turbulence_OTF where output is nan."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 1.0),
        ],
    )
    def test_inf(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
    ) -> None:
        """Test turbulence_OTF where output is inf."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        assert np.isinf(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "alpha"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, 0.0),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        alpha: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test turbulenceOTF with normal inputs and expected outputs."""
        output = otf.turbulence_OTF(u, v, lambda0, D, r0, alpha)
        snapshot_custom.assert_match(output)


class TestWindSpeedTurbulenceOTF:
    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
    ) -> None:
        """Test wind_speed_turbulence_OTF with empty input."""
        output = otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
    ) -> None:
        """Test wind_speed_turbulence_OTF where output is nan."""
        output = otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0, 1.0, 1.0),
        ],
    )
    def test_zero_division(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "r0", "td", "vel"),
        [
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
            ),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        r0: float,
        td: float,
        vel: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test wind_speed_turbulence_OTF with normal inputs and expected outputs."""
        output = otf.wind_speed_turbulence_OTF(u, v, lambda0, D, r0, td, vel)
        snapshot_custom.assert_match(output)


class TestFilterOTF:
    @pytest.mark.parametrize(
        ("u", "v", "kernel", "ifov"),
        [
            (np.array([]), np.array([]), np.array([[1.0]]), 1.0),
            (np.array([1.0]), np.array([]), np.array([[1.0]]), 1.0),
            (np.array([]), np.array([1.0]), np.array([[1.0]]), 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        kernel: float,
        ifov: float,
    ) -> None:
        """Test filter_OTF with empty input."""
        output = otf.filter_OTF(u, v, kernel, ifov)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "kernel", "ifov"),
        [
            (np.array([1.0]), np.array([1.0]), np.array([[1.0]]), 1.0),
            (np.array([1.0]), np.array([1.0]), np.array([[0.5, 0.5]]), 10.0),
            (
                np.array([10.0, 10.0]),
                np.array([0.5, 1.0]),
                np.array([[1.0]]),
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        kernel: float,
        ifov: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test turbulenceOTF with normal inputs and expected outputs."""
        output = otf.filter_OTF(u, v, kernel, ifov)
        snapshot_custom.assert_match(output)


class TestWavefrontOTF:
    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "pv", "L_x", "L_y"),
        [
            (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        pv: float,
        L_x: float,  # noqa: N803
        L_y: float,  # noqa: N803
    ) -> None:
        """Test wavefront_OTF with empty input."""
        output = otf.wavefront_OTF(u, v, lambda0, pv, L_x, L_y)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "pv", "L_x", "L_y"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0, 0.0),
        ],
    )
    def test_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        pv: float,
        L_x: float,  # noqa: N803
        L_y: float,  # noqa: N803
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.wavefront_OTF(u, v, lambda0, pv, L_x, L_y)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "pv", "L_x", "L_y"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 0.0),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                0.0,
                0.0,
            ),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 1.0),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                1.0,
                0.0,
            ),
            (
                np.array([1.0]),
                np.array([1.0]),
                1.0,
                1.0,
                0.0,
                1.0,
            ),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0, 1.0),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        pv: float,
        L_x: float,  # noqa: N803
        L_y: float,  # noqa: N803
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test wavefront_OTF with normal inputs and expected outputs."""
        output = otf.wavefront_OTF(u, v, lambda0, pv, L_x, L_y)
        snapshot_custom.assert_match(output)


class TestCircularApertureOTF:
    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta"),
        [
            (np.array([]), np.array([]), 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 1.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 1.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
    ) -> None:
        """Test circular_aperture_OTF with empty input."""
        output = otf.circular_aperture_OTF(u, v, lambda0, D, eta)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 1.0),
        ],
    )
    def test_zero_division(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.circular_aperture_OTF(u, v, lambda0, D, eta)

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 1.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 1.0),
        ],
    )
    def test_nan(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
    ) -> None:
        """Cover cases where nan output occurs."""
        output = otf.circular_aperture_OTF(u, v, lambda0, D, eta)
        assert np.isnan(output).all()

    @pytest.mark.parametrize(
        ("u", "v", "lambda0", "D", "eta"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                0.0,
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                0.5,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lambda0: float,
        D: float,  # noqa: N803
        eta: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test circular_aperture_OTF with normal inputs and expected outputs."""
        output = otf.circular_aperture_OTF(u, v, lambda0, D, eta)
        snapshot_custom.assert_match(output)


class TestCircularApertureOTFWithDefocus:
    @pytest.mark.parametrize(
        ("u", "v", "wavelength", "D", "f", "defocus"),
        [
            (np.array([]), np.array([]), 1.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([]), 1.0, 1.0, 0.0, 0.0),
            (np.array([]), np.array([1.0]), 1.0, 1.0, 0.0, 0.0),
            (np.array([]), np.array([]), 1.0, 1.0, 1.0, 0.0),
        ],
    )
    def test_empty_array(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelength: float,
        D: float,  # noqa: N803
        f: float,
        defocus: float,
    ) -> None:
        """Test circular_aperture_OTF_with_defocus with empty input."""
        output = otf.circular_aperture_OTF_with_defocus(u, v, wavelength, D, f, defocus)
        assert output.size == 0

    @pytest.mark.parametrize(
        ("u", "v", "wavelength", "D", "f", "defocus"),
        [
            (np.array([1.0]), np.array([1.0]), 0.0, 1.0, 0.0, 0.0),
            (np.array([1.0]), np.array([1.0]), 1.0, 0.0, 0.0, 0.0),
        ],
    )
    def test_zero_division(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelength: float,
        D: float,  # noqa: N803
        f: float,
        defocus: float,
    ) -> None:
        """Cover cases where ZeroDivision occurs."""
        with pytest.raises(ZeroDivisionError):
            otf.circular_aperture_OTF_with_defocus(u, v, wavelength, D, f, defocus)

    @pytest.mark.parametrize(
        ("u", "v", "wavelength", "D", "f", "defocus"),
        [
            (np.array([1.0]), np.array([1.0]), 1.0, 1.0, 0.0, 0.0),
            (
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                1.0,
                1.0,
                0.0,
                0.0,
            ),
            (
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                1.0,
                1.0,
                0.5,
                0.0,
            ),
        ],
    )
    def test(
        self,
        u: np.ndarray,
        v: np.ndarray,
        wavelength: float,
        D: float,  # noqa: N803
        f: float,
        defocus: float,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Test circular_aperture_OTF with normal inputs and expected outputs."""
        output = otf.circular_aperture_OTF_with_defocus(u, v, wavelength, D, f, defocus)
        snapshot_custom.assert_match(output)


@pytest.mark.skipif(
    not is_usable,
    reason="OpenCV not found. Please install 'pybsm[graphics]' or `pybsm[headless]`.",
)
@mock.patch("pybsm.otf.functional.is_usable", False)
def test_missing_deps() -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    with pytest.raises(ImportError, match=r"OpenCV not found"):
        otf.resample_2D(np.ones((5, 5)), 1.0, 1.0)
