"""The Python Based Sensor Model (pyBSM) is a collection of electro-optical camera modeling functions.

Developed by the Air Force Research Laboratory, Sensors Directorate.

Author citation:
LeMaster, Daniel A.; Eismann, Michael T., "pyBSM: A Python package for modeling
imaging systems", Proc. SPIE 10204 (2017)

Distribution A.  Approved for public release.
Public release approval for version 0.0: 88ABW-2017-3101
Public release approval for version 0.1: 88ABW-2018-5226

Maintainer: Kitware, Inc. <nrtk@kitware.com>
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, cast

import numba
import numpy as np
from scipy import interpolate
from scipy.ndimage import correlate
from scipy.signal import fftconvolve, oaconvolve

from pybsm import noise, radiance

# from pybsm.simulation.functional import img_to_reflectance
from pybsm.simulation.scenario import Scenario
from pybsm.simulation.sensor import Sensor

ConvolutionMethods = Literal["fftconvolve", "pad_fftconvolve", "correlate", "oaconvolve"]
ResampleBases = Literal["pixel-angle", "ground-angle"]


class ImageSimulator(ABC):
    """Base class for optical image simulation.

    This class performs the calculations necessary to apply optical transfer functions
    to images.

    Attributes:
        sensor: The sensor configuration for the perturbation.
        scenario: The scenario configuration used for perturbation.
        add_noise: Whether to apply noise after the psf is applied.
        rng: The random generator used to calculate noise.
        use_reflectance: Whether to use reflectance to convert to photoelectrons before
            applying convolution.
        reflectance_range: The reflectance range of values used to optionally convert
            image to photoelectrons.
        mtf_wavelengths: Wavelengths for MTF calculations.
        mtf_weights: Weights for MTF calculations.
        slant_range: Optional override for the slant_range.
        altitude: Optional override for the altitude.
    """

    def __init__(  # noqa: C901
        self,
        *,
        sensor: Sensor,
        scenario: Scenario,
        add_noise: bool = False,
        rng: np.random.Generator | int | None = 1,
        use_reflectance: bool = False,
        reflectance_range: np.ndarray | None = None,
        mtf_wavelengths: np.ndarray | None = None,
        mtf_weights: np.ndarray | None = None,
        slant_range: float | None = None,
        altitude: float | None = None,
        do_resample: bool = True,
    ) -> None:
        """Initialize the ImageSimulator base class.

        Args:
            sensor: Sensor configuration.
            scenario: pyBSM scenario configuration.
            add_noise: Whether to apply noise after PSF.
            rng: Random generator for noise calculation.
            use_reflectance: Whether to convert to photoelectrons using reflectance.
            reflectance_range: Reflectance range for conversion.
            mtf_wavelengths: Wavelengths for MTF calculations.
            mtf_weights: Weights for MTF calculations.
            slant_range: Optional slant range override.
            altitude: Optional altitude override.

        Raises:
            ValueError: If use_reflectance is True and reflectance_range was not provided.
            ValueError: If reflectance_range length != 2.
            ValueError: If reflectance_range not strictly ascending.
            ValueError: If mtf_wavelengths and mtf_weights are not equal length.
            ValueError: If mtf_wavelengths is empty or mtf_weights is empty.
        """
        if use_reflectance and reflectance_range is None:
            raise ValueError("Must provide refl_values when use_reflectance = True")

        if reflectance_range is not None:
            if reflectance_range.shape[0] != 2:
                raise ValueError(f"Reflectance range array must have length of 2, got {reflectance_range.shape[0]}")
            if reflectance_range[0] >= reflectance_range[1]:
                raise ValueError(f"Reflectance range array values must be strictly ascending, got {reflectance_range}")

        if mtf_wavelengths is not None and mtf_wavelengths.size == 0:
            raise ValueError("mtf_wavelengths is empty")

        if mtf_weights is not None and mtf_weights.size == 0:
            raise ValueError("mtf_weights is empty")

        if mtf_wavelengths is not None and mtf_weights is not None and mtf_wavelengths.size != mtf_weights.size:
            raise ValueError("mtf_wavelengths and mtf_weights are not the same length")

        """Compute config hash from sensor and scenario and compute common terms"""
        # Store deep copies to prevent external mutation affecting us
        self._sensor = copy.deepcopy(sensor)
        self._scenario = copy.deepcopy(scenario)

        self._use_reflectance = use_reflectance
        self._rng = np.random.default_rng(rng)

        # Noise simulation setup
        self._add_noise = add_noise
        if self._add_noise:
            quantization_noise = noise.quantization_noise(
                pe_range=self._sensor.max_n,
                bit_depth=self._sensor.bit_depth,
            )
            self._g_noise = np.sqrt(quantization_noise**2.0 + self._sensor.read_noise**2.0)

        if (mtf_wavelengths is None) or (self._use_reflectance):
            if reflectance_range is not None:
                self._reflectance_range: np.ndarray = reflectance_range
            ref, pe, spectral_weights = radiance.reflectance_to_photoelectrons(
                atm=self._scenario.atm,
                sensor=self._sensor,
                int_time=self._sensor.int_time,
            )

            if self._use_reflectance:
                self._reflect_to_photoelectrons: Callable = interpolate.interp1d(ref, pe)

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # Cut down the wavelength range to only the regions of interest
            pos_weights = weights > 0.0

            self._mtf_wavelengths: np.ndarray = wavelengths[pos_weights]
            self._mtf_weights: np.ndarray = weights[pos_weights]
        else:
            self._mtf_wavelengths: np.ndarray = np.asarray(mtf_wavelengths)
            self._mtf_weights: np.ndarray = np.asarray(mtf_weights)

        # Pre-compute common derived values
        self._altitude = altitude if altitude else self._scenario.altitude
        self._slant_range = (
            slant_range if slant_range else np.sqrt(self._scenario.altitude**2 + self._scenario.ground_range**2)
        )
        self._ifov = (self._sensor.p_x + self._sensor.p_y) / 2 / self._sensor.f
        self._cutoff_frequency = self._sensor.D / np.min(self._mtf_wavelengths)

        # Initialize spatial frequency grid (computed once)
        _u_rng = np.linspace(-1.0, 1.0, 1501) * self._cutoff_frequency
        _v_rng = np.linspace(1.0, -1.0, 1501) * self._cutoff_frequency
        self._uu, self._vv = np.meshgrid(_u_rng, _v_rng)
        self._df = (abs(_u_rng[1] - _u_rng[0]) + abs(_v_rng[0] - _v_rng[1])) / 2

        # PSF cache keyed by (config_hash, gsd_rounded | None)
        self._psf_cache: dict[tuple[int, float | None], np.ndarray] = dict()

        self.do_resample = do_resample

    def apply_convolution(self, image: np.ndarray, psf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa C901
        """Apply convolution using this simulator's method.

        Args:
            image: Input image array.
            psf: Point spread function kernel.

        Returns:
            Tuple of (true_img, blur_img) where true_img is the image converted to
            photoelectrons and blur_img is the convolved result.
        """
        true_img = image
        if self._use_reflectance:
            p1, p2 = image.min(), image.max()
            r1, r2 = self._reflectance_range
            scale = (r2 - r1) / (p2 - p1)
            reflectance_img = true_img.astype(np.float64)
            np.subtract(reflectance_img, p1, out=reflectance_img)
            np.multiply(reflectance_img, scale, out=reflectance_img)
            np.add(reflectance_img, r1, out=reflectance_img)
            np.clip(reflectance_img, 0, 1, out=reflectance_img)
            true_img = self._reflect_to_photoelectrons(reflectance_img)

        method = self._get_convolution_method()

        if method == "oaconvolve":
            # Correlation via convolution: flip kernel
            k = psf[::-1, ::-1]

            kh, kw = k.shape
            # Asymmetric reflect padding to match correlate
            pad_top = kh // 2
            pad_bottom = kh - 1 - pad_top
            pad_left = kw // 2
            pad_right = kw - 1 - pad_left
            pads = ((pad_top, pad_bottom), (pad_left, pad_right))

            if true_img.ndim == 2:
                img_temp = true_img.astype(np.float64, copy=False)
                img_pad = np.pad(img_temp, pads, mode="reflect")
                blur_img = oaconvolve(img_pad, k, mode="valid")
            else:
                blur_img = np.empty_like(true_img, dtype=np.float64)
                for c in range(true_img.shape[2]):
                    img_temp = true_img[..., c].astype(np.float64, copy=False)
                    img_pad = np.pad(img_temp, pads, mode="reflect")
                    blur_img[..., c] = oaconvolve(img_pad, k, mode="valid")

        elif method == "pad_fftconvolve":
            from scipy import fft

            # In the case of an even-length kernel, we need to pad the kernel to have
            # odd-length in order to match the behavior of `correlate`
            if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
                # pad by 1 in each dimension that has even length
                py = (psf.shape[0] + 1) % 2
                px = (psf.shape[1] + 1) % 2
                psf = np.pad(psf, ((py, 0), (px, 0)), constant_values=0.0)

            ky = psf.shape[0] // 2
            kx = psf.shape[1] // 2

            padding = ((ky, ky), (kx, kx))
            if true_img.ndim == 3:
                padding = padding + ((0, 0),)
            padded = np.pad(true_img, padding, mode="reflect")
            shape = tuple(s + k - 1 for s, k in zip(padded.shape[:2], psf.shape, strict=False))
            fshape = [fft.next_fast_len(s, real=True) for s in shape]

            # fft of kernel (only need to do once, instead of once per image channel
            # as in fftconvolve)
            psf_f = cast(np.ndarray, fft.rfft2(psf, fshape, axes=(0, 1)))
            # fft of image
            img_f = cast(np.ndarray, fft.rfft2(padded, fshape, axes=(0, 1)))

            if true_img.ndim == 3:
                psf_f = psf_f[..., None]

            # convolution
            np.multiply(img_f, psf_f, out=img_f)
            blur_img = cast(np.ndarray, fft.irfft2(img_f, fshape, axes=(0, 1)))

            # slice off padding
            blur_img = blur_img[2 * ky : 2 * ky + true_img.shape[0], 2 * kx : 2 * kx + true_img.shape[1], ...]
            blur_img = np.ascontiguousarray(blur_img)

        elif method == "fftconvolve":
            if true_img.ndim == 3:
                blur_img = np.empty_like(true_img, dtype=float)
                for c in range(blur_img.shape[2]):
                    blur_img[..., c] = fftconvolve(true_img[..., c], psf, mode="same")
            else:
                blur_img = fftconvolve(true_img, psf, mode="same")

        elif method == "correlate":
            if true_img.ndim == 3:
                blur_img = np.empty_like(true_img, dtype=float)
                for c in range(blur_img.shape[2]):
                    blur_img[..., c] = correlate(true_img[..., c], psf, mode="mirror")
            else:
                blur_img = correlate(true_img, psf, mode="mirror")
        else:
            raise ValueError(f"Unknown convolution method: {method}")

        return true_img, blur_img

    def _calculate_dx_out(self, gsd: float) -> float:
        resample_basis = self._get_resample_basis()

        if resample_basis == "pixel-angle":
            dx_out = self._ifov
        elif resample_basis == "ground-angle":
            dx_out = gsd / self._altitude
        else:
            raise ValueError(f"Unknown resample basis: {resample_basis}")

        return dx_out

    def apply_resampling(self, image: np.ndarray, gsd: float) -> np.ndarray:
        """Apply resampling based on sensor parameters.

        Args:
            image: Input image array.
            gsd: Ground sample distance.

        Returns:
            Resampled image array.

        Raises:
            ValueError: If resample basis is unknown.
        """
        from PIL import Image

        from pybsm.otf.functional import resampled_dimensions

        dx_in = gsd / self.slant_range

        dx_out = self._calculate_dx_out(gsd=gsd)
        new_y, new_x = resampled_dimensions(img_hw=(image.shape[0], image.shape[1]), dx_in=dx_in, dx_out=dx_out)
        mode = Image.Resampling.BILINEAR
        if image.dtype == np.uint8:
            if image.ndim == 3:
                sim_img = np.array(Image.fromarray(image).resize((new_x, new_y), mode))
            else:
                sim_img = np.array(Image.fromarray(image, "L").resize((new_x, new_y), mode))
        else:
            # for floating point images, we need to handle each channel as a separate
            # single-channel PIL floating point image
            shape = (new_y, new_x, 3 if image.ndim == 3 else 1)
            sim_img = np.empty(shape, dtype=image.dtype)
            if image.ndim == 2:
                image = image[..., None]
            for i in range(shape[2]):
                pil_img = Image.fromarray(image[..., i].astype("f"), "F")
                sim_img[..., i] = np.array(pil_img.resize((new_y, new_x), mode))
            sim_img = np.squeeze(sim_img)

        return sim_img

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise if enabled.

        Args:
            image: Input image array.

        Returns:
            Image with noise applied if add_noise is True, otherwise original image.
        """
        if not self.add_noise:
            return image

        # poisson_noisy_img = self._rng.poisson(lam=image)
        # return self._rng.normal(poisson_noisy_img, self._g_noise)

        seeds = self._rng.integers(0, 1 << 63, image.shape[0], dtype=np.uint64)
        if image.ndim == 2:
            return _apply_noise2d(image, self._g_noise, seeds)
        return _apply_noise3d(image, self._g_noise, seeds)

    def simulate_image(self, image: np.ndarray, gsd: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Apply the convolution and optionally resampling if gsd is provided.

        Args:
            image: Input image array.
            gsd: Ground sample distance. If None, no resampling is applied.

        Returns:
            Tuple of (true_img, blur_img, noisy_img) where noisy_img is None if
            add_noise is False.
        """
        psf = self._get_psf_cached(gsd=gsd, use_default=(gsd is None))

        true_img, blur_img = self.apply_convolution(image, psf)
        if gsd and self.do_resample:
            blur_img = self.apply_resampling(blur_img, gsd)

        noisy_img = None
        if self.add_noise:
            noisy_img = self.apply_noise(blur_img)

        return true_img, blur_img, noisy_img

    @property
    def mtf_wavelengths(self) -> np.ndarray:
        """Getter for _mtf_wavelengths"""
        return self._mtf_wavelengths

    @property
    def mtf_weights(self) -> np.ndarray:
        """Getter for _mtf_weights"""
        return self._mtf_weights

    @property
    def uu(self) -> np.ndarray:
        """Getter for _uu"""
        return self._uu

    @property
    def vv(self) -> np.ndarray:
        """Getter for _vv"""
        return self._vv

    @property
    def slant_range(self) -> float:
        """Getter for _slant_range"""
        return self._slant_range

    @property
    def add_noise(self) -> bool:
        """Getter for _add_noise"""
        return self._add_noise

    @property
    def sensor(self) -> Sensor:
        """Getter for _sensor"""
        return self._sensor

    @property
    def scenario(self) -> Scenario:
        """Getter for _scenario"""
        return self._scenario

    @abstractmethod
    def _compute_otf(self) -> np.ndarray:
        """Compute the OTF."""
        pass

    @abstractmethod
    def _get_convolution_method(self) -> ConvolutionMethods:
        """Return the convolution method this simulator should use."""
        pass

    @abstractmethod
    def _get_resample_basis(self) -> ResampleBases:
        """Return the resample basis used to define dx_out in resample2D"""
        pass

    def _get_config_hash(self) -> int:
        """Get hash representing the current sensor/scenario configuration."""
        return hash((hash(self.sensor), hash(self.scenario)))

    def _get_psf(self, gsd: float) -> np.ndarray:
        from pybsm.otf.functional import otf_to_psf

        """Compute PSF for given GSD"""
        otf = self._compute_otf()
        dx_out = 2 * np.arctan(gsd / 2 / self.slant_range)
        return otf_to_psf(otf=otf, df=self._df, dx_out=dx_out)

    def _get_default_psf(self) -> np.ndarray:
        from pybsm.otf.functional import otf_to_psf

        """Compute default PSF for when ifov/slant_range are invalid."""
        otf = self._compute_otf()
        dx_out = 1.0 / (otf.shape[0] * self._df)
        return otf_to_psf(otf=otf, df=self._df, dx_out=dx_out)

    def _get_psf_cached(self, gsd: float | None = None, use_default: bool = False) -> np.ndarray:
        """Get PSF with caching based on simulator configuration and GSD.

        Args:
            gsd: Ground sample distance. If None and use_default=True, uses default PSF.
            use_default: If True, forces use of default PSF regardless of GSD value.

        Returns:
            The cached or computed PSF array.
        """
        # Determine cache key based on use_default flag and GSD
        if use_default or gsd is None:  # noqa: SIM108
            gsd_rounded = None
        else:
            # Round GSD to 6 decimal places (micrometer precision)
            gsd_rounded = round(gsd, 6)

        config_hash = self._get_config_hash()
        cache_key = (config_hash, gsd_rounded)

        if cache_key not in self._psf_cache:
            if use_default or gsd is None:  # noqa: SIM108
                psf = self._get_default_psf()
            else:
                psf = self._get_psf(gsd)
            self._psf_cache[cache_key] = psf

        return self._psf_cache[cache_key]


class SystemOTFSimulator(ImageSimulator):
    """Simulator using common_OTFs with optional noise."""

    def _compute_otf(self) -> np.ndarray:
        from pybsm import otf

        return otf.common_OTFs(
            sensor=self.sensor,
            scenario=self.scenario,
            uu=self.uu,
            vv=self.vv,
            mtf_wavelengths=self.mtf_wavelengths,
            mtf_weights=self.mtf_weights,
            slant_range=self.slant_range,
            int_time=self.sensor.int_time,
        ).system_OTF

    def _get_convolution_method(self) -> ConvolutionMethods:
        # return "correlate"
        return "pad_fftconvolve"

    def _get_resample_basis(self) -> ResampleBases:
        return "pixel-angle"


class JitterSimulator(ImageSimulator):
    """Simulator for jitter-based optical effects."""

    def _compute_otf(self) -> np.ndarray:
        from pybsm.otf.functional import jitter_OTF

        return jitter_OTF(
            u=self.uu,
            v=self.vv,
            s_x=self.sensor.s_x,
            s_y=self.sensor.s_y,
        )

    def _get_convolution_method(self) -> ConvolutionMethods:
        return "oaconvolve"

    def _get_resample_basis(self) -> ResampleBases:
        return "pixel-angle"


class CircularApertureSimulator(ImageSimulator):
    """Simulator for circular aperture diffraction effects."""

    def _compute_otf(self) -> np.ndarray:
        from pybsm.otf.functional import circular_aperture_OTF, weighted_by_wavelength

        # Apply wavelength weighting using existing pybsm function
        def ap_function(wavelength: float) -> np.ndarray:
            return circular_aperture_OTF(
                u=self.uu,
                v=self.vv,
                lambda0=wavelength,
                D=self.sensor.D,
                eta=self.sensor.eta,
            )

        return weighted_by_wavelength(
            wavelengths=self.mtf_wavelengths,
            weights=self.mtf_weights,
            my_function=ap_function,
        )

    def _get_convolution_method(self) -> ConvolutionMethods:
        return "oaconvolve"

    def _get_resample_basis(self) -> ResampleBases:
        return "pixel-angle"


class DetectorSimulator(ImageSimulator):
    """Simulator for detector spatial integration effects."""

    def _compute_otf(self) -> np.ndarray:
        from pybsm.otf.functional import detector_OTF

        return detector_OTF(
            u=self.uu,
            v=self.vv,
            w_x=self.sensor.w_x,
            w_y=self.sensor.w_y,
            f=self.sensor.f,
        )

    def _get_convolution_method(self) -> ConvolutionMethods:
        return "oaconvolve"

    def _get_resample_basis(self) -> ResampleBases:
        return "pixel-angle"


class DefocusSimulator(ImageSimulator):
    """Simulator for defocus blur effects."""

    def _compute_otf(self) -> np.ndarray:
        from pybsm.otf.functional import defocus_OTF

        return defocus_OTF(
            u=self.uu,
            v=self.vv,
            w_x=self.sensor.w_x,
            w_y=self.sensor.w_y,
        )

    def _get_convolution_method(self) -> ConvolutionMethods:
        return "fftconvolve"

    def _get_resample_basis(self) -> ResampleBases:
        return "pixel-angle"


class TurbulenceApertureSimulator(ImageSimulator):
    """Simulator for atmospheric turbulence and aperture effects."""

    def _compute_otf(self) -> np.ndarray:
        from pybsm.otf.functional import polychromatic_turbulence_OTF

        turbulence_otf, _ = polychromatic_turbulence_OTF(
            u=self.uu,
            v=self.vv,
            wavelengths=self.mtf_wavelengths,
            weights=self.mtf_weights,
            altitude=self._altitude,
            slant_range=self.slant_range,
            D=self.sensor.D,
            ha_wind_speed=self.scenario.ha_wind_speed,
            cn2_at_1m=self.scenario.cn2_at_1m,
            int_time=self.sensor.int_time * self.sensor.n_tdi,
            aircraft_speed=self.scenario.aircraft_speed,
        )
        return turbulence_otf

    def _get_convolution_method(self) -> ConvolutionMethods:
        return "oaconvolve"

    def _get_resample_basis(self) -> ResampleBases:
        return "ground-angle"


@numba.njit("(int32,)")
def _seed_numba_rng(seed: int) -> None:
    np.random.seed(seed)  # noqa: NPY002


@numba.njit("float64[:, :, :](float64[:, :, :], float64, uint64[:])", fastmath=True, parallel=True, cache=True)
def _apply_noise3d(img: np.ndarray, g_noise: float, seeds: np.ndarray) -> np.ndarray:
    out = np.empty_like(img)
    for i in numba.prange(out.shape[0]):
        _seed_numba_rng(seeds[i])
        for j in range(out.shape[1]):
            for c in range(out.shape[2]):
                out[i, j, c] = np.random.normal(np.random.poisson(img[i, j, c]), g_noise)  # noqa: NPY002
    return out


@numba.njit("float64[:, :](float64[:, :], float64, uint64[:])", fastmath=True, parallel=True, cache=True)
def _apply_noise2d(img: np.ndarray, g_noise: float, seeds: np.ndarray) -> np.ndarray:
    out = np.empty_like(img)
    for i in numba.prange(out.shape[0]):
        _seed_numba_rng(seeds[i])
        for j in range(out.shape[1]):
            out[i, j] = np.random.normal(np.random.poisson(img[i, j]), g_noise)  # noqa: NPY002
    return out
