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
import warnings
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
            raise ValueError("Must provide reflectance_range when use_reflectance=True")

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
        # Sensor and scenario are set-once (no public setter, deep-copied above),
        # so the PSF cache key never changes — compute the hash once instead of
        # rehashing on every _get_psf_cached lookup.
        self._config_hash = hash((hash(self._sensor), hash(self._scenario)))

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
                self._init_inverse_coefs(ref, pe)
                # Once-per-instance warning flags reset only at __init__,
                # never on cache rebuild.
                self._inverse_clip_warned: bool = False
                self._inverse_fallback_warned: bool = False

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # Cut down the wavelength range to only the regions of interest
            pos_weights = weights > 0.0

            self._mtf_wavelengths: np.ndarray = wavelengths[pos_weights]
            self._mtf_weights: np.ndarray = weights[pos_weights]
        else:
            self._mtf_wavelengths: np.ndarray = np.asarray(mtf_wavelengths)
            self._mtf_weights: np.ndarray = np.asarray(mtf_weights)

        # Always-readable last-call clip-fraction state, regardless of `use_reflectance`.
        self._last_clip_fraction: tuple[float, float] = (0.0, 0.0)

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
            Tuple of (true_img, blur_img) where ``true_img`` is the image
            converted to photoelectrons when ``use_reflectance=True``
            (otherwise passed through), and ``blur_img`` is the convolved
            result.

        Note:
            With ``use_reflectance=True`` the input pixel range is first
            mapped to reflectance and then to photoelectrons. To go the other
            way (e.g. to display the result), use
            ``photoelectrons_to_reflectance`` or ``photoelectrons_to_pixels``.

            A uniform input image (``image.min() == image.max()``) cannot be
            stretched to the reflectance range, so every pixel is mapped to
            the midpoint of ``reflectance_range``. This keeps the result
            finite, but a uniform input does *not* round-trip to its
            original pixel value — its output depends on the sensor
            configuration, not on the input.
        """
        true_img = image
        if self._use_reflectance:
            p1, p2 = image.min(), image.max()
            r1, r2 = self._reflectance_range
            reflectance_img = true_img.astype(np.float64)
            if p2 == p1:
                # Uniform input: scale is undefined. Map every pixel to the
                # midpoint of the reflectance range so the forward path stays
                # finite (without this, scale -> inf and NaN propagates into
                # the FFT, hanging the simulator on synthetic flat inputs).
                reflectance_img.fill((float(r1) + float(r2)) / 2.0)
            else:
                scale = (r2 - r1) / (p2 - p1)
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

    def _apply_resampling_uint8(self, image: np.ndarray, new_wh: tuple[int, int], mode: int) -> np.ndarray:
        from PIL import Image

        array_mode = None if image.ndim == 3 else "L"
        return np.array(Image.fromarray(image, array_mode).resize(new_wh, mode))

    def _apply_resampling_float(self, image: np.ndarray, new_wh: tuple[int, int], mode: int) -> np.ndarray:
        from PIL import Image

        # for floating point images, we need to handle each channel as a separate
        # single-channel PIL floating point image
        shape = (new_wh[1], new_wh[0], 3 if image.ndim == 3 else 1)
        sim_img = np.empty(shape, dtype=image.dtype)
        if image.ndim == 2:
            image = image[..., None]
        for i in range(shape[2]):
            pil_img = Image.fromarray(image[..., i].astype("f"), "F")
            sim_img[..., i] = np.array(pil_img.resize(new_wh, mode))
        if sim_img.shape[-1] == 1:
            sim_img = np.squeeze(sim_img, axis=-1)

        return sim_img

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
        new_wh = resampled_dimensions(img_hw=(image.shape[0], image.shape[1]), dx_in=dx_in, dx_out=dx_out)
        mode = Image.Resampling.BILINEAR
        if image.dtype == np.uint8:
            return self._apply_resampling_uint8(image=image, new_wh=new_wh, mode=mode)
        return self._apply_resampling_float(image=image, new_wh=new_wh, mode=mode)

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise if enabled.

        Args:
            image: Input image array.

        Returns:
            Image with noise applied if add_noise is True, otherwise original image.

        Raises:
            RuntimeError: If ``image`` contains any ``NaN`` or ``Inf`` value
                while ``add_noise`` is enabled. Non-finite inputs are not
                supported; the error message reports the counts and shape
                so the caller can locate them upstream.
        """
        if not self.add_noise:
            return image

        if not np.all(np.isfinite(image)):
            raise RuntimeError(
                f"ImageSimulator.apply_noise: non-finite image "
                f"({int(np.isnan(image).sum())} NaN, "
                f"{int(np.isinf(image).sum())} Inf, shape={image.shape}). "
                f"Clean or filter non-finite values out of the input "
                f"before calling apply_noise.",
            )

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
            add_noise is False. When ``use_reflectance=True``, all three arrays
            are in photoelectron units; pass the chosen array through
            ``photoelectrons_to_pixels`` to recover display pixels. When
            ``use_reflectance=False``, the arrays remain in the input's native
            pixel units (with Poisson noise applied to those values directly);
            ``photoelectrons_to_pixels`` will route through the sensor's ADC
            model.

        Note:
            One simulator instance can be reused across input frames of
            different shapes — the PSF cache and noise kernels do not
            depend on input shape.

            If a pixel of the input is non-finite, ``apply_noise`` raises
            ``RuntimeError`` instead of hanging silently. See ``apply_noise``
            for the rationale.
        """
        psf = self._get_psf_cached(gsd=gsd, use_default=(gsd is None))

        true_img, blur_img = self.apply_convolution(image, psf)
        if gsd and self.do_resample:
            blur_img = self.apply_resampling(blur_img, gsd)

        noisy_img = None
        if self.add_noise:
            noisy_img = self.apply_noise(blur_img)

        return true_img, blur_img, noisy_img

    def photoelectrons_to_reflectance(
        self,
        photoelectrons: np.ndarray,
        *,
        clip_to_unit: bool = True,
    ) -> np.ndarray:
        """Convert photoelectrons back to reflectance using the cached inverse formula.

        Computes ``ref = (pe - B) / A`` where ``(A, B)`` are cached at
        ``__init__`` from the first and last points of the forward grid.

        Precision: the inverse matches the forward map to machine
        precision (~1e-13 relative) only when the forward map is linear
        in reflectance — i.e. ``pe = A * ref + B`` for some constants
        ``A`` and ``B``. This is the case for pyBSM's built-in forward
        chain. If a caller replaces the forward map with a non-linear
        one, the endpoint-fit ``(A, B)`` may not match interior grid
        points and residuals can be arbitrarily large.

        The method detects the most common non-linear case — a
        non-monotonic forward map (e.g. saturation flattening the tail)
        — and emits a once-per-instance warning. A smooth non-linear
        monotonic forward map will silently degrade precision; that case
        is caller responsibility.

        Args:
            photoelectrons: Photoelectron array of any shape. Float-coerced
                internally.
            clip_to_unit: If True (default), clip the output to ``[0, 1]``.
                Photoelectron inputs that fall outside the forward grid
                bounds map to reflectance values outside ``[0, 1]`` without
                this clip.

        Returns:
            Reflectance array of the same shape as ``photoelectrons``, dtype
            ``float64``.

        Raises:
            ValueError: If ``use_reflectance=False``. This method requires
                reflectance mode; for raw-pixel mode use
                ``photoelectrons_to_pixels`` (the ADC path).
            ValueError: If the sensor's radiometric calibration is flat —
                every reflectance produces the same photoelectron count,
                so the inverse is undefined. Check the sensor/scenario
                inputs to ``ImageSimulator``.
        """
        if not self._use_reflectance:
            raise ValueError(
                "ImageSimulator.photoelectrons_to_reflectance: requires "
                "use_reflectance=True; for raw-pixel mode use "
                "photoelectrons_to_pixels (ADC path).",
            )
        # Non-monotonic forward grid means the endpoint-fit (A, B) does not
        # match interior points; warn once but still return the endpoint-fit
        # result. (The pixel-domain path falls back to ADC in this case;
        # here there is no fallback, so the caller is responsible.)
        if not self._forward_is_monotonic:
            self._warn_inverse_precision_degraded(reason="forward_nonmonotonic")
        affine_a, affine_b = self._affine_pe_coefs
        if affine_a == 0.0:
            raise ValueError(
                "ImageSimulator.photoelectrons_to_reflectance: cannot invert "
                "because the sensor's radiometric calibration is flat — "
                "every scene reflectance maps to the same photoelectron "
                "count, so the input is not recoverable. Check the sensor "
                "and scenario configuration passed to ImageSimulator.",
            )
        ref = (np.asarray(photoelectrons, dtype=np.float64) - affine_b) / affine_a
        if clip_to_unit:
            np.clip(ref, 0.0, 1.0, out=ref)
        return ref

    def photoelectrons_to_pixels(
        self,
        photoelectrons_img: np.ndarray,
        p1: float = 0.0,
        p2: float = 255.0,
        *,
        mode: Literal["radiometric", "minmax"] = "radiometric",
    ) -> np.ndarray:
        """Convert a photoelectron array into display pixels in ``[p1, p2]``.

        Two mapping modes:

        - ``mode="radiometric"`` (default): sensor-calibrated. With
          ``use_reflectance=True``, the forward photoelectron-to-reflectance
          map is inverted analytically and reflectance is then remapped into
          ``[p1, p2]``. With ``use_reflectance=False``, the same pe array is
          digitized through the sensor's ADC (well capacity + bit-depth
          quantization) before being remapped. Either way, the output
          depends on the sensor, not on the per-image min/max — so two
          sensors viewing the same scene produce comparable pixels. Choose
          this for cross-sensor comparison or comparison against a reference.
        - ``mode="minmax"``: per-image histogram stretch into ``[p1, p2]``.
          Ignores the sensor entirely. Use this for side-by-side qualitative
          comparisons (e.g. PSF / blur kernels on a fixed scene) where
          radiometric fidelity is irrelevant. **Not** suitable for
          cross-sensor comparison.

        Args:
            photoelectrons_img: Photoelectron array.
            p1: Lower bound of the output pixel range. Default 0.0.
            p2: Upper bound of the output pixel range. Default 255.0.
            mode: ``"radiometric"`` (default) or ``"minmax"``. See above for
                the trade-offs.

        Returns:
            Pixel array of the same shape as ``photoelectrons_img``, dtype
            ``float64``. The caller is responsible for any final ``uint8``
            cast.

        Raises:
            ValueError: If the output range is not strictly ascending
                (``p1 >= p2``), or if ``mode`` is not one of
                ``"radiometric"`` / ``"minmax"``.
            RuntimeError: If ``photoelectrons_img`` contains ``NaN`` or
                ``Inf`` — fail fast rather than silently cast non-finite
                values to 0.

        Note:
            On the radiometric path, photoelectrons outside the forward
            grid bounds are clipped to those bounds. A once-per-instance
            warning fires when more than 1% of the input is clipped. The
            output is also clipped to ``[p1, p2]`` so a subsequent
            ``uint8`` cast does not wrap around.
        """
        self._validate_pe_to_pixels_args(photoelectrons_img, p1, p2, mode)

        # Empty input: short-circuit before any per-image min/max or
        # clip-fraction division by size.
        if photoelectrons_img.size == 0:
            return np.empty(photoelectrons_img.shape, dtype=np.float64)

        if mode == "minmax":
            return self._photoelectrons_to_pixels_minmax(photoelectrons_img, p1, p2)
        if self._use_reflectance:
            return self._photoelectrons_to_pixels_radiometric(photoelectrons_img, p1, p2)
        return self._adc_photoelectrons_to_pixels(photoelectrons_img, out_range=(p1, p2))

    @staticmethod
    def _validate_pe_to_pixels_args(
        photoelectrons_img: np.ndarray,
        p1: float,
        p2: float,
        mode: str,
    ) -> None:
        """Raise on bad inputs to ``photoelectrons_to_pixels`` before any computation.

        Order of checks matches the public method's documented contract:
        output range, mode literal, then non-finite values.
        """
        if p1 >= p2:
            raise ValueError(
                f"ImageSimulator.photoelectrons_to_pixels: output range must be strictly ascending; "
                f"got p1={p1!r}, p2={p2!r}.",
            )
        if mode not in {"radiometric", "minmax"}:
            raise ValueError(
                f"ImageSimulator.photoelectrons_to_pixels: mode must be 'radiometric' or 'minmax'; got mode={mode!r}.",
            )
        if not np.all(np.isfinite(photoelectrons_img)):
            raise RuntimeError(
                f"ImageSimulator.photoelectrons_to_pixels: non-finite "
                f"photoelectrons_img ({int(np.isnan(photoelectrons_img).sum())} NaN, "
                f"{int(np.isinf(photoelectrons_img).sum())} Inf, "
                f"shape={photoelectrons_img.shape}). Clean or filter "
                f"non-finite values out of the input before calling "
                f"photoelectrons_to_pixels.",
            )

    def _photoelectrons_to_pixels_minmax(
        self,
        photoelectrons_img: np.ndarray,
        p1: float,
        p2: float,
    ) -> np.ndarray:
        """Stretch ``photoelectrons_img`` linearly into ``[p1, p2]`` from its own min/max.

        Sensor-agnostic. Uniform input maps every pixel to the midpoint of
        ``[p1, p2]`` so the result stays finite (matches the uniform-input
        convention in ``apply_convolution``).
        """
        pe_min = float(photoelectrons_img.min())
        pe_max = float(photoelectrons_img.max())
        if pe_max == pe_min:
            return np.full(photoelectrons_img.shape, (p1 + p2) / 2.0, dtype=np.float64)
        pixels = (photoelectrons_img.astype(np.float64) - pe_min) / (pe_max - pe_min) * (p2 - p1) + p1
        return np.clip(pixels, p1, p2)

    def _photoelectrons_to_pixels_radiometric(
        self,
        photoelectrons_img: np.ndarray,
        p1: float,
        p2: float,
    ) -> np.ndarray:
        """Apply the inverse formula and remap reflectance into ``[p1, p2]``.

        If the forward grid is non-monotonic (saturation flattened the tail),
        the inverse formula is not reliable; fall back to the ADC path and
        warn once. Otherwise: record the clip fraction (warning once if it
        exceeds 1%), clip to the forward grid endpoints, invert to
        reflectance, and remap from ``reflectance_range`` into ``[p1, p2]``.
        """
        if not self._forward_is_monotonic:
            self._warn_inverse_fallback(reason="forward_nonmonotonic")
            return self._adc_photoelectrons_to_pixels(photoelectrons_img, out_range=(p1, p2))

        self._record_clip_fraction(photoelectrons_img)
        clipped = np.clip(photoelectrons_img, self._fwd_y_min, self._fwd_y_max)
        ref = self.photoelectrons_to_reflectance(clipped, clip_to_unit=True)
        r1, r2 = float(self._reflectance_range[0]), float(self._reflectance_range[1])
        pixels = (ref - r1) / (r2 - r1) * (p2 - p1) + p1
        # ref in [0, 1] but [r1, r2] may be a sub-interval — the linear remap
        # can exit [p1, p2]. Final clip prevents silent uint8-cast wraparound.
        return np.clip(pixels, p1, p2)

    def _record_clip_fraction(self, photoelectrons_img: np.ndarray) -> None:
        """Compute the clip fraction, store it, and warn once if it exceeds 1%."""
        n_total = photoelectrons_img.size
        frac_lo = float((photoelectrons_img < self._fwd_y_min).sum()) / n_total
        frac_hi = float((photoelectrons_img > self._fwd_y_max).sum()) / n_total
        self._last_clip_fraction = (frac_lo, frac_hi)
        if (frac_lo > 0.01 or frac_hi > 0.01) and not self._inverse_clip_warned:
            warnings.warn(
                f"ImageSimulator.photoelectrons_to_pixels: >1% of pixels "
                f"clipped to forward-table bounds "
                f"(low={frac_lo:.3%}, high={frac_hi:.3%}). "
                f"Physically correct (well floor/ceiling) but worth "
                f"investigating if unexpected. Once-per-instance.",
                UserWarning,
                stacklevel=4,
            )
            self._inverse_clip_warned = True

    def _adc_photoelectrons_to_pixels(
        self,
        photoelectrons: np.ndarray,
        *,
        out_range: tuple[float, float] = (0.0, 255.0),
    ) -> np.ndarray:
        """Quantize photoelectrons through the sensor's ADC, then map to pixels.

        Models: ``pe -> clipped to well capacity -> digitized to bit_depth
        levels -> linearly remapped to out_range``. The ADC truncates
        (``floor``), not rounds.

        Args:
            photoelectrons: Photoelectron array.
            out_range: Inclusive lower/upper bounds for the output pixels.

        Returns:
            Pixel array of the same shape as ``photoelectrons``, dtype
            ``float64``.

        Raises:
            ValueError: If the sensor's effective well capacity
                (``max_n * max_well_fill``) is non-positive, if
                ``bit_depth`` is below 1, or if ``out_range`` is not
                strictly ascending.
        """
        p_lo, p_hi = out_range
        if p_lo >= p_hi:
            raise ValueError(
                f"ImageSimulator._adc_photoelectrons_to_pixels: out_range must be "
                f"strictly ascending; got out_range={out_range!r}.",
            )
        well_cap = float(self._sensor.max_n) * float(self._sensor.max_well_fill)
        if well_cap <= 0.0:
            raise ValueError(
                f"ImageSimulator._adc_photoelectrons_to_pixels: degenerate sensor well capacity "
                f"(max_n={self._sensor.max_n!r}, "
                f"max_well_fill={self._sensor.max_well_fill!r}).",
            )
        bit_depth_int = int(self._sensor.bit_depth)
        if bit_depth_int < 1:
            raise ValueError(
                f"ImageSimulator._adc_photoelectrons_to_pixels: bit_depth must be >= 1, "
                f"got {self._sensor.bit_depth!r}.",
            )
        adc_max = (2**bit_depth_int) - 1
        pe_clipped = np.clip(np.asarray(photoelectrons, dtype=np.float64), 0.0, well_cap)
        dn = np.floor((pe_clipped / well_cap) * adc_max)
        pixels = (dn / adc_max) * (p_hi - p_lo) + p_lo
        return np.clip(pixels, p_lo, p_hi)

    def _init_inverse_coefs(self, ref: np.ndarray, pe: np.ndarray) -> None:
        """Compute and store the inverse coefficients and the monotonicity flag.

        ``A`` and ``B`` are taken from the line through the first and last
        grid points. For pyBSM's built-in forward chain this line coincides
        with every interior grid point (reflectance enters the radiance
        integral only as a scalar multiplier), so the inverse is one
        subtract-divide per pixel instead of a per-call interpolation. The
        monotonicity flag is computed alongside so both inverse entry points
        can short-circuit on it without rescanning the forward grid.

        Set once at ``__init__``. Tests that swap the forward map post-init
        must call this explicitly to keep the derived attrs consistent.

        Args:
            ref: Reflectance grid array.
            pe: Photoelectron grid array, same length as ``ref``.
        """
        self._fwd_y_min = float(pe[0])
        self._fwd_y_max = float(pe[-1])
        affine_a = float(pe[-1] - pe[0]) / float(ref[-1] - ref[0])
        affine_b = float(pe[0]) - affine_a * float(ref[0])
        self._affine_pe_coefs = (affine_a, affine_b)
        self._forward_is_monotonic = bool(np.all(np.diff(pe) > 0))

    def _warn_inverse_fallback(self, *, reason: str) -> None:
        """Emit a once-per-instance warning when falling back to ADC quantization.

        Args:
            reason: Short tag describing the fallback trigger; embedded in the
                emitted warning text. Currently only ``"forward_nonmonotonic"``.
        """
        if self._inverse_fallback_warned:
            return
        warnings.warn(
            f"ImageSimulator.photoelectrons_to_pixels: the inverse formula "
            f"cannot be used (reason={reason!r}), so output is falling back "
            f"to ADC quantization. Outputs from this simulator may no "
            f"longer be comparable to those from other sensors. "
            f"Once-per-instance.",
            UserWarning,
            stacklevel=4,
        )
        self._inverse_fallback_warned = True

    def _warn_inverse_precision_degraded(self, *, reason: str) -> None:
        """Emit a once-per-instance warning when the inverse precision claim is degraded.

        Shares ``_inverse_fallback_warned`` with ``_warn_inverse_fallback``, so
        the user sees at most one inverse-degradation warning per simulator
        instance regardless of which entry point (pixels or reflectance)
        detects the problem first.

        Args:
            reason: Short tag describing the trigger. Currently only
                ``"forward_nonmonotonic"`` (saturated forward grid).
        """
        if self._inverse_fallback_warned:
            return
        warnings.warn(
            f"ImageSimulator.photoelectrons_to_reflectance: the forward map "
            f"is non-monotonic (reason={reason!r}), so the inverse formula "
            f"may have errors well above machine precision at interior "
            f"points. Once-per-instance.",
            UserWarning,
            stacklevel=3,
        )
        self._inverse_fallback_warned = True

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
        """Return the cached sensor/scenario configuration hash (set once at ``__init__``)."""
        return self._config_hash

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
