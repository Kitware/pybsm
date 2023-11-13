# -*- coding: utf-8 -*-
"""
The python Based Sensor Model (pyBSM) is a collection of electro-optical camera
modeling functions developed by the Air Force Research Laboratory, Sensors Directorate.

Please use the following citation:
LeMaster, Daniel A.; Eismann, Michael T., "pyBSM: A Python package for modeling
imaging systems", Proc. SPIE 10204 (2017)

Distribution A.  Approved for public release.
Public release approval for version 0.0: 88ABW-2017-3101
Public release approval for version 0.1: 88ABW-2018-5226


contact: daniel.lemaster@us.af.mil

version 0.2: CURRENTLY IN BETA!!


"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import inspect
import warnings

# pybsm imports
import pybsm.otf as otf
import pybsm.noise as noise
import pybsm.radiance as radiance


#new in version 0.2.  We filter warnings associated with calculations in the function
#circularApertureOTF.  These invalid values are caught as NaNs and appropriately
#replaced.
warnings.filterwarnings('ignore', r'invalid value encountered in arccos')
warnings.filterwarnings('ignore', r'invalid value encountered in sqrt')
warnings.filterwarnings('ignore',r'invalid value encountered in true_divide')
warnings.filterwarnings('ignore',r'divide by zero encountered in true_divide')

#find the current path (used to locate the atmosphere database)
#dirpath = os.path.dirname(os.path.abspath(__file__))
dirpath = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


def instantaneousFOV(w, f):
    """The instantaneous field-of-view, i.e. the angular footprint
        of a single detector in object space.

    Parameters
    ----------
    w:
        detector size (width) in the x and y directions (m)
    f :
        focal length (m)

    Returns
    -------
    ifov:
        detector instantaneous field-of-view (radians)
    """
    ifov = w/f
    return ifov


def wienerFiler(otf,noiseToSignalPS):
    """An image restoration transfer function based on the Wiener Filter.  See
    from Gonzalex and Woods, "Digital Image Processing," 3 ed.  Note that the
    filter is not normalized so that WF = 1 at 0 spatial frequency.  This is
    easily fixed for the case where noiseToSignalPS is a
    scalar: (1.0+noisetosignalPS)*WF = 1 at 0 spatial frequency.
    This is noteworthy because, without normalization of some kind, system
    MTF * WF is not a proper MTF. Also, for any frequency space sharpening
    kernel, be sure that sharpening does not extend past the Nyquist frequency,
    or, if so, ensure that the filter response wraps around Nyquist
    appropriately.

    Parameters
    ----------
    otf:
        System optical transfer function.
    noiseTosignalPS:
        Ratio of the noise power spectrum to the signal power spectrum.  This
        may be a function of spatial frequency (same size as otf) or an scalar.


    Returns
    -------
    WF :
        Frequency space representation of the Wienerfilter.
    """
    WF = np.conj(otf) / (np.abs(otf)**2+noiseToSignalPS)

    return WF


def img2reflectance(img,pix_values,refl_values):
    """Maps pixel values to reflectance values with linear interpolation between
    points.  Pixel values that map below zero reflectance or above unity reflectance
    are truncated.  Implicitly, reflectance is contast across the camera bandpass.

    Parameters
    ----------
    img :
        the image that will be transformed into reflectance (counts)
    pix_values :
        array of values in img that map to a known reflectance (counts)
    refl_values :
        array of reflectances that correspond to pix_values (unitless)


    Returns
    -------
    refImg :
        the image in reflectance space
    """
    f = interpolate.interp1d(pix_values,refl_values, fill_value='extrapolate', assume_sorted = 0)
    refImg = f(img)
    refImg[refImg > 1.0] = 1.0
    refImg[refImg < 0.0] = 0.0

    return refImg


def simulate_image(ref_img, sensor, scenario):
    """Simulates radiometrically accurate imagery collected through a sensor.

    We start with a notionally ideal reference image 'img', which captures a
    view of the world of which we would like to simulate a degraded view that
    would be collected from another imaging system. Our framework for image
    simulation requires that this reference image is of higher-quality and
    ideally higher-resolution than the view we would like to simulate as we can
    only model further degradation of image quality.


    Parameters
    ----------
    ref_img : RefImage
        Reference image to use as the source view of the world to be emulated
        by the virtual camera defined by 'sensor'.
    sensor : Sensor
        Virtual sensor definition.
    scenario : Scenario
        Specification of the deployment of the virtual sensor within the world
        relative to the target.

    Returns
    -------
    trueImg : Numpy float64 array
        The true image in units of photoelectrons.
    blurImg : Numpy float64 array
        The image after blurring and resampling is applied to trueImg (still
        units of photoelectrons).
    noisyImg : Numpy float64 array
        The blur image with photon (Poisson) noise and gaussian noise applied
        (still units of photoelectrons).

    WARNING
    -------
    imggsd must be small enough to properly sample the blur kernel! As a guide,
    if the image system transfer function goes to zero at angular spatial frequency, coff,
    then the sampling requirement will be readily met if imggsd <= rng/(4*coff).
    In practice this is easily done by upsampling imgin.
    """
    # integration time (s)
    intTime = sensor.intTime

    ref, pe, spectral_weights = radiance.reflectance2photoelectrons(scenario.atm,
                                                                    sensor,
                                                                    intTime)

    wavelengths = spectral_weights[0]
    weights = spectral_weights[1]

    slant_range = np.sqrt(scenario.altitude**2 + scenario.ground_range**2)

    #cut down the wavelength range to only the regions of interest
    mtfwavelengths =  wavelengths[weights > 0.0]
    mtfweights =  weights[weights > 0.0]

    # Assume if nothing else cuts us off first, diffraction will set the limit
    # for spatial frequency that the imaging system can resolve (1/rad).
    cutoffFrequency = sensor.D/np.min(mtfwavelengths)

    urng = np.linspace(-1.0, 1.0, 101)*cutoffFrequency
    vrng = np.linspace(1.0, -1.0, 101)*cutoffFrequency

    #meshgrid of spatial frequencies out to the optics cutoff
    uu, vv = np.meshgrid(urng, vrng)

    system_otf = otf.commonOTFs(sensor, scenario, uu, vv, mtfwavelengths,
                                mtfweights, slant_range, intTime).systemOTF

    df = urng[1]-urng[0]

    assert df > 0

    ifov = (sensor.px + sensor.py)/2 / sensor.f

    # Standard deviation of additive Gaussian noise (e.g. read noise,
    # quantization). Should be the RSS value if multiple terms are combined.
    # This should not include photon noise.
    quantizationNoise = noise.quantizationNoise(sensor.maxN, sensor.bitdepth)
    gnoise = np.sqrt(quantizationNoise**2.0 + sensor.readNoise**2.0)

    # Convert to reference image into a floating point reflectance image.
    refImg = img2reflectance(ref_img.img, ref_img.pix_values,
                             ref_img.refl_values)

    # Convert the reflectance image to photoelectrons.
    f = interpolate.interp1d(ref, pe)
    trueImg = f(refImg)

    #blur and resample the image
    blurImg, _ = otf.apply_otf_to_image(trueImg, ref_img.gsd, slant_range,
                                        system_otf, df, ifov)

    #add photon noise (all sources) and dark current noise
    noisyImg = np.random.poisson(lam=blurImg)
    #add any noise from Gaussian sources, e.g. readnoise, quantizaiton
    noisyImg = np.random.normal(noisyImg, gnoise)

    if noisyImg.shape[0] > ref_img.img.shape[0]:
        print("Warning!  The simulated image has oversampled the reference image!  This result should not be trusted!!")

    return trueImg, blurImg, noisyImg


def stretch_contrast_convert_8bit(img, perc=[0.1, 99.9]):
    img = img.astype(float)
    img = img - np.percentile(img.ravel(), perc[0])
    img = img/(np.percentile(img.ravel(), perc[1])/255)
    img = np.clip(img, 0, 255)
    return np.round(img).astype(np.uint8)


#################################################################
#start defining classes for cameras, scenarios, etc.
################################################################


class RefImage(object):
    """Reference image.

    img : Numpy array
        Reference image.
    gsd : float
        Spatial sampling for 'img' in meters. Each pixel in 'img' is assumed to
        capture a 'gsd' x 'gsd' square of some world surface. We assume the
        sampling is isotropic (x and y sampling are identical) and uniform
        across the whole field of view. This is generally a valid assumption
        for remote sensing imagery.
    pix_values : array-like of float
        Pixel count values within 'img' that should be associated with the
        corresponding reflectance values in 'refl_values' by linear
        interpolation. This is used to convert raw image values into an assumed
        spectral reflectance of the scene being viewed.
    refl_values : array-like of float
        Reflectance values associated with the corresponding pixel count values
        in 'pix_values' used to convert raw image values into an assumed
        spectral reflectance of the scene being viewed.
    name : str | None
        Name of the image.
    """
    def __init__(self, img, gsd, pix_values=None, refl_values=None, name=None,
                 orthophoto=True):
        self.img = img
        self.gsd = gsd
        self.name = name

        if pix_values is None:
            pix_values = np.array([np.percentile(img.ravel(), 0.2),
                                   np.percentile(img.ravel(), 99.8)])
            refl_values = np.array([.05, .95])
        else:
            assert refl_values is not None, \
                'if \'pix_values\' is provided, \'refl_values\' must be as well'

        self.pix_values = pix_values
        self.refl_values = refl_values

    def estimate_capture_parameters(self, altitude=2000000):
        """Estimate the scenario and sensor parameters that are consistent with
        this image.

        This provides a no-degradation baseline from which to alter parameters
        to explore further degradation.
        """
        # Let's assume the full visible spectrum.
        optTransWavelengths = np.array([380, 700])*1.0e-9    # m

        scenario = Scenario(self.name, 1, altitude, ground_range=0,
                            aircraftSpeed=0, haWindspeed=0, cn2at1m=0)

        # Guess at a relatively large pixel pitch, which should have a large
        # well depth.
        p = 20e-6    # m

        # Calculate the focal length (m) that matches the GSD for the
        # prescribed altitude and pixel pitch.
        f = altitude*p/self.gsd  # m

        # Instantenous field of view (iFOV), the angular extent of the world
        # covered by one pixel (radians).
        ifov = 2*np.arctan(p/2/f)

        # We are assuming a circular aperture without obscuration. The
        # diffraction limited angular resolution (where on Airy disk sits in
        # the first ring of another Airy disk) is 1.22*lambda/D. But, let's use
        # a coefficient of 4 for safety.
        D = 4*np.median(optTransWavelengths)/ifov

        sensor = Sensor(self.name, D, f, p, optTransWavelengths)
        return sensor, scenario

    def show(self):
        h, w = self.img.shape[:2]
        plt.imshow(self.img, extent=[-w/2*self.gsd, w/2*self.gsd,
                                     -h/2*self.gsd, h/2*self.gsd])
        plt.xlabel('X-Position (m)', fontsize=24)
        plt.ylabel('Y-Position (m)', fontsize=24)
        plt.tight_layout()


class Sensor(object):
    """Example details of the camera system.  This is not intended to be a
    complete list but is more than adequate for the NIIRS demo (see pybsm.niirs).

    Attributes (the first four are mandatory):
    ------------------------------------------
    name :
        Name of the sensor (string)
    D :
        Effective aperture diameter (m)
    f :
        Focal length (m)
    px and py :
        Detector center-to-center spacings (pitch) in the x and y directions
        (meters). IF py is not provided, it is assumed equal to px.
    optTransWavelengths : numpy array
        Specifies the spectral bandpass of the camera (m).  At minimum, and
        start and end wavelength should be specified.

    opticsTransmission :
        Full system in-band optical transmission (unitless).  Loss due to any
        telescope obscuration should *not* be included in with this optical
        transmission array.
    eta :
        Relative linear obscuration (unitless). Obscuration of the aperture
        commonly occurs within telescopes due to secondary mirror or spider
        supports.
    wx and wy :
        Detector width in the x and y directions (m). If set equal to px and
        py, this corresponds to an assumed full pixel fill factor. In general,
        wx and wy are less than px and py due to non-photo-sensitive area
        (typically transistors) around each pixel.
    qe :
        Quantum efficiency as a function of wavelength (e-/photon).
    qewavelengths :
        Wavelengths corresponding to the array qe (m).
    otherIrradiance :
        Spectral irradiance from other sources (W/m^2 m). This is particularly
        useful for self emission in infrared cameras.  It may also represent
        stray light.
    darkCurrent :
        Detector dark current (e-/s). Dark current is the relatively small
        electric current that flows through photosensitive devices even when no
        photons enter the device.
    maxN :
        Detector electron well capacity (e-). The default 100 million,
        initializes to a large number so that, in the absence of better
        information, it doesn't affect outcomes.
    maxFill :
        Desired electron well fill, i.e. maximum well size x desired fill
        fraction.
    bitdepth :
        Resolution of the detector ADC in bits (unitless). Default of 100 is
        sufficiently large number so that in the absense of better information,
        it doesn't affect outcomes.
    ntdi :
        Number of TDI stages (unitless).
    coldshieldTemperature :
        Temperature of the cold shield (K).  It is a common approximation to assume
        that the coldshield is at the same temperature as the detector array.
    opticsTemperature :
        Temperature of the optics (K)
    opticsEmissivity :
        Emissivity of the optics (unitless) except for the cold filter.
        A common approximation is 1-optics transmissivity.
    coldfilterTransmission :
        Transmission through the cold filter (unitless)
    coldfilterTemperature :
        Temperature of the cold filter.  It is a common approximation to assume
        that the filter is at the same temperature as the detector array.
    coldfilterEmissivity :
        Emissivity through the cold filter (unitless).  A common approximation
        is 1-cold filter transmission
    sx and sy :
        Root-mean-squared jitter amplitudes in the x and y directions respectively. (rad)
    dax and day :
        Line-of-sight angular drift rate during one integration time in the x and y
        directions respectively. (rad/s)
    pv :
        Wavefront error phase variance (rad^2) - tip: write as (2*pi*waves of error)^2
    pvwavelength :
        Wavelength at which pv is obtained (m)
    Lx and Ly :
        Correlation lengths of the phase autocorrelation function.  Apparently,
        it is common to set the Lx and Ly to the aperture diameter.  (m)
    otherNoise :
        A catch all for noise terms that are not explicitly included elsewhere
        (read noise, photon noise, dark current, quantization noise are
        all already included)
    filterKernel:
         2-D filter kernel (for sharpening or whatever).  Note that
         the kernel is assumed to sum to one.
    framestacks:
         The number of frames to be added together for improved SNR.

    """
    def __init__(self, name, D, f, px, optTransWavelengths,
                 eta=0.0, py=None, wx=None, wy=None, intTime=1, darkCurrent=0,
                 otherIrradiance=0.0, readNoise=0, maxN=int(100.0e6),
                 maxWellFill=1.0, bitdepth=100.0, ntdi=1.0,
                 coldshieldTemperature=70.0, opticsTemperature=270.0,
                 opticsEmissivity=0.0, coldfilterTransmission=1.0,
                 coldfilterTemperature=70.0, coldfilterEmissivity=0.0,
                 sx=0.0, sy=0.0, dax=0.0, day=0.0, pv=0.0,
                 pvwavelength=0.633e-6):
        """Returns a sensor object whose name is *name* and...."""
        self.name = name
        self.D = D
        self.f = f
        self.px = px
        self.optTransWavelengths = optTransWavelengths
        self.opticsTransmission = np.ones(optTransWavelengths.shape[0])
        self.eta = eta

        if py is None:
            # Assume square pixels.
            self.py = px
        else:
            py = self.py

        if wx is None:
            # Assume is 100% fill factor and square detectors.
            self.wx = px
        else:
            self.wx = wx

        if wy is None:
            # Assume is same fill factor as along the x.
            self.wy = px/self.wx*self.py
        else:
            self.wy = wy

        self.intTime = intTime
        self.darkCurrent =  darkCurrent
        self.otherIrradiance = otherIrradiance
        self.readNoise = readNoise
        self.maxN = maxN
        self.maxWellFill = maxWellFill
        self.bitdepth=bitdepth
        self.ntdi = ntdi

        # TODO this should be exposed so a custom one can be provided.
        self.qewavelengths = optTransWavelengths #tplaceholder
        self.qe = np.ones(optTransWavelengths.shape[0]) #placeholder

        # TODO I don't think these automatically get used everywhere they
        # should, some functions overridde by assuming different temperatures.
        self.coldshieldTemperature = coldshieldTemperature
        self.opticsTemperature = opticsTemperature
        self.opticsEmissivity = opticsEmissivity
        self.coldfilterTransmission = coldfilterTransmission
        self.coldfilterTemperature = coldfilterTemperature
        self.coldfilterEmissivity = coldfilterEmissivity
        self.sx = sx
        self.sy = sy
        self.dax = dax
        self.day = day
        self.pv = pv
        self.pvwavelength = pvwavelength
        self.Lx = D
        self.Ly = D
        self.otherNoise = np.array([0])

        # TODO, before we expose these, we should track down whether they are
        # actually used anywhere downstream.
        self.filterKernel = np.array([1])
        self.framestacks = 1


class Scenario(object):
    """Everything about the target and environment.  NOTE:  if the niirs model
    is called, values for target/background temperature, reflectance, etc. are
    overridden with the NIIRS model defaults.
    ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural extinction with 23 km visibility)
        or ihaze = 2 (Rural extinction with 5 km visibility)
    altiude:
        Sensor height above ground level in meters.  The database includes the
        following altitude options: 2 32.55 75 150 225 500 meters, 1000 to
        12000 in 1000 meter steps, and 14000 to 20000 in 2000 meter steps,
        24500.
    ground_range:
        Projection of line of sight between the camera and target along on the
        ground in meters. The distance between the target and the camera is
        given by sqrt(altitude^2 + ground_range^2).
        The following ground ranges are included in the database at each altitude
        until the ground range exceeds the distance to the spherical earth horizon:
        0 100 500 1000 to 20000 in 1000 meter steps, 22000 to 80000 in 2000 m steps,
        and  85000 to 300000 in 5000 meter steps.
    aircraftSpeed:
        Ground speed of the aircraft (m/s)
    targetReflectance:
        Object reflectance (unitless). The default 0.15 is the giqe standard.
    targetTemperature:
        Object temperature (Kelvin). 282 K is used for GIQE calculation.
    backgroundReflectance:
        Background reflectance (unitless)
    backgroundTemperature:
        Background temperature (Kelvin). 280 K used for GIQE calculation.
    haWindspeed:
        The high altitude windspeed (m/s).  Used to calculate the turbulence
        profile. The default, 21.0, is the HV 5/7 profile value.
    cn2at1m:
        The refractive index structure parameter "near the ground"
        (e.g. at h = 1 m). Used to calculate the turbulence profile. The
        default, 1.7e-14, is the HV 5/7 profile value.

    """
    def __init__(self, name, ihaze, altitude, ground_range, aircraftSpeed=0,
                 targetReflectance=0.15, targetTemperature=295,
                 backgroundReflectance=0.07, backgroundTemperature=293,
                 haWindspeed=21, cn2at1m=1.7e-14):
        self.name = name
        self._ihaze = ihaze
        self._altitude = altitude
        self._ground_range = ground_range
        self.aircraftSpeed = aircraftSpeed
        self.targetReflectance = targetReflectance
        self.targetTemperature = targetTemperature
        self.backgroundReflectance = backgroundReflectance
        self.backgroundTemperature = backgroundTemperature
        self.haWindspeed = haWindspeed
        self.cn2at1m = cn2at1m

        # Will be loaded on demand for a particular altitude.
        self._atm = None

    @property
    def ihaze(self):
        return self._ihaze

    @ihaze.setter
    def ihaze(self, value):
        self._atm = None
        self._ihaze = value

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        self._atm = None
        self._altitude = value

    @property
    def ground_range(self):
        return self._ground_range

    @ground_range.setter
    def ground_range(self, value):
        self._atm = None
        self._ground_range = value

    @property
    def atm(self):
        """Return atmospheric spectral absorption.

        Returns
        -------
        atm[:,0]:
            wavelengths from .3 to 14 x 10^-6 m in 0.01x10^-6 m steps
        atm[:,1]:
            (TRANS) total transmission through the defined path.
        atm[:,2]:
            (PTH THRML) radiance component due to atmospheric emission and scattering received at the observer.
        atm[:,3]:
            (SURF EMIS) component of radiance due to surface emission received at the observer.
        atm[:,4]:
            (SOL SCAT) component of scattered solar radiance received at the observer.
        atm[:,5]:
            (GRND RFLT) is the total solar flux impingent on the ground and reflected directly to the sensor from the ground. (direct radiance + diffuse radiance) * surface reflectance
        NOTE: units for columns 1 through 5 are in radiance W/(sr m^2 m)

        """
        if self._atm is None:
            # Read in and cache results.
            self._atm = radiance.loadDatabaseAtmosphere(self.altitude,
                                                        self.ground_range,
                                                        self.ihaze)

        return self._atm
