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
from scipy.special import jn
import cv2
import os, inspect
import warnings

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

#define some useful physical constants
hc = 6.62607004e-34 # Plank's constant  (m^2 kg / s)
cc = 299792458.0 # speed of light (m/s)
kc = 1.38064852e-23 #Boltzmann constant (m^2 kg / s^2 K)
qc = 1.60217662e-19 # charge of an electron (coulombs)
rEarth = 6378.164e3 #radius of the earth (m)

def turbulenceOTF(u,v,lambda0,D,r0,alpha):
    """IBSM Equation 3-3.  The long or short exposure turbulence OTF.
    
    Parameters
    ---------
    u or v : 
        angular spatial frequency coordinates (rad^-1)        
    lambda0 :
        wavelength (m)
    D :
        effective aperture diameter (m)
    r0 : 
        Fried's correlation diameter (m)
    alpha :
        long exposure (alpha = 0) or short exposure (alpha = 0.5)
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
        """
    rho = np.sqrt(u**2.0+v**2.0) # radial spatial frequency
    H = np.exp(-3.44*(lambda0*rho/r0)**(5.0/3.0) * \
        (1-alpha*(lambda0*rho/D)**(1.0/3.0)))
    return H
    
def coherenceDiameter(lambda0,zPath,cn2):
    """
    This is an improvement / replacement for IBSM Equation 3-5: calculation of
    Fried's coherence diameter (m) for spherical wave propagation.  
    It is primarily used in calculation of 
    the turbulence OTF.  This version comes from Schmidt, "Numerical Simulation 
    of Optical Wave Propagation with Examples in Matlab" SPIE Press (2010). In 
    turn, Schmidt references Sasiela, "Electromagnetic Wave Propagation in
    Turbulence: Evaluation of Application of Mellin Transforms" SPIE Press (2007).
    
    Parameters
    ---------
    lambda0 :
        wavelength (m).  As an implementation note, r0 can be calculated at a
        1e-6 m and then multiplied by lambda^6/5 to scale to other
        wavelengths.  This saves the time lost to needlessly evaluating extra
        integrals.
    zPath :
        array of samples along the path from the target (zPath = 0) to the 
        sensor. (m) WARNING: trapz will FAIL if you give it a two element path,
        use a long zPath array, even if cn2 is constant
    cn2 :
        refractive index structure parameter values at the sample locations in
        zPath (m^(-2/3)).  Typically Cn2 is a function of height so, as an
        intermediate step, heights should be calculated at each point along
        zPath (see altitudeAlongSlantPath)
        
    Returns
    -------
    r0 :
        correlation diameter (m) at wavelength lambda0
        """
    #the path integral of the structure parameter term
    spIntegral = np.trapz(cn2*(zPath/zPath.max())**(5.0/3.0),zPath)
    
    r0 = (spIntegral*0.423*(2*np.pi/lambda0)**2)**(-3.0/5.0)

    return r0

def nadirAngle(hTarget,hSensor,slantRange):
    """
    Work through the law of cosines to calculate the sensor nadir angle above
    a circular earth. (i.e. angle between looking straight down (nadir = 0) and looking along the
    slant path).      
    
    Parameters
    ---------
    hTarget:
        height of the target above sea level (m).
    hSensor:
        height of the sensor above sea level (m).
    slantRange:
        distance between the target and sensor (m).
        
    Returns
    -------
    nadir :
        the sensor nadir angle. (rad)
    """
       
    a = rEarth+hSensor
    b = slantRange
    c = rEarth+hTarget
    nadir = np.arccos(-1.0*(c**2.0 - a**2.0 - b**2.0)/(2.0*a*b))
    
    return nadir

def curvedEarthSlantRange(hTarget,hSensor,groundRange):
    """Returns the slant range from target to sensor above a curved (circular)
    Earth.
    
    Parameters
    ----------
    hTarget:
        height of the target above sea level (m).
    hSensor:
        height of the sensor above sea level (m).
    groundRange:
        distance between the target and sensor on the ground (m).
        
    Returns
    -------
    slantRange :
        distance between the target and sensor (m).
    """
    a = rEarth+hSensor
    c = rEarth+hTarget
    theta=groundRange/rEarth #exact arc length angle (easy to derive)
    slantRange=np.sqrt(c**2.0+a**2.0 -2.0*a*c*np.cos(theta));
    
    return slantRange

def altitudeAlongSlantPath(hTarget,hSensor,slantRange):
    """
    Calculate the height above the curved earth at points along a path from the
    target (zPath=0) to the sensor (zPath.max()).  This is primarily useful for 
    calculating the atmospheric coherence diameter, r0.
    
    
    Parameters
    ---------
    hTarget:
        height of the target above sea level (m).
    hSensor:
        height of the sensor above sea level (m).
    slantRange:
        distance between the target and sensor (m).
        
    Returns
    -------
    zPath :
        array of samples along the path from the target (zPath = 0) to the 
        sensor. (m)
    hPath :
        height above the earth along a slantpath defined by zPath. (m)

        """
    
    #this is simple law of cosines problem
    nadir = nadirAngle(hTarget,hSensor,slantRange)
    a = rEarth+hSensor
    b = np.linspace(0.0,slantRange,10000) #arbitrary choice of 100,000 data points
    c  = np.sqrt(a**2+b**2-2*a*b*np.cos(nadir))
    
    zPath = b
    hPath = (c-rEarth)[::-1]
    #It is correct to reverse the order of hPath.  The height of the target above
    #the earth (hTarget) should be the first element in the array.  Effectively,
    #this reversal just changes the location of the origin of zPath to the target
    #location.  It was just more convinient to use the origin at the sensor in
    #the law of cosines calculation

    return (zPath,hPath)

def hufnagelValleyTurbulenceProfile(h,v,cn2at1m):
    """Replaces IBSM Equations 3-6 through 3-8.  The Hufnagel-Valley Turbulence
    profile (i.e. a profile of the refractive index structure parameter as a function
    of altitude).  I suggest the HV profile because it seems to be in more widespread
    use than the profiles listed in the IBSM documentation.  This is purely a
    personal choice.  The HV equation comes from Roggemann et al., "Imaging 
    Through Turbulence", CRC Press (1996).  The often quoted HV 5/7 model is a 
    special case where Cn2at1m = 1.7e-14 and v = 21.  HV 5/7 should result 
    in a 5 cm coherence diameter (r0) and 7 urad isoplanatic angle along a 
    vertical slant path into space.
    
    Parameters
    ---------
    h:
        height above ground level in (m)
    v: 
        the high altitude windspeed (m/s)
    cn2at1m:
        the refractive index structure parameter "near the ground" (e.g. at h = 1 m)
    
    Returns
    -------
    cn2 :
        refractive index structure parameter as a function of height (m^(-2/3))
        """
    cn2 = 5.94e-53*(v/27.0)**2.0*h**10.0*np.exp(-h/1000.0) \
    + 2.7e-16*np.exp(-h/1500.0) +cn2at1m*np.exp(-h/100.0)
    
    return cn2

def windspeedTurbulenceOTF(u,v,lambda0,D,r0,td,vel):
    """IBSM Equation 3-9.  Turbulence OTF adjusted for windspeed and
    integration time.
    
    Parameters
    ---------
    u or v : 
        angular spatial frequency coordinates (rad^-1)        
    lambda0 :
        wavelength (m)
    D : 
        effective aperture diameter (m)
    r0 : 
        Fried's coherence diameter (m)
    td : 
        dwell (i.e. integration) time (seconds)
    vel : 
        apparent atmospheric velocity (m/s)
    
    Returns
    -------
    H :
        OTF at spatial frequency (u,v) (unitless)
        """
    weight = np.exp(-vel*td/r0)
    H = weight*turbulenceOTF(u,v,lambda0,D,r0,0.5) + \
        (1-weight)*turbulenceOTF(u,v,lambda0,D,r0,0.0)
    return H
    
def circularApertureOTF(u,v,lambda0,D,eta):
    """IBSM Equation 3-20.  Obscured circular aperture diffraction OTF.  If eta 
    is set to 0, the function will return the unobscured aperture result.
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)        
    lambda0 : 
        wavelength (m)
    D : 
        effective aperture diameter (m)
    eta : 
        relative linear obscuration (unitless)
    
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    Notes
    -----
    You will see several runtime warnings when this code is first accessed.  The
    issue (calculating arccos and sqrt outside of their domains) is captured 
    and corrected np.nan_to_num
    """
    
    rho = np.sqrt(u**2.0+v**2.0) # radial spatial frequency
    r0=D/lambda0          # diffraction limited cutoff spatial frequency (cy/rad)
        
    #this A term by itself is the unobscured circular aperture OTF
    A=(2.0/np.pi)*(np.arccos(rho/r0)-(rho/r0)*np.sqrt(1.0-(rho/r0)**2.0))
    A = np.nan_to_num(A)    
     
    #   region where (rho < (eta*r0)):
    B=(2.0*eta**2.0/np.pi)*(np.arccos(rho/eta/r0)-(rho/eta/r0)* \
    np.sqrt(1.0-(rho/eta/r0)**2.0))
    B = np.nan_to_num(B) 
    
    #   region where (rho < ((1.0-eta)*r0/2.0)):
    C1 = -2.0*eta**2.0*(rho < (1.0-eta)*r0/2.0)
    
    #   region where (rho <= ((1.0+eta)*r0/2.0)):
    phi=np.arccos((1.0+eta**2.0-(2.0*rho/r0)**2)/2.0/eta)
    C2=2.0*eta*np.sin(phi)/np.pi+(1.0+eta**2.0)*phi/np.pi-2.0*eta**2.0
    C2=C2-(2.0*(1.0-eta**2.0)/np.pi)*np.arctan((1.0+eta)* \
            np.tan(phi/2.0)/(1.0-eta))
    C2 = np.nan_to_num(C2)         
    C2 = C2*(rho <= ((1.0+eta)*r0/2.0))
            
    #note that C1+C2 = C from the IBSM documentation
    
    if (eta > 0.0):
        H=(A+B+C1+C2)/(1.0-eta**2.0)
    else:
        H=A
    return H

def defocusOTF(u,v,D,wx,wy):
    """IBSM Equation 3-25.  Gaussian approximation for defocus on the optical 
    axis. This function is retained for backward compatibility.  See 
    circularApertureOTFwithDefocus for an exact solution.
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)        
    D : 
        effective aperture diameter (m)
    (wx,wy) :
        the 1/e blur spot radii in the x and y directions
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    H = np.exp((-np.pi**2.0/4.0) * (wx**2.0*u**2.0 + wy**2.0*v**2.0))    
    
    return H

def objectDomainDefocusRadii(D,R,R0):
    """IBSM Equation 3-26.  Axial defocus blur spot radii in the object domain.
    
    Parameters
    ----------        
    D : 
        effective aperture diameter (m)
    R :
        object range (m)
    R0 :
        range at which the focus is set (m)
        
    Returns
    -------
    w :
        the 1/e blur spot radii (rad) in one direction
    
    """
    w = 0.62*D*(1.0/R-1.0/R0)
    return w

def imageDomainDefocusRadii(D,dz,f):
    """IBSM Equation 3-27.  Axial defocus blur spot radii in the image domain.
    
    Parameters
    ----------        
    D : 
        effective aperture diameter (m)
    dz :
        axial defocus in the image domain (m)
    f :
        focal length (m)
        
    Returns
    -------
    w :
        the 1/e blur spot radii (rad) in one direction
    
    """
    w = 0.62*D*dz/(f**2.0)
    return w

def jitterOTF(u,v,sx,sy):
    """IBSM Equation 3-28.  Blur due to random line-of-sight motion that occurs at high 
    frequency, i.e. many small random changes in line-of-sight during a single integration time.
    #Note that there is an error in Equation 3-28 - pi should be squared in the exponent.
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)        
    sx or sy :
        Root-mean-squared jitter amplitudes in the x and y directions respectively. (rad)
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    H = np.exp((-2.0*np.pi**2.0) * (sx**2.0*u**2.0 + sy**2.0*v**2.0))    
    
    return H

def driftOTF(u,v,ax,ay):
    """IBSM Equation 3-29.  Blur due to constant angular line-of-sight motion during
    the integration time.
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)        
    ax or ay :
        line-of-sight angular drift during one integration time in the x and y 
        directions respectively. (rad)       
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    H = np.sinc(ax*u)*np.sinc(ay*v)   
    
    return H

def wavefrontOTF(u,v,lambda0,pv,Lx,Ly):
    """IBSM Equation 3-31.  Blur due to small random wavefront errors in the pupil.
    Use with the caution that this function assumes a specifc phase autocorrelation 
    function.  Refer to the discussion on random phase screens in Goodman, "Statistical Optics" 
    for a full explanation (this is also the source cited in the IBSM documentation).
    As an alternative, see wavefrontOTF2.
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)
    lambda0 : 
        wavelength (m)
    pv : 
        phase variance (rad^2) - tip: write as (2*pi*waves of error)^2.  pv is
        often defined at a specific wavelength (e.g. 633 nm) so scale appropriately.
        
    Lx or Ly :
        correlation lengths of the phase autocorrelation function.  Apparently,
        it is common to set Lx and Ly to the aperture diameter.  (m)       
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    autoc = np.exp(-lambda0**2 * ( (u/Lx)**2 + (v/Ly)**2 )) 
    H = np.exp(-pv * (1-autoc))    
    
    return H

def radialUserOTF(u,v,fname):
    """IBSM Section 3.2.6.  Import a user-defined, 1-dimensional radial OTF and
    interpolate it onto a 2-dimensional spatial frequency grid.  Per ISBM Table
    3-3a, the OTF data are ASCII text, space delimited data.  Each line of text
    is formatted as - spatial_frequency OTF_real OTF_imaginary.
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)
    fname : 
        filename and path to the radial OTF data.       
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    radialData = np.genfromtxt(fname)
    radialSF = np.sqrt(u**2.0 + v**2.0) #calculate radial spatial frequencies
    
    H = np.interp(radialSF,radialData[:,0],radialData[:,1]) + \
    np.interp(radialSF,radialData[:,0],radialData[:,2])*1.0j
    
    return H

def xandyUserOTF(u,v,fname):
    """USE xandyUserOTF2 INSTEAD!  The original pyBSM documentation contains an error. 
    IBSM Equation 3-32.  Import user-defined, 1-dimensional x-direction and
    y-direction OTFs and interpolate them onto a 2-dimensional spatial frequency grid.  Per ISBM Table
    3-3c, the OTF data are ASCII text, space delimited data.  (Note: There
    appears to be a typo in the IBSM documentation - Table 3-3c should represent
    the "x and y" case, not "x or y".)
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)
    fname : 
        filename and path to the x and y OTF data.       
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    xandyData = np.genfromtxt(fname)
    
    Hx = np.interp(np.abs(u),xandyData[:,0],xandyData[:,1]) + \
    np.interp(np.abs(u),xandyData[:,0],xandyData[:,2])*1.0j
    
    Hy = np.interp(np.abs(v),xandyData[:,3],xandyData[:,4]) + \
    np.interp(np.abs(v),xandyData[:,3],xandyData[:,5])*1.0j

    H = Hx*Hy
    
    return H

def userOTF2D(u,v,fname,nyquist):
    """IBSM Section 3.2.7.  Import an user-defined, 2-dimensional OTF and 
    interpolate onto a 2-dimensional spatial frequency grid.  The OTF data is assumed to
    be stored as a 2D Numpy array (e.g. 'fname.npy'); this is easier than trying to resurrect the 
    IBSM image file format.  Zero spatial frequency is taken to be at the center
    of the array.  All OTFs values extrapolate to zero outside of the domain of 
    the imported OTF.
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)
    fname : 
        filename and path to the OTF data.  Must include the .npy extension.
    nyquist: the Nyquist (i.e. maximum) frequency of the OFT file.  The support
    of the OTF is assumed to extend from -nyquist to nyquist. (rad^-1)
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    rawOTF = np.load(fname)   
    
    #find the row and column space of the raw OTF data
    vspace = np.linspace(1,-1,rawOTF.shape[0])*nyquist
    uspace = np.linspace(-1,1,rawOTF.shape[1])*nyquist 
    ugrid, vgrid = np.meshgrid(uspace,vspace)
    
    #reshape the data to be acceptable input to scipy's interpolate.griddata
    #this apparently works but I wonder if there is a better way?
    rawOTF = rawOTF.reshape(-1)    
    ugrid = ugrid.reshape(-1)
    vgrid = vgrid.reshape(-1)
    
    H = interpolate.griddata((ugrid,vgrid),rawOTF,(u,v),method='linear', \
    fill_value=0)
    
    return H
    
def atFocalPlaneIrradiance(D,f,L):
    """Converts pupil plane radiance to focal plane irradiance for an extended source.
    This is a variation on part of IBSM Equation 3-34.  There is one modification:
    the IBSM conversion factor pi/(4(f/#)^2) is replaced with pi/(1+ 4(f/#)^2),
    which is valid over a wider range of f-numbers (source: John Schott,"Remote
    Sensing: The Image Chain Approach," Oxford University Press, 1997).  
    If the telescope is obscured, E is further reduced by 1-eta**2, where
    eta is the relative linear obscuration.
    
    Parameters
    ----------
    D : 
        effective aperture diameter (m)
    f :
        focal length (m)
    L : 
        total radiance (W/m^2 sr) or spectral radiance (W/m^2 sr m)  
    Returns
    -------
    E : 
        total irradiance (W/m^2) or spectral irradiance (W/m^2 m) at the focal plane
    
    """
    E = L * np.pi / (1.0+4.0*(f/D)**2.0)   
    
    return E

def blackbodyRadiance(lambda0,T):
    """Calculates blackbody spectral radiance.  IBSM Equation 3-35.
    
    Parameters
    ----------
    lambda0 : 
        wavelength (m)
    T :
        temperature (K)

        
    Returns
    -------
    Lbb : 
        blackbody spectral radiance (W/m^2 sr m)
    """
    #lambda0 = lambda0+1e-20
    Lbb =  (2.0*hc*cc**2.0 / lambda0**5.0 ) * (np.exp(hc*cc/(lambda0*kc*T))-1.0)**(-1.0)
    
    return Lbb
    
def focalplaneIntegratedIrradiance(L,Ls,topt,eopt,lambda0,dlambda,opticsTemperature,D,f):
    """IBSM Equation 3-34.  Calculates band integrated irradiance at the focal plane, 
    including at-aperture scene radiance, optical self-emission, and non-thermal
    stray radiance.  NOTE: this function is only included for completeness.  It
    is much better to use spectral quantities throughout the modeling process.
    
    Parameters
    ----------
    L : 
        band integrated at-aperture radiance (W/m^2 sr)
    Ls :
        band integrated stray radiance from sources other than self emission
        (W/m^2 sr)
    topt :
        full system in-band optical transmission (unitless).  If the telescope
        is obscured, topt is further reduced by 1-eta**2, where
        eta is the relative linear obscuration
    eopt :
        full system in-band optical emissivity (unitless).  1-topt is a good approximation.
    lambda0 :
        wavelength at the center of the system bandpass (m)
    dlambda :
        system spectral bandwidth (m)
    opticsTemperature :
        temperature of the optics (K)
    D : 
        effective aperture diameter (m)
    f :
        focal length (m)
    
    Returns
    -------
    E : 
        integrated irradiance (W/m^2) at the focal plane
    """
    L = topt*L + eopt*blackbodyRadiance(lambda0,opticsTemperature)*dlambda + Ls
    E =  atFocalPlaneIrradiance(D,f,L)
    
    return E

def detectorOTF(u,v,wx,wy,f): 
    """A simplified version of IBSM Equation 3-36.  Blur due to the spatial  
    integrating effects of the detector size.  See detectorOTFwithAggregation
    if detector aggregation is desired (new for version 1).
     
    Parameters 
    ---------- 
    u or v :  
        spatial frequency coordinates (rad^-1) 
    wx and wy :
        detector size (width) in the x and y directions (m) 
    f : 
        focal length (m) 
     
    Returns 
    ------- 
    H :  
        detector OTF 
    """ 
     
    H = np.sinc(wx*u/f)*np.sinc(wy*v/f) 
 
    return H 
 
def tdiOTF(uorv,w,ntdi,phasesN,beta,f): 
    """IBSM Equation 3-38.  Blur due to a mismatch between the time-delay-integration 
    clocking rate and the image motion. 
     
    Parameters 
    ---------- 
    u or v :  
        spatial frequency coordinates in the TDI direction.  (rad^-1) 
    w: 
        detector size (width) in the TDI direction (m) 
    ntdi: 
        number of TDI stages (unitless) 
    phasesN: 
        number of clock phases per transfer (unitless) 
    beta: 
        ratio of TDI clocking rate to image motion rate (unitless) 
    f : 
        focal length (m) 
     
    Returns 
    ------- 
    H :  
        tdi OTF 
    """ 
     
    xx = w*uorv/(f*beta) #this occurs twice, so we'll pull it out to simplify the 
    #the code 
     
    expsum=0.0     
    iind = np.arange(0,ntdi*phasesN) #goes from 0 to tdiN*phasessN-1 
    for ii in iind: 
        expsum = expsum + np.exp(-2.0j*np.pi*xx*(beta-1.0)*ii)   
    H = np.sinc(xx)*expsum / (ntdi*phasesN) 
    return H 
 
def cteOTF(u,v,px,py,cteNx,cteNy,phasesN,cteEff,f): 
    """IBSM Equation 3-39.  Blur due to charge transfer efficiency losses in a 
    CCD array. 
     
    Parameters
    ---------- 
    u or v :  
        spatial frequency coordinates (rad^-1) 
    (px,py) : 
        detector center-to-center spacings (pitch) in the x and y directions (m) 
    (cteNx,cteNy) : 
        number of change transfers in the x and y directions (unitless) 
    phasesN: 
        number of clock phases per transer (unitless) 
    beta : 
        ratio of TDI clocking rate to image motion rate (unitless) 
    cteEff : 
        charge transfer efficiency (unitless) 
    f : 
        focal length (m) 
     
    Returns 
    ------- 
    H :  
        cte OTF 
    """ 
    #this OTF has the same form in the x and y directions so we'll define 
    #an inline function to save us the trouble of doing this twice 
    #N is either cteNx or cteNy and pu is the product of pitch and spatial 
    #frequency - either v*py or u*px      
    fcn = lambda N,pu: np.exp(-1.0*phasesN*N*(1.0-cteEff)*(1.0-np.cos(2.0*np.pi*pu/f)))     
     
    H =  fcn(cteNx,px*u)*fcn(cteNy,py*v) 
 
    return H

def diffusionOTF(u,v,alpha,ald,al0,f): 
    """IBSM Equation 3-40.  Blur due to the effects of minority carrier diffusion
    in a CCD sensor.  Included for completeness but this isn't a good description
    of modern detector structures.
     
    Parameters
    ---------- 
    u or v :  
        spatial frequency coordinates (rad^-1) 
    alpha : 
        carrier spectral diffusion coefficient (m^-1). Note that IBSM Table 3-4
        contains aplpha values as a function of wavelength for silicon
    ald : 
        depletion layer width (m)
    al0: 
        diffusion length (m)  
    f : 
        focal length (m) 
     
    Returns 
    ------- 
    H :  
        diffusion OTF 
    """       
    fcn = lambda xx: 1.0-np.exp(-alpha*ald)/(1.0+alpha*xx)      
    
    rho = np.sqrt(u**2+v**2)    
    
    alrho = np.sqrt((1.0/al0**2+(2.0*np.pi*rho/f)**2)**(-1))    
    
    H =  fcn(alrho)/fcn(al0) 
 
    return H

def photonDetectionRate(E,wx,wy,wavelengths,qe): 
    """IBSM Equation 3-42 with dark current, integration time, and tdi separated out.  
    Conversion of photons into photoelectrons.  There is a possible disconnect here in the documentation.
    Equation 3-42 appears to be a spectral quantity but the documentation calls for 
    an integrated irradiance.  It is definitely used here as a spectral quantity.
     
    Parameters
    ---------- 
    E :  
        spectral irradiance (W/m^2 m) at the focal plane at each wavelength
    wx or wy: 
        detector size (width) in the x and y directions (m)
    wavelengths :
        wavelength array (m)
    qe :
        quantum efficiency (e-/photon)
     
    Returns 
    ------- 
    dN :
        array of photoelectrons/wavelength/second (e-/m)
    Notes
    -------
    To calculate total integrated photoelectrons, N = td*ntdi*np.trapz(dN,wavelens)
    where td is integration time (s) and ntdi is the number of tdi stages (optional)
    """
       
    dN = (wavelengths/(hc*cc))*qe*wx*wy*E 
    return dN

def darkCurrentFromDensity(jd,wx,wy): 
    """The dark current part of Equation 3-42.  Use this function to calculate
    the total number of electrons generated from dark current during an 
    integration time given a dark current density.  It is useful to separate
    this out from 3-42 for noise source analysis purposes and because sometimes
    dark current is defined in another way.
     
    Parameters
    ---------- 
    jd :  
        dark current density (A/m^2)
    (wx,wy): 
        detector size (width) in the x and y directions (m)

    Returns 
    ------- 
    jde :  
        dark current electron rate (e-/s).  For TDI systems, just multiply the result
        by the number of TDI stages.
    """       
    jde = jd*wx*wy/qc #recall that qc is defined as charge of an electron
    return jde

def quantizationNoise(peRange,bitdepth): 
    """Effective noise contribution from the number of photoelectrons quantized
    by a single count of the analog to digital converter.  Quantization noise 
    is buried in the definition of signal-to-noise in IBSM equation 3-47.
     
    Parameters
    ---------- 
    peRange :  
        the difference between the maximum and minimum number of photoelectrons
        that may be sampled by the read out electronics (e-)
    bitdepth :
        number of bits in the analog to digital converter (unitless)
        
    Returns 
    ------- 
    sigmaq :  
        quantization noise given as a photoelectron standard deviation (e-)
    """       
    sigmaq = peRange/(np.sqrt(12)*(2.0**bitdepth-1.0))
    return sigmaq

def groundResolvedDistance(mtfslice,df,snr,ifov,slantRange):
    """IBSM Equation 3-54.  The ground resolved distance is the period of the
    smallest square wave pattern that can be resolved in an image.  GRD can be
    limited by the detector itself (in which case GRD = 2*GSD) but, in general
    is a function of the system MTF and signal-to-noise ratio.
     
    Parameters
    ---------- 
    mtfslice :
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0 cycles/radian
    df : 
        spatial frequency step size (cycles/radian)
    snr :
        contrast signal-to-noise ratio (unitless)
    ifov: 
        instantaneous field-of-view of a detector (radians)
    slantRange :
 
    Returns 
    ------- 
    grd : ground resolved distance (m)  
    """
    w = df*np.arange(1.0*mtfslice.size)
    ur = np.interp(3.0/snr,mtfslice[::-1],w[::-1]) + 1e-12 #1e-12 prevents division by zero in grdcases
    #arrays were reversed to satisfy the requirements of np.interp
    
    
    grdcases = slantRange*np.array([1.0/ur,2.0*ifov])
    grd = np.max(grdcases)
    
    return grd
    
def giqe3(rer,gsd,eho,ng,snr):
    """IBSM Equation 3-56.  The General Image Quality Equation version 3.0.  
        The GIQE returns values on the National Image Interpretability Rating Scale.
        Note: geometric mean values are simply sqrt(value_x * value_y), where x
        and y are orthogonal directions in the image.
     
    Parameters
    ---------- 
    rer : 
        geometric mean relative edge response (unitless)  
    gsd : 
        geometric mean ground sample distance (m)
    eho : 
        geometric mean edge height overshoot (unitless)
    ng :
        noises gain, i.e. increase in noise due to sharpening (unitless). If no sharpening is 
        applied then ng = 1
    snr :
        contrast signal-to-noise ratio (unitless)
        
    Returns 
    ------- 
    niirs : 
        a National Image Interpretability Rating Scale value (unitless)  
    """
    niirs = 11.81+3.32*np.log10(rer/(gsd/.0254)) - 1.48*eho - ng/snr
    #note that, within the GIQE, gsd is defined in inches, hence the conversion
    return niirs

def giqeEdgeTerms(mtf,df,ifovx,ifovy): 
    """Calculates the geometric mean relative edge response and edge high overshoot, 
    from a 2-D MTF.  This function is primarily for use with the GIQE. It
    implements IBSM equations 3-57 and 3-58. 
     
    Parameters 
    ---------- 
    mtf: 
        2-dimensional full system modulation transfer function (unitless).  MTF is 
        the magnitude of the OTF. 
    df :  
        spatial frequency step size (cycles/radian) 
    ifovx:  
        x-direction instantaneous field-of-view of a detector (radians)
    ifovy:  
        y-direction instantaneous field-of-view of a detector (radians)
     
    Returns 
    ------- 
    rer: 
        geometric mean relative edge response (unitless) 
    eho: 
        geometric mean edge height overshoot (unitless) 
    """ 
    uslice = sliceotf(mtf,0) 
    vslice = sliceotf(mtf,np.pi/2) 
     
    urer = relativeEdgeResponse(uslice,df,ifovx) 
    vrer = relativeEdgeResponse(vslice,df,ifovy) 
    rer = np.sqrt(urer*vrer) 
     
    ueho = edgeHeightOvershoot(uslice,df,ifovx) 
    veho = edgeHeightOvershoot(vslice,df,ifovy) 
    eho = np.sqrt(ueho*veho) 
     
    return rer, eho


def edgeHeightOvershoot(mtfslice,df,ifov):
    """IBSM Equation 3-60.  Edge Height Overshoot is a measure of image distortion
    caused by sharpening.  Note that there is a typo in Equation 3-60.  The
    correct definition is given in Leachtenauer et al., "General Image-Quality 
    Equation: GIQE" APPLIED OPTICS Vol. 36, No. 32 10 November 1997. "The 
    overshoot-height term H models the edge response overshoot that is due to 
    MTFC. It is measured over the range of 1.0 to 3.0 pixels from the edge
    in 0.25-pixel increments. If the edge is monotonically increasing, it is defined as the
    value at 1.25 pixels from the edge."
     
    Parameters
    ---------- 
    mtfslice :
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0 cycles/radian
    df : 
        spatial frequency step size (cycles/radian)
    ifov: 
        instantaneous field-of-view of a detector (radians)
 
    Returns 
    ------- 
    eho : 
        edge height overshoot (unitless)  
    """    
    rng = np.arange(1.0,3.25,.25)
    er = np.zeros(rng.size)
    index = 0
    
    for dist in rng:
        er[index] =  edgeResponse(dist,mtfslice,df,ifov)
        index = index + 1
    
    if np.all(np.diff(er) > 0): #when true, er is monotonically increasing
        eho = er[1] #the edge response at 1.25 pixels from the edge
    else:
        eho = np.max(er)

    return eho

def relativeEdgeResponse(mtfslice,df,ifov):
    """IBSM Equation 3-61.  The slope of the edge response of the system taken
    at +/-0.5 pixels from a theoretical edge.  Edge response is used in the calculation of NIIRS 
    via the General Image Quality Equation.
     
    Parameters
    ---------- 
    mtfslice :
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0 cycles/radian
    df : 
        spatial frequency step size (cycles/radian)
    ifov: 
        instantaneous field-of-view of a detector (radians)
        
        
    Returns 
    ------- 
    rer : 
        relative edge response (unitless)  
    """
    rer = edgeResponse(0.5,mtfslice,df,ifov)-edgeResponse(-0.5,mtfslice,df,ifov)
    return rer

def groundSampleDistance(ifov,slantRange):
    """IBSM Equation 3-62.  The ground sample distance, i.e. the footprint
        of a single detector in object space.
     
    Parameters
    ---------- 
    ifov: 
        instantaneous field-of-view of a detector (radians)
    slantRange:
        slant range to target (m)
 
    Returns 
    ------- 
    gsd : 
        ground sample distance (m)  
    """
    gsd = slantRange*ifov
    return gsd

def edgeResponse(pixelPos,mtfslice,df,ifov):
    """IBSM Equation 3-63.  Imagine a perfectly sharp edge in object space.  After 
    the edge is blurred by the system MTF, the edge response is the normalized
    value of this blurred edge in image space at a distance of pixelPos pixels 
    away from the true edge.  Edge response is used in the calculation of NIIRS 
    via the General Image Quality Equation.
     
    Parameters
    ---------- 
    pixelPos :  
        distance from the theoretical edge (pixels)
    mtfslice :
        1-D modulation transfer function (unitless) mtf[0] = 1 is at 0 cycles/radian
    df : 
        spatial frequency step size (cycles/radian)
    ifov: 
        instantaneous field-of-view of a detector (radians)
        
        
    Returns 
    ------- 
    er : 
        normalized edge response (unitless)  
    """
    w = df*np.arange(1.0*mtfslice.size) + 1e-6 #note tiny offset to avoid infs
    y = (mtfslice/w)*np.sin(2*np.pi*w*ifov*pixelPos)
    
    er = 0.5 + (1.0/np.pi)*np.trapz(y,w)
    return er
    

def instantaneousFOV(w,f):
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

#############################################################################
# THE FOLLOWING FUNCTIONS ARE SUPPLIMENTAL TO IBSM
#############################################################################
def gaussianOTF(u,v,blurSizeX,blurSizeY):
    """A real-valued Gaussian OTF.  This is useful for modeling systems when
    you have some general idea of the width of the point-spread-function or
    perhaps the cutoff frequency.  The blur size is defined to be where the PSF 
    falls to about .043 times it's peak value.
    
    Parameters
    ---------- 
    u and v : 
        angular spatial frequency in the x and y directions (cycles/radian) 
    blurSizeX and blurSizeY : 
        angular extent of the blur spot in image space (radians) 
 
    Returns 
    ------- 
    H :
        gaussian optical transfer function.
    
    Notes: The cutoff frequencies (where the MTF falls to .043 cycles/radian) 
    are the inverse of the blurSizes and the point spread function is therefore: 
    psf(x,y) = (fxX*fcY)*exp(-pi((fxX*x)^2+(fcY*y)^2))
    """
    fcX = 1/blurSizeX #x-direction cutoff frequency
    fcY = 1/blurSizeY #y-direction cutoff frequency
    
    H = np.exp(-np.pi*((u/fcX)**2+(v/fcY)**2))
        
    return H

def noiseGain(kernel):
    """Noise Gain is the GIQE term representing increase in noise due to image sharpening.
    The definition is not included in the IBSM manual.  This version comes from
    Leachtenauer et al., "General Image-Quality Equation: GIQE" APPLIED OPTICS 
    Vol. 36, No. 32 10 November 1997. 
    
    Parameters
    ---------- 
    kernal: 
         the 2-D image sharpening kernel.  Note that 
         the kernel is assumed to sum to one.

    Returns 
    ------- 
    ng:
        noise gain (unitless)
    """    
    ng = np.sqrt(np.sum(np.sum(kernel**2)))
    return ng

def filterOTF(u,v,kernel,ifov):
    """Returns the OTF of any filter applied to the image (e.g. a sharpening
    filter).   
    
    Parameters
    ---------- 
    u and v : 
        angular spatial frequency coordinates (rad^-1)   
    kernel: 
         the 2-D image sharpening kernel.  Note that 
         the kernel is assumed to sum to one.
    ifov: 
        instantaneous field-of-view of a detector (radians)
 
    Returns 
    ------- 
    H:
        optical transfer function of the filter at spatial frequencies u and v
    """    
    #most filter kernels are only a few pixels wide so we'll use zero-padding
    #to make the OTF larger.  The exact size doesn't matter too much
    #because the result is interpolated    
    N = 100 # array size for the transform
    
    #transform of the kernel
    xferfcn = np.abs(np.fft.fftshift(np.fft.fft2(kernel,[N,N])))

    nyquist = 0.5/ifov    
    
    #spatial freuqency coordinates for the transformed filter
    urng = np.linspace(-nyquist, nyquist, xferfcn.shape[0]) 
    vrng = np.linspace(nyquist, -nyquist, xferfcn.shape[1]) 
    nu,nv = np.meshgrid(urng, vrng)
    

    #reshape everything to comply with the griddata interpolator requirements
    xferfcn = xferfcn.reshape(-1)    
    nu = nu.reshape(-1)
    nv = nv.reshape(-1)

    #use this function to wrap spatial frequencies beyond Nyquist
    wrapval = lambda value: (( value + nyquist) % (2 * nyquist) - nyquist)
    
    #and interpolate up to the desired range
    H = interpolate.griddata((nu,nv),xferfcn,(wrapval(u),wrapval(v)), \
    method='linear',fill_value=0)
    
    return H


def loadDatabaseAtmosphere_nointerp(altitude,groundRange,ihaze):
    """Loads a precalculated MODTRAN 5.2.1 Tape 7 atmosphere over a wavelength range
    of 0.3 to 14 micrometers.  All screnario details are in 'atmosphere_README.txt'.
    NOTE: the _nointerp suffix was added for version 0.2.  See pybsm.loadDatabaseAtmosphere
    for more information.
    
    Parameters
    ---------- 
    altiude:
        sensor height above ground level in meters.  The database includes the following
        altitude options: 2 32.55 75 150 225 500 meters, 1000 to 12000 in 1000 meter steps, 
        and 14000 to 20000 in 2000 meter steps, 24500 meters 
    groundRange:
        distance *on the ground* between the target and sensor in meters.
        The following ground ranges are included in the database at each altitude 
        until the ground range exceeds the distance to the spherical earth horizon: 
        0 100 500 1000 to 20000 in 1000 meter steps, 22000 to 80000 in 2000 m steps, 
        and  85000 to 300000 in 5000 meter steps.
    ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural extinction with 23 km visibility)
        or ihaze = 2 (Rural extinction with 5 km visibility)

 
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
    
    #decoder maps filenames to atmospheric attributes
    atmpath = os.path.join(dirpath,"atms","fileDecoder.csv")
    decoder = np.genfromtxt(atmpath, delimiter=',',skip_header=1)
    
    decoder = decoder[decoder[:,3]==ihaze] #downselects to the right ihaze mode    
    decoder = decoder[decoder[:,1]==altitude/1000.0] #downselects to the right altitude
    decoder = decoder[decoder[:,2]==groundRange/1000.0] #downselects to the right ground range   
    
    rawdata = np.fromfile(dirpath + '/atms/' +  \
        str(int(decoder[0,0])) + '.bin',dtype = np.float32, count=-1)
    
    rawdata = rawdata.reshape((1371,5),order='F')
    rawdata[:,1:5] = rawdata[:,1:5]/1e-10 #convert radiance columns to W/(sr m^2 m) 
    
    #append wavelength as first column
    wavl = 1e-6*np.linspace(.3,14.0,1371)
    wavl = np.expand_dims(wavl, axis=1)
    atm = np.hstack((wavl,rawdata))
    
    return atm
    
def totalRadiance(atm,reflectance,temperature):
    """Calculates total spectral radiance at the aperture for a object of interest.
    
    Parameters
    ---------- 
    atm:
        matrix of atmospheric data (see loadDatabaseAtmosphere for details)
    reflectance:
        object reflectance (unitless)
    temperature:
        object temperature (Kelvin)

    Returns 
    ------- 
    radiance:
        radiance = path thermal + surface emission + solar scattering + ground reflected (W/m^2 sr m)
    
    Notes
    -------
    In the emissive infrared region (e.g. >= 3 micrometers), the nighttime case is
    very well approximated by subtracting off atm[:,4] from the total spectral radiance
    """
    
    dbreflectance = 0.15 #object reflectance used in the database
    radiance = atm[:,2]+(1.0-reflectance)*blackbodyRadiance(atm[:,0],temperature)*atm[:,1] \
    + atm[:,4]+atm[:,5]*(reflectance/dbreflectance)   
    
    return radiance

def giqeRadiance(atm,isEmissive):
    """This function provides target and background spectral radiance as defined by the
    GIQE.  
    
    Parameters
    ---------- 
    atm:
        an array containing the following data:
        atm[:,0] - wavelengths from .3 to 14 x 10^-6 m in 0.01x10^-6 m steps
        atm[:,1] - (TRANS) total transmission through the defined path.
        atm[:,2] - (PTH THRML) radiance component due to atmospheric emission and scattering received at the observer.
        atm[:,3] - (SURF EMIS) component of radiance due to surface emission received at the observer.
        atm[:,4] - (SOL SCAT) component of scattered solar radiance received at the observer.
        atm[:,5] - (GRND RFLT) is the total solar flux impingent on the ground and reflected directly to the sensor from the ground. (direct radiance + diffuse radiance) * surface reflectance
        NOTE: units for columns 1 through 5 are in radiance W/(sr m^2 m)
    isEmissive:
        isEmissive = 1 for thermal emissive band NIIRS, otherwise isEmissive = 0
         
    Returns 
    ------- 
    targetRadiance:
        apparent target spectral radiance at the aperture including all atmospheric
        contributions
    backgroundRadiance:
        apparent background spectral radiance at the aperture including all atmospheric
        contributions
    
    Notes
    -----
    The nighttime emissive case is well approximated by subtracting off atm[:,4]
    from the returned values.
        
    """    
    tgtTemp = 282.0 #target temperature (original GIQE suggestion was 282 K)
    bkgTemp = 280.0 #background temperature (original GIQE 280 K)
    tgtRef  = 0.15 #percent reflectance of the target (should be .15 for GIQE)
    bkgRef = 0.07 #percent reflectance of the background (should be .07 for GIQE)
    
    if isEmissive:
        #target and background are blackbodies
        targetRadiance = totalRadiance(atm,0.0,tgtTemp)
        backgroundRadiance = totalRadiance(atm,0.0,bkgTemp)
    else:
        targetRadiance = totalRadiance(atm,tgtRef,tgtTemp)
        backgroundRadiance = totalRadiance(atm,bkgRef,bkgTemp)
            
    return targetRadiance,backgroundRadiance

def coldshieldSelfEmission(wavelengths,coldshieldTemperature,D,f):
    """For infrared systems, this term represents spectral irradiance on the FPA due to 
    emissions from the walls of the dewar itself.

    Parameters
    ---------- 
    wavelengths :
        wavelength array (m)
    coldshieldTemperature:
        temperature of the cold shield (K).  It is a common approximation to assume
        that the coldshield is at the same temperature as the detector array.
    D : 
        effective aperture diameter (m)
    f :
        focal length (m)
    
    Returns
    -------
    coldshieldE:
        cold shield spectral irradiance at the FPA (W / m^2 m)
    """
    #coldshield solid angle x blackbody emitted radiance
    coldshieldE = (np.pi - np.pi/(4.0*(f/D)**2.0+1.0))* \
    blackbodyRadiance(wavelengths,coldshieldTemperature)
    
    return coldshieldE

def opticsSelfEmission(wavelengths,opticsTemperature,opticsEmissivity,
                       coldfilterTransmission,D,f):
    """For infrared systems, this term represents spectral irradiance emitted
    by the optics (but not the cold stop) on to the FPA.  

    Parameters
    ---------- 
    wavelengths :
        wavelength array (m)
    opticsTemperature:
        temperature of the optics (K)
    opticsEmissivity:
        emissivity of the optics (unitless) except for the cold filter.  
        A common approximation is 1-optics transmissivity. 
    coldfilterTransmission:
        transmission through the cold filter (unitless)       
    D : 
        effective aperture diameter (m)
    f :
        focal length (m)
    
    Returns
    -------
    opticsE:
        optics emitted irradiance on to the FPA (W / m^2 m)
    """

    opticsL = coldfilterTransmission*opticsEmissivity*blackbodyRadiance(wavelengths,opticsTemperature)
    opticsE = atFocalPlaneIrradiance(D,f,opticsL)
    return opticsE

def coldstopSelfEmission(wavelengths,coldfilterTemperature,coldfilterEmissivity,D,f):
    """For infrared systems, this term represents spectral irradiance emitted
    by the cold stop on to the FPA.  

    Parameters
    ---------- 
    wavelengths :
        wavelength array (m)
    coldfilterTemperature:
        temperature of the cold filter.  It is a common approximation to assume
        that the filter is at the same temperature as the detector array.  
    coldfilterEmissivity:
        emissivity through the cold filter (unitless).  A common approximation 
        is 1-cold filter transmission.         
    D : 
        effective aperture diameter (m)
    f :
        focal length (m)
    
    Returns
    -------
    coldstopE:
        optics emitted irradiance on to the FPA (W / m^2 m)
    """

    coldstopL = coldfilterEmissivity*blackbodyRadiance(wavelengths,coldfilterTemperature)
    coldstopE = atFocalPlaneIrradiance(D,f,coldstopL)
    return coldstopE

def sliceotf(otf,ang): 
    """Returns a one dimensional slice of a 2D OTF (or MTF) along the direction  
    specified by the input angle. 
 
    Parameters 
    ----------  
        otf :  
            OTF defined by spatial frequencies (u,v) (unitless) 
        ang : 
            slice angle (radians) A 0 radian slice is along the u axis.  The 
            angle rotates counterclockwise. Angle pi/2 is along the v axis. 
    Returns 
    -------- 
        oslice: 
            One dimensional OTF in the direction of angle.  The sample spacing
            of oslice is the same as the original otf
    """ 
    u = np.linspace(-1.0, 1.0, otf.shape[0]) 
    v = np.linspace(1.0, -1.0, otf.shape[1]) 
    r = np.arange(0.0, 1.0, u[1]-u[0]) 
     
    f = interpolate.interp2d(u, v, otf) 
    oslice = np.diag(f(r*np.cos(ang), r*np.sin(ang))) 
    #the interpolator, f, calculates a bunch of points that we don't really need 
    #since everything but the diagonal is thrown away.  It works but it's inefficient. 
     
    return oslice     
 
def weightedByWavelength(wavelengths,weights,myFunction): 
    """Returns a wavelength weighted composite array based on myFunction 
    Parameters 
    ---------- 
    wavelengths: 
        array of wavelengths (m) 
    weights: 
        array of weights corresponding to the "wavelengths" array. 
        Weights are normalized within this function so that weights.sum()==1 
    myFunction 
        a lambda function parameterized by wavelength, e.g. 
        otfFunction = lambda wavelengths: pybsm.circularApertureOTF(uu,vv,wavelengths,D,eta) 
     
    Returns 
    -------- 
        weightedfcn: 
            the weighted function  
    """ 
    weights = weights/weights.sum() 
    weightedfcn = weights[0]*myFunction(wavelengths[0])     
     
    for wii in wavelengths[1:]: 
        weightedfcn = weightedfcn + weights[wavelengths==wii]*myFunction(wii)         
     
    return weightedfcn

def resampleByWavelength(wavelengths,values,newWavelengths):
    """Resamples arrays that are input as a function of wavelength.

    Parameters
    ---------- 
    wavelengths: 
        array of wavelengths (m) 
    values: 
        array of values to be resampled (arb)
    newWavelengths:
        the desired wavelength range and step size (m)       
     
    Returns 
    -------- 
        newValues: array of values resampled to match newWavelengths.
        Extrapolated values are set to 0.
    """
    newValues = np.interp(newWavelengths,wavelengths,values,0.0,0.0)     
    return newValues

def checkWellFill(totalPhotoelectrons,maxfill):
    """Check to see if the total collected photoelectrons are greater than the
    desired maximum well fill.  If so, provide a scale factor to reduce the 
    integration time.

    Parameters
    ---------- 
    totalPhotoelectrons: 
        array of wavelengths (m) 
    maxFill:
        desired well fill, i.e. Maximum well size x Desired fill fraction
     
    Returns 
    -------- 
        scalefactor:
            the new integration time is scaled by scalefactor
    """
    scalefactor = 1.0    
    if (totalPhotoelectrons > maxfill):
        scalefactor = maxfill/totalPhotoelectrons
    return scalefactor

def polychromaticTurbulenceOTF(u,v, wavelengths, weights, altitude, slantRange, \
    D, haWindspeed,cn2at1m, intTime, aircraftSpeed):
    """Returns a polychromatic turbulence MTF based on the Hufnagel-Valley turbulence
    profile and the pyBSM function "windspeedTurbulenceOTF", i.e. IBSM Eqn 3.9.
    
    Parameters
    ---------
    u or v : 
        angular spatial frequency coordinates (rad^-1)        
    wavelengths :
        wavelength array (m)
    weights :
        how contributions from each wavelength are weighted
    altitude :
        height of the aircraft above the ground (m)
    slantRange :
        line-of-sight range between the aircraft and target (target is assumed
        to be on the ground)
    D : 
        effective aperture diameter (m)

    intTime : 
        dwell (i.e. integration) time (seconds)
    aircraftSpeed : 
        apparent atmospheric velocity (m/s).  This can just be the windspeed at
        the sensor position if the sensor is stationary.
    haWindspeed: 
        the high altitude windspeed (m/s).  Used to calculate the turbulence profile.
    cn2at1m:
        the refractive index structure parameter "near the ground" (e.g. at h = 1 m).
        Used to calculate the turbulence profile.
        
    Returns
    -------
    turbulenceOTF :
        turbulence OTF (unitless)
    r0band :
        the effective coherence diameter across the band (m)
        """
    #calculate the Structure constant along the slant path
    (zPath,hPath) = altitudeAlongSlantPath(0.0,altitude,slantRange)
    cn2 = hufnagelValleyTurbulenceProfile(hPath,haWindspeed,cn2at1m)
    
    #calculate the coherence diameter over the band
    r0at1um = coherenceDiameter(1.0e-6,zPath,cn2)
    r0function = lambda wav: r0at1um*wav**(6.0/5.0)*(1e-6)**(-6.0/5.0)
    r0band = weightedByWavelength(wavelengths,weights,r0function)
    
    #calculate the turbulence OTF    
    turbFunction = lambda wavelengths: windspeedTurbulenceOTF(u, v, \
    wavelengths,D,r0function(wavelengths),intTime,aircraftSpeed)
    turbulenceOTF = weightedByWavelength(wavelengths,weights,turbFunction)
    
    return turbulenceOTF, r0band
          

def signalRate(wavelengths,targetRadiance,opticalTransmission,D,f,wx,wy,qe,otherIrradiance,darkCurrent):
    """For semiconductor-based detectors, returns the signal rate (total photoelectrons/s) 
    generated at the output of the detector along with a number of other 
    related quantities.  Multiply this quantity by the integration time (and the 
    number of TDI stages, if applicable) to determine the total number of detected photoelectrons.
    Parameters
    ----------
    wavelengths:
        array of wavelengths (m)
    targetRadiance:
        apparent target spectral radiance at the aperture including all atmospheric
        contributions (W/sr m^2 m)
    backgroundRadiance:
        apparent background spectral radiance at the aperture including all atmospheric
        contributions (W/sr m^2 m)
    opticalTransmission:
        transmission of the telescope optics as a function of wavelength (unitless)
    D : 
        effective aperture diameter (m)
    (wx,wy): 
        detector size (width) in the x and y directions (m) 
    f : 
        focal length (m) 
    qe :
        quantum efficiency as a function of wavelength (e-/photon)
    otherIrradiance:
        spectral irradiance from other sources (W/m^2 m).
        This is particularly useful for self emission in infrared cameras.  It may
        also represent stray light.
    darkCurrent:
        detector dark current (e-/s)
    
    Returns 
    -------- 
    tgtRate:
        total integrated photoelectrons per seconds (e-/s)
    tgtFPAirradiance:
        spectral irradiance at the FPA (W/m^2 m)
    tgtdN:
        spectral photoelectrons (e-/s m)

    """
    #get at FPA spectral irradiance
    tgtFPAirradiance = opticalTransmission*atFocalPlaneIrradiance(D,f,targetRadiance) + otherIrradiance
    
    #convert spectral irradiance to spectral photoelectron rate
    tgtdN = photonDetectionRate(tgtFPAirradiance,wx,wy,wavelengths,qe)
    
    #calculate total detected target and background photoelectron rate
    tgtRate = np.trapz(tgtdN,wavelengths) + darkCurrent
    
    return tgtRate,tgtFPAirradiance,tgtdN

def photonDetectorSNR(sensor,radianceWavelengths,targetRadiance,backgroundRadiance):
    """Calculates extended target contrast SNR for semiconductor-based photon detector systems (as 
    opposed to thermal detectors).  This code originally served the NIIRS model
    but has been abstracted for other uses.  Photon, dark current, quantization,
    and read noise are all explicitly considered.  You can also pass in other noise 
    terms (as rms photoelectrons) as a numpy array sensor.otherNoise.
    
    Parameters
    ---------- 
    sensor :
        an object from the class sensor
    radianceWavelengths :
        a numpy array of wavelengths (m)
    targetRadiance :
        a numpy array of target radiance values corresponding to radianceWavelengths
        (W/m^2 sr m)
    backgroundRadiance :
        a numpy array of target radiance values corresponding to radianceWavelengths
        (W/m^2 sr m)
     
    Returns 
    -------- 
    snr: 
        an object containing results of the SNR calculation along with many
        intermediate calculations.  The SNR value is contained in snr.snr
    """
    snr = metrics('signal-to-noise calculation')
    
    #resample the optical transmission and quantum efficiency functions
    snr.optTrans = sensor.coldfilterTransmission*(1.0-sensor.eta**2)*resampleByWavelength(sensor.optTransWavelengths,sensor.opticsTransmission,radianceWavelengths)
    snr.qe = resampleByWavelength(sensor.qewavelengths,sensor.qe,radianceWavelengths)
    
    #for infrared systems, calculate FPA irradiance contributions from within
    #the sensor system itself
    snr.otherIrradiance =  coldshieldSelfEmission(radianceWavelengths \
    ,sensor.coldshieldTemperature,sensor.D,sensor.f) + \
    opticsSelfEmission(radianceWavelengths,sensor.opticsTemperature, \
    sensor.opticsEmissivity, sensor.coldfilterTransmission,sensor.D,sensor.f) + \
    coldstopSelfEmission(radianceWavelengths,sensor.coldfilterTemperature, \
    sensor.coldfilterEmissivity,sensor.D,sensor.f)
    
    #initial estimate of  total detected target and background photoelectrons
    #first target.  Note that snr.weights is useful for later calculations that
    #require weighting as a function of wavelength (e.g. aperture OTF)    
    snr.tgtNrate,snr.tgtFPAirradiance,snr.weights = signalRate(radianceWavelengths,targetRadiance,snr.optTrans, \
        sensor.D,sensor.f,sensor.wx,sensor.wy,snr.qe,snr.otherIrradiance,sensor.darkCurrent)
    snr.tgtN = snr.tgtNrate*sensor.intTime*sensor.ntdi
    #then background
    snr.bkgNrate,snr.bkgFPAirradiance,_ = signalRate(radianceWavelengths,backgroundRadiance,snr.optTrans, \
    sensor.D,sensor.f,sensor.wx,sensor.wy,snr.qe,snr.otherIrradiance,sensor.darkCurrent)
    snr.bkgN = snr.bkgNrate*sensor.intTime*sensor.ntdi
    
    #check to see that well fill is within a desirable range and, if not, scale
    #back the integration time and recalculate the total photon counts
    scalefactor = checkWellFill(np.max([snr.tgtN,snr.bkgN]),sensor.maxWellFill*sensor.maxN)
    snr.tgtN = scalefactor*snr.tgtN
    snr.bkgN = scalefactor*snr.bkgN
    snr.intTime = scalefactor*sensor.intTime
    snr.wellfraction = np.max([snr.tgtN,snr.bkgN]) / sensor.maxN
    #another option would be to reduce TDI stages if applicable, this should be
    #a concern if TDI mismatch MTF is an issue
    
    #calculate contrast signal (i.e. target difference above or below the background)
    snr.contrastSignal = snr.tgtN-snr.bkgN
    
    #break out noise terms (rms photoelectrons)
    # signalNoise includes scene photon noise, dark current noise, and self emission noise       
    snr.signalNoise = np.sqrt(np.max([snr.tgtN,snr.bkgN]))
    #just noise from dark current
    snr.darkcurrentNoise = np.sqrt(sensor.ntdi*sensor.darkCurrent*snr.intTime)
    #quantization noise
    snr.quantizationNoise = quantizationNoise(sensor.maxN,sensor.bitdepth)
    #photon noise due to self emission in the optical system    
    snr.selfEmissionNoise = np.sqrt(np.trapz(photonDetectionRate(snr.otherIrradiance,sensor.wx, \
    sensor.wy,radianceWavelengths,snr.qe),radianceWavelengths)*snr.intTime*sensor.ntdi)

     
    #note that signalNoise includes sceneNoise, dark current noise, and self
    #emission noise       
    snr.totalNoise = np.sqrt(snr.signalNoise**2+snr.quantizationNoise**2+ \
    sensor.readNoise**2 + np.sum(sensor.otherNoise**2)) 
    
    #calculate signal-to-noise ratio
    snr.snr = snr.contrastSignal/snr.totalNoise
    
    return snr

def commonOTFs(sensor,scenario,uu,vv,mtfwavelengths,mtfweights,slantRange,intTime):
    """Returns optical transfer functions for the most common sources.  This code 
    originally served the NIIRS model but has been abstracted for other uses.  
    OTFs for the aperture, detector, turbulence, jitter, drift, wavefront
    errors, and image filtering are all explicity considered.
    
    Parameters
    ---------- 
    sensor :
        an object from the class sensor
    scenario :
        an object from the class scenario
    uu and vv:
        spatial frequency arrays in the x and y directions respectively (cycles/radian)
    mtfwavelengths :
        a numpy array of wavelengths (m)
    mtfweights :
        a numpy array of weights for each wavelength contribution (arb)
    slantRange :
        distance between the sensor and the target (m)
    intTime :
        integration time (s)
     
    Returns 
    -------- 
    otf: 
        an object containing results of the OTF calculations along with many
        intermediate calculations.  The full system OTF is contained in otf.systemOTF.
    """
    
    otf = metrics('optical transfer function calculation')
    #aperture OTF
    apFunction = lambda wavelengths: circularApertureOTF(uu,vv,wavelengths,sensor.D,sensor.eta)
    otf.apOTF = weightedByWavelength(mtfwavelengths,mtfweights,apFunction)
    
    #turbulence OTF
    if scenario.cn2at1m > 0.0: #this option allows you to turn off turbulence completely
        #by setting cn2 at the ground level to 0
        otf.turbOTF, otf.r0band = polychromaticTurbulenceOTF(uu,vv,mtfwavelengths, \
        mtfweights, scenario.altitude, slantRange, sensor.D, \
        scenario.haWindspeed, scenario.cn2at1m, intTime*sensor.ntdi, scenario.aircraftSpeed)
    else:
        otf.turbOTF = 1.0
        otf.r0band = 1e6
        
    
    #detector OTF
    otf.detOTF = detectorOTF(uu,vv,sensor.wx,sensor.wy,sensor.f)
    
    #jitter OTF
    otf.jitOTF = jitterOTF(uu,vv,sensor.sx,sensor.sy)
    
    #drift OTF
    otf.drftOTF = driftOTF(uu,vv,sensor.dax*intTime*sensor.ntdi,sensor.day*intTime*sensor.ntdi)
    
    #wavefront OTF
    wavFunction = lambda wavelengths: wavefrontOTF(uu,vv, \
    wavelengths,sensor.pv*(sensor.pvwavelength/wavelengths)**2,sensor.Lx,sensor.Ly)    
    otf.wavOTF = weightedByWavelength(mtfwavelengths,mtfweights,wavFunction)
    
    #filter OTF (e.g. a sharpening filter but it could be anything)
    if (sensor.filterKernel.shape[0] > 1):
        #note that we're assuming equal ifovs in the x and y directions
        otf.filterOTF = filterOTF(uu,vv,sensor.filterKernel,sensor.px/sensor.f)
    else:
        otf.filterOTF = np.ones(uu.shape)
        

    #system OTF
    otf.systemOTF = otf.apOTF*otf.turbOTF*otf.detOTF \
    *otf.jitOTF*otf.drftOTF*otf.wavOTF*otf.filterOTF
    
    return otf

def niirs(sensor,scenario):
    """returns NIIRS values and all intermetiate calculations.  This function
    implements the original MATLAB based NIIRS model and can serve as a template
    for building other sensor models.

    Parameters
    ---------- 
    sensor:
        an object from the class sensor
    scenario:
        an object from the class scenario
     
    Returns 
    -------- 
    nm: 
        an object containing results of the GIQE calculation along with many
        intermediate calculations
    """
    #initialize the output    
    nm = metrics('niirs ' + sensor.name + ' ' + scenario.name)
    nm.sensor = sensor
    nm.scenario = scenario    
    nm.slantRange = curvedEarthSlantRange(0.0,scenario.altitude,scenario.groundRange)
    
    ##########CONTRAST SNR CALCULATION#########    
    #load the atmosphere model
    nm.atm = loadDatabaseAtmosphere(scenario.altitude,scenario.groundRange,scenario.ihaze)
    
    #crop out out-of-band data (saves time integrating later)    
    nm.atm = nm.atm[nm.atm[:,0] >= nm.sensor.optTransWavelengths[0],:]
    nm.atm = nm.atm[nm.atm[:,0] <= nm.sensor.optTransWavelengths[-1],:]

    if (nm.sensor.optTransWavelengths[0] >= 2.9e-6):
        isEmissive = 1 #toggle the GIQE to assume infrared imaging
        #the next four lines are bookkeeping for interpreting the results since
        #the giqeRadiance function assumes these values anyway         
        nm.scenario.targetReflectance = 0.0
        nm.scenario.backgroundReflectance = 0.0
        nm.scenario.targetTemperature = 282.0
        nm.scenario.targetTemperature = 280.0
        
    else:
        isEmissive = 0
        #more bookkeeping (see previous comment)
        nm.scenario.targetReflectance = 0.15
        nm.scenario.backgroundReflectance = 0.07
    
    #get at aperture radiances for reflective target and background with GIQE type target parameters
    nm.tgtRadiance,nm.bkgRadiance = giqeRadiance(nm.atm,isEmissive)
    nm.radianceWavelengths = nm.atm[:,0]
    
    #now calculate now characteristics ****for a single frame******    
    nm.snr = photonDetectorSNR(sensor, nm.radianceWavelengths,nm.tgtRadiance,nm.bkgRadiance)
    
    #break out photon noise sources (not required for NIIRS but useful for analysis)
    #photon noise due to the scene itself (target,background, and path emissions/scattering)    
    tgtNoise= np.sqrt(np.trapz(photonDetectionRate(nm.snr.tgtFPAirradiance-nm.snr.otherIrradiance,nm.sensor.wx, \
    nm.sensor.wy,nm.radianceWavelengths,nm.snr.qe),nm.radianceWavelengths)*nm.snr.intTime*sensor.ntdi)     
    bkgNoise= np.sqrt(np.trapz(photonDetectionRate(nm.snr.bkgFPAirradiance-nm.snr.otherIrradiance,nm.sensor.wx, \
    nm.sensor.wy,nm.radianceWavelengths,nm.snr.qe),nm.radianceWavelengths)*nm.snr.intTime*sensor.ntdi)    #assign the scene Noise to the larger of the target or background noise    
    sceneAndPathNoise = np.max([tgtNoise,bkgNoise])
    #calculate noise due to just the path scattered or emitted radiation
    scatrate,_,_ = signalRate(nm.radianceWavelengths,nm.atm[:,2]+nm.atm[:,4],nm.snr.optTrans, \
    nm.sensor.D,nm.sensor.f,nm.sensor.wx,nm.sensor.wy,nm.snr.qe,0.0,0.0)
    nm.snr.pathNoise = np.sqrt(scatrate*nm.snr.intTime*sensor.ntdi)
    nm.snr.sceneNoise = np.sqrt(sceneAndPathNoise**2-nm.snr.pathNoise**2)
    #####OTF CALCULATION#######
    
    #cut down the wavelength range to only the regions of interest
    nm.mtfwavelengths =  nm.radianceWavelengths[nm.snr.weights > 0.0]
    nm.mtfweights =  nm.snr.weights[nm.snr.weights > 0.0] 
    
    #setup spatial frequency array    
    nm.cutoffFrequency = sensor.D/np.min(nm.mtfwavelengths)
    urng = np.linspace(-1.0, 1.0, 101)*nm.cutoffFrequency
    vrng = np.linspace(1.0, -1.0, 101)*nm.cutoffFrequency
    nm.uu,nm.vv = np.meshgrid(urng, vrng) #meshgrid of spatial frequencies out to the optics cutoff
    nm.df = urng[1]-urng[0] #spatial frequency step size   
    
    nm.otf = commonOTFs(sensor,scenario,nm.uu,nm.vv,nm.mtfwavelengths,nm.mtfweights,nm.slantRange,nm.snr.intTime)

    ##CALCULATE NIIRS##############
    nm.ifovx = sensor.px / sensor.f
    nm.ifovy = sensor.py / sensor.f
    nm.gsdx = nm.ifovx*nm.slantRange
    nm.gsdy = nm.ifovy*nm.slantRange
    nm.gsdgm = np.sqrt(nm.gsdx*nm.gsdy)
    nm.rergm,nm.ehogm = giqeEdgeTerms(np.abs(nm.otf.systemOTF),nm.df,nm.ifovx,nm.ifovy)

    nm.ng = noiseGain(sensor.filterKernel)
    #note that NIIRS is calculated using the SNR ***after frame stacking**** if any 
    nm.niirs = giqe3(nm.rergm,nm.gsdgm,nm.ehogm,nm.ng,np.sqrt(sensor.framestacks)*nm.snr.snr)
    
    #NEW FOR VERSION 0.2 - GIQE 4 
    nm.elevAngle = np.pi/2-nadirAngle(0.0,scenario.altitude,nm.slantRange)
    nm.niirs4, nm.gsdgp = giqe4(nm.rergm,nm.gsdgm,nm.ehogm,nm.ng,np.sqrt(sensor.framestacks)*nm.snr.snr,nm.elevAngle)
    #see pybsm.niir5 for GIQE 5
    
    return nm

#################################################################
#BEGINNING OF VERSION 0.1 FUNCTIONS
#################################################################
def circularApertureOTFwithDefocus(u,v,wvl,D,f,defocus):
    '''
    Calculate MTF for an unobscured circular aperture with a defocus aberration.From "The 
    frequency response of a defocused optical system" (Hopkins, 1955)
    Variable changes made to use angular spatial frequency and approximation of 1/(F/#) = sin(a).
    Contributed by Matthew Howard.
    
    Parameters
    ----------
    (u,v) : 
        angular spatial frequency coordinates (rad^-1)        
    wvl : 
        wavelength (m)
    D : 
        effective aperture diameter (m)
    f:
        focal length (m)
    defocus : 
        focus error distance between in focus and out of focus planes (m).  In other
        words, this is the distance between the geometric focus and the actual focus.
    
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    Note:
    ----
        Code contributed by Matt Howard
    '''
    rho = np.sqrt(u**2.0+v**2.0) # radial spatial frequency
    r0=D/wvl          # diffraction limited cutoff spatial frequency (cy/rad)
        
    s = 2.0*rho/r0
    w20 = .5/(1.0+4.0*(f/D)**2.0)*defocus #note that this is the OPD error at
    #the edge of the pupil.  w20/wavelength is a commonly used specification (e.g. waves of defocus) 
    alpha = 4*np.pi/wvl*w20*s
    beta = np.arccos(0.5*s)
    
    if defocus:
        defocus_otf = 2/(np.pi*alpha) * np.cos(alpha*0.5*s)*(beta*jn(1,alpha) \
                                +1/2.*np.sin(2*beta*(jn(1,alpha)-jn(3,alpha))) \
                                -1/4.*np.sin(4*beta*(jn(3,alpha)-jn(5,alpha))) \
                                +1/6.*np.sin(6*beta*(jn(5,alpha)-jn(7,alpha)))) \
                                - 2/(np.pi*alpha) * np.sin(alpha*0.5*s)*(np.sin(beta*(jn(0,alpha)-jn(2,alpha))) \
                                -1/3.*np.sin(3*beta*(jn(2,alpha)-jn(4,alpha)))\
                                +1/5.*np.sin(5*beta*(jn(4,alpha)-jn(6,alpha))) \
                                -1/7.*np.sin(7*beta*(jn(6,alpha)-jn(8,alpha))))
                                
        defocus_otf[rho==0] = 1
    else:
        defocus_otf = 1/np.pi*(2*beta-np.sin(2*beta))
        
    H = np.nan_to_num(defocus_otf)
    
    return H 

def detectorOTFwithAggregation(u,v,wx,wy,px,py,f,N=1): 
    """Blur due to the spatial integrating effects of the detector size and aggregation.
    Contributed by Matt Howard.  Derivation verified by Ken Barnard.  Note: this 
    function is particularly important for aggregating detectors with less 
    than 100% fill factor (e.g. px > wx). 
     
    Parameters 
    ---------- 
    (u,v) :  
        spatial frequency coordinates (rad^-1) 
    (wx,wy): 
        detector size (width) in the x and y directions (m) 
    (px,py): 
        detector pitch in the x and y directions (m) 
    f : 
        focal length (m) 
    N:
        number of pixels to aggregate
     
    Returns 
    ------- 
    H :  
        detector OTF
    Note:
    ----
        Code contributed by Matt Howard
    """ 

    agg_u=0.0
    agg_v=0.0
    for i in range (N):
        phi_u=2.0*np.pi*((i*px*u/f)-((N-1.0)*px*u/2.0/f))
        agg_u=agg_u+np.cos(phi_u)
        phi_v=2.0*np.pi*((i*py*v/f)-((N-1.0)*py*v/2.0/f))
        agg_v=agg_v+np.cos(phi_v)
     
    H = (agg_u*agg_v/N**2)*np.sinc(wx*u/f)*np.sinc(wy*v/f) 
 
    return H

def wavefrontOTF2(u,v,cutoff,wrms):
    """MTF due to wavefront errors.  In an ideal imaging system, a spherical waves
    converge to form an image at the focus.  Wavefront errors represent a departures
    from this ideal that lead to degraded image quality.  This function is an
    alternative to wavefrontOTF.  For more details see the R. Shannon, "Handbook
    of Optics," Chapter 35, "Optical Specifications."  Useful notes from the author:
    for most imaging systems, wrms falls between 0.1 and 0.25 waves rms.  This MTF
    becomes progressively less accurate as wrms exceeds .18 waves.
     
    Parameters 
    ---------- 
    (u,v) :  
        spatial frequency coordinates (rad^-1)
    cutoff:
        spatial frequency cutoff due to diffraction, i.e. aperture diameter / wavelength (rad^-1)
    wrms:
        root mean square wavefront error (waves of error).
        
     
    Returns 
    ------- 
    H :  
        wavefront OTF  
    """     
    
    v = np.sqrt(u**2.0+v**2.0)/cutoff
    
    H = 1.0-((wrms/.18)**2.0) * (1.0-4.0*(v-0.5)**2.0)
    
    return H

def wienerFiler(otf,noiseToSignalPS):
    """An image restoration transfer function based on the Wiener Filter.  See
    from Gonzalex and Woods, "Digital Image Processing," 3 ed.  Note that the filter
    is not normalized so that WF = 1 at 0 spatial frequency.  This is easily fixed for the case where
    noiseToSignalPS is a scalar: (1.0+noisetosignalPS)*WF = 1 at 0 spatial frequency.
    This is noteworthy because, without normalization of some kind, System MTF * WF is not a proper MTF.
    Also, for any frequency space sharpening kernel, be sure that sharpening does
    not extend past the Nyquist frequency, or, if so, ensure that the filter
    response wraps around Nyquist appropriately.  
    
    Parameters 
    ---------- 
    otf:
        system optical transfer function 
    noiseTosignalPS:
        ratio of the noise power spectrum to the signal power spectrum.  This may
        be a function of spatial frequency (same size as otf) or an scalar
        
     
    Returns 
    ------- 
    WF :  
        frequency space representation of the Wienerfilter  
    """   
    WF = np.conj(otf) / (np.abs(otf)**2+noiseToSignalPS)
    
    return WF

def otf2psf(otf,df,dxout):
    """transform an optical transfer function into a point spread function 
    (i.e., image space blur filter)
    
    Parameters
    ----------
    otf :
        optical transfer function
    df : 
        sample spacing for the optical transfer function (radians^-1)
    dxout : 
        desired sample spacing of the point spread function (radians).
        WARNING: dxout must be small enough to properly sample the blur kernel!!!
        
    Returns
    -------
    psf : 
        blur kernel
    
    """  
    #transform the psf
    psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(otf))))
    
    #determine image space sampling
    dxin = 1/(otf.shape[0]*df)
    
    #resample to the desired sample size
    psf = resample2D(psf,dxin,dxout)
    
    #ensure that the psf sums to 1
    psf = psf/psf.sum()    
    
    #crop function for the desired kernel size
    getmiddle = lambda x,ksize: x[tuple([slice(int(np.floor(d/2-ksize/2)),int(np.ceil(d/2+ksize/2))) for d in x.shape])]
    
    #find the support region of the blur kernel
    for ii in np.arange(10,np.min(otf.shape),5):    
        psfout = getmiddle(psf,ii)
        if psfout.sum() > .95: #note the 0.95 is heuristic (but seems to work well)
            break
    
    #make up for cropped out portions of the psf
    psfout = psfout/psfout.sum()     #bug fix 3 April 2020

    
    return psfout

def resample2D(imgin,dxin,dxout):
    """Resample an image. 
    
    Parameters
    ----------
    img :
        the input image
    dxin : 
        sample spacing of the input image (radians)
    dxout : 
        sample spacing of the output image (radians)
        
    Returns
    -------
    imgout : 
        output image
    
    """  
    
    newx= int(imgin.shape[1]*dxin/dxout) 
    newy = int(imgin.shape[0]*dxin/dxout)
    imgout = cv2.resize(imgin,(newx,newy))
    
    return imgout

def simulateImage(imgin,imggsd,rng, otf, df,ifov):
    """Blurs and resamples an image according to the given OTF and sampling parameters.
    
    Parameters
    ----------
    imgin :
        the test image to be resampled
    imggsd :
        spatial sampling for imgin (m).  Don't confuse this with spatial sampling
        in the imaging system!
    rng :
        range from camera to the target image (imgin)
    otf :
        the complex transfer function of the imaging system
    df :
        spatial frequency sampling of the otf (radians^-1)
    ifov : 
        IFOV of the imaging system (radians)
    
    Returns
    -------
    simimg :
        the blurred and resampled image
    simpsf :
        the resampled blur kernel (useful for checking the health of the simulation)
        
    WARNING
    -------
    imggsd must be small enough to properly sample the blur kernel! As a guide,
    if the image system transfer function goes to zero at angular spatial frequency, coff,
    then the sampling requirement will be readily met if imggsd <= rng/(4*coff).
    """
    
    #generate a blur function from the OTF that is resampled to match the angular
    #dimensions of imgin
    psf = otf2psf(otf,df,imggsd/rng)
    
    #filter the image
    blurimg = cv2.filter2D(imgin,-1,psf)
    
    #resample the image to the camera's ifov
    simimg = resample2D(blurimg,imggsd/rng,ifov)
    
    #resample psf (good for health checks on the simulation)
    simpsf = resample2D(psf,imggsd/rng,ifov)
    
    return simimg, simpsf

def img2reflectance(img,pixValues,refValues):
    """Maps pixel values to reflectance values with linear interpolation between
    points.  Pixel values that map below zero reflectance or above unity reflectance
    are truncated.  Implicitly, reflectance is contast across the camera bandpass.
    
    Parameters
    ----------
    img :
        the image that will be transformed into reflectance (counts)
    pixValues :
        array of values in img that map to a known reflectance (counts)
    refValues :
        array of reflectances that correspond to pixValues (unitless)
    
    
    Returns
    -------
    refImg :
        the image in reflectance space
    """
    f = interpolate.interp1d(pixValues,refValues, fill_value='extrapolate', assume_sorted = 0)
    refImg = f(img)
    refImg[refImg > 1.0] = 1.0
    refImg[refImg < 0.0] = 0.0      
    
    return refImg

def reflectance2photoelectrons(atm,sensor,intTime):
    """Provides a mapping between reflectance (0 to 1 in 100 steps) on the ground 
    and photoelectrons collected in the sensor well for a given atmosphere.  
    The target is assumed to be fully illuminated and spectrally flat.  Dark
    current is included.  Target temperature is assumed to be 300 K.
    
    
    Parameters
    ----------
    atm :
        atmospheric data as defined in loadDatabaseAtmosphere.  The slant
        range between the target and sensor are implied by this choice.
    sensor :
        sensor parameters as defined in the pybsm sensor class
    intTime :
        camera integration time (s).
    
    Returns
    -------
    
    ref :
        array of reflectance values (unitless) from 0 to 1 in 100 steps
    pe :
        photoelectrons generated during the integration time corresponding to 
        the reflectance values in ref
    """
    
    ref = np.linspace(0.0,1.0,100)
    pe = np.zeros(ref.shape)
    atm = atm[atm[:,0] >= sensor.optTransWavelengths[0],:]
    atm = atm[atm[:,0] <= sensor.optTransWavelengths[-1],:]
            
    for idx in np.arange(ref.size):
        radiance = totalRadiance(atm,ref[idx],300.0)
        #we'll recycle the SNR code to extract photoelectron generation rate        
        data = photonDetectorSNR(sensor,atm[:,0],radiance,0.0)
        pe[idx] = data.tgtNrate*intTime*sensor.ntdi
        pe[pe > sensor.maxN] = sensor.maxN
    
    return ref, pe

def noisyBlurImage(img,pixValues,refValues,ref,pe,imggsd,rng, otf, df,ifov,gnoise):
    """simulates radiometrically accurate imagery collected through a sensor 
       from a reference image.
    
    Parameters
    ----------
    img :
        the reference image
    pixValues :
        array of values in img that map to a known reflectance (counts)
    refValues :
        array of reflectances that correspond to pixValues (unitless)
    ref :
        array of reflectance values (unitless) from 0 to 1 in 100 steps
        See reflectance2photoelectrons.
    pe :
        photoelectrons generated during the integration time corresponding to 
        the reflectance values in ref. See reflectance2photoelectrons (electrons)
    imggsd :
        spatial sampling for imgin (m).  Don't confuse the with spatial sampling
        in the imaging system!
    rng :
        range from camera to the test img target (m)
    otf :
        the complex transfer function of the imaging system
    df :
        spatial frequency sampling of the otf (radians^-1)
    ifov : 
        IFOV of the imaging system (radians)
    gnoise :
        standard deviation of additive Gaussian noise (e.g. read noise, quantization)
        Should be the RSS value if multiple terms are combined.  This should not
        include photon noise
    
    Returns
    -------
    trueImg :
        The true image in units of photoelectrons
    blurImg :
        The image after blurring and resampling is applied to trueImg
    noisyImg :
        The blur image with photon (Poisson) noise and gaussian noise applied
    
    WARNING
    -------
    imggsd must be small enough to properly sample the blur kernel! As a guide,
    if the image system transfer function goes to zero at angular spatial frequency, coff,
    then the sampling requirement will be readily met if imggsd <= rng/(4*coff).
    In practice this is easily done by upsampling imgin.
    """
    #convert to image into reflectance space
    refImg = img2reflectance(img,pixValues,refValues)
    
    #convert the reflectance image to photoelectrons
    f = interpolate.interp1d(ref,pe)    
    trueImg = f(refImg)
    
    #blur and resample the image
    blurImg,_ = simulateImage(trueImg,imggsd,rng, otf, df,ifov)
    
    #add photon noise (all sources) and dark current noise
    noisyImg = np.random.poisson(lam=blurImg)
    #add any noise from Gaussian sources, e.g. readnoise, quantizaiton
    noisyImg = np.random.normal(noisyImg,gnoise)
    
    return trueImg, blurImg, noisyImg

def metrics2image(metrics,testimg,imggsd,pixValues,refValues):
    """applies the results of a niirs simulation to a reference image in order to
    simulate how the image would be degraded in this context.
     
    
    Parameters
    ----------
    metrics :
        the object output of (for instance) pybsm.niirs or equivalent
    testimg :
        the reference image (2D array of counts)
    imggsd :
        spatial sampling for imgin (m).  Don't confuse the with spatial sampling
        in the imaging system!
    pixValues :
        array of values in img that map to a known reflectance (counts)
    refValues :
        array of reflectances that correspond to pixValues (unitless)
    
    Returns
    -------
    trueImg :
        The true image in units of photoelectrons
    blurImg :
        The image after blurring and resampling is applied to trueImg
    noisyImg :
        The blur image with photon (Poisson) noise and gaussian noise applied
    
    WARNING
    -------
    imggsd must be small enough to properly sample the blur kernel! As a guide,
    if the image system transfer function goes to zero at angular spatial frequency, coff,
    then the sampling requirement will be readily met if imggsd <= rng/(4*coff).
    In practice this is easily done by upsampling imgin.
    """
    atm = metrics.atm
    intTime = metrics.snr.intTime

    ref,pe = reflectance2photoelectrons(atm,metrics.sensor,intTime)

    rng = metrics.slantRange
    otf = metrics.otf.systemOTF
    df = metrics.df
    ifov = metrics.ifovx

    gnoise = np.sqrt(metrics.snr.quantizationNoise**2.0+metrics.sensor.readNoise**2.0)

    trueImg, blurImg, noisyImg = noisyBlurImage(testimg,pixValues,refValues,ref,pe,imggsd,rng, otf, df,ifov,gnoise)
    
    if noisyImg.shape[0] > testimg.shape[0]:
        print("Warning!  The simulated image has oversampled the reference image!  This result should not be trusted!!")
    
    return trueImg, blurImg, noisyImg

def getGroundRangeArray(maxGroundRange):
    """returns an array of ground ranges that are valid in the precalculated MODTRAN database
    
    Parameters
    ----------
        maxGroundRange :
            largest ground Range of interest (m)
            
    Returns
    -------
        G :
            array of ground ranges less than maxGroundRange (m)
    """    
    G = np.array([0.0, 100.0, 500.0])
    G = np.append(G,np.arange(1000.0,20000.01,1000.0))
    G = np.append(G,np.arange(22000.0,80000.01,2000.0))
    G = np.append(G,np.arange(85000.0,300000.01,5000.0))
    G = G[G <=maxGroundRange]
    return G

def plotCommonMTFs(metrics,orientationAngle=0):
    """Generates a plot of common MTF components: aperture,turbulence,detector,
    jitter,drift,wavefront,image processing,system.  The Nyquist frequency is 
    annotated on the plot with a black arrow.  Spatial frequencies are converted
    to image plane units (cycles/mm).
    
    Parameters
    ----------
    metrics :
        the object output of (for instance) pybsm.niirs or equivalent
    
    orientationAngle:
        angle to slice the MTF (radians).  A 0 radian slice is along the u axis.  
        The angle rotates counterclockwise. Angle pi/2 is along the v axis.  The
        default value is 0 radians.
    Returns
    -------
        A plot.
    """
        
    #spatial frequencies in the image plane in (cycles/mm)
    radfreq = np.sqrt(metrics.uu**2+metrics.vv**2)
    sf = sliceotf(.001*(1.0/metrics.sensor.f)*radfreq,orientationAngle)
    
    #extract MTFs
    apmtf = np.abs(sliceotf(metrics.otf.apOTF,orientationAngle))
    turbmtf = np.abs(sliceotf(metrics.otf.turbOTF,orientationAngle))
    detmtf = np.abs(sliceotf(metrics.otf.detOTF,orientationAngle))
    jitmtf = np.abs(sliceotf(metrics.otf.jitOTF,orientationAngle))
    drimtf = np.abs(sliceotf(metrics.otf.drftOTF,orientationAngle))
    wavmtf = np.abs(sliceotf(metrics.otf.wavOTF,orientationAngle))
    sysmtf = np.abs(sliceotf(metrics.otf.systemOTF,orientationAngle))
    filmtf = np.abs(sliceotf(metrics.otf.filterOTF,orientationAngle))
        
    plt.plot(sf,apmtf,sf,turbmtf,sf,detmtf,sf,jitmtf,sf,drimtf, sf,wavmtf,sf,filmtf,'gray')
    plt.plot(sf,sysmtf,'black',linewidth = 2)
    plt.axis([0, sf.max(), 0, filmtf.max()])
    plt.xlabel('spatial frequency (cycles/mm)')
    plt.ylabel('MTF')    
    plt.legend(['aperture','turbulence','detector','jitter','drift','wavefront','image processing','system'])
    
    #add nyquist frequency to plot
    nyquist = 0.5/(metrics.ifovx*metrics.sensor.f)*.001        
    plt.annotate('', xy=(nyquist, 0), xytext=(nyquist, 0.1), arrowprops=dict(facecolor='black', shrink=0.05)) 
    return 0

def plotNoiseTerms(metrics,maxval = 0,ax = 0):
    """Generates a plot of common noise components in units of equivalent
    photoelectrons.  Components Total, Scene photons, Path photons, Emission / Stray photons, 
    Dark Current, Quantization, Readout.
    
    Parameters
    ----------
    metrics :
        the object output of (for instance) pybsm.niirs or equivalent
    
    maxval: 
        (optional) sets the y-axis limit on photoelectrons.  Useful when comparing across plots.
        Default value of 0 allows matplotlib to automatically select the scale.
    Returns
    -------
        A plot.
    """
    fig, ax = plt.subplots()   
    ms = metrics.snr
    noiseterms = np.array([ms.totalNoise,ms.sceneNoise,ms.pathNoise,ms.selfEmissionNoise,ms.darkcurrentNoise,ms.quantizationNoise,metrics.sensor.readNoise])    
    ind = np.arange(noiseterms.shape[0])    
    ax.bar(ind, noiseterms, color='b')  
    ax.set_ylabel('RMS Photoelectrons')    
    ax.set_xticklabels((' ','Total','Scene', 'Path', 'Emission / Stray', 'Dark Current', 'Quantization','Readout'),rotation = 45)
    #NOTE: I'm not sure why the first label is ingnored in the previous line of code
    #seems to be a new issue when I transitions to Python 3.6
    plt.title('total photoelectrons per pixel: ' + str(int(metrics.snr.tgtN)) +'\n contrast photoelectrons per pixel: ' + str(int(metrics.snr.tgtN-metrics.snr.bkgN)) +'\n well fill: ' + str(int(metrics.snr.wellfraction*100.0)) +'%')
    if maxval > 0 :
        plt.ylim([0, maxval])
    plt.tight_layout() #prevents text from getting cutoff
    return 0
    
#################################################################
#BEGINNING OF VERSION 0.2 FUNCTIONS
#################################################################
def giqe5(rer1,rer2,gsd,snr,elevAngle):
    """NGA The General Image Quality Equation version 5.0. 16 Sep 2015  
    https://gwg.nga.mil/ntb/baseline/docs/GIQE-5_for_Public_Release.pdf
    This version of the GIQE replaces the ealier versions and should be used in all
    future analyses.  See also "Airborne Validation of the General Image Quality Equation 5"
    https://www.osapublishing.org/ao/abstract.cfm?uri=ao-59-32-9978
         
    Parameters
    ---------- 
    rer1, rer2 : 
        relative edge response in two directions (e.g., along- and across- scan,
         horizontal and vertical, etc.) See also pybsm.giqe5RER. (unitless)  
    gsd : 
        image plane geometric mean ground sample distance (m), as defined for GIQE3.  The GIQE 5 version
        of GSD is calculated within this function
    snr :
        contrast signal-to-noise ratio (unitless), as defined for GIQE3
    elevangle:
        sensor elevation angle as measured from the target (rad), i.e., pi/2-nadirAngle.
        See pybsm.nadirAngle for more information
        
    Returns 
    ------- 
    niirs : 
        a National Image Interpretability Rating Scale value (unitless)
    gsdw :
        elevlation angle weighted GSD (m)
    rer : 
        weighted relative edge response (rer)
    NOTES
    -----
    NIIRS 5 test case: rer1=rer2=0.35, gsd = 0.52832 (20.8 inches), snr = 50, elevangle = np.pi/2
    From Figure 1 in the NGA GIQE5 paper.
    
    """
    #note that, within the GIQE, gsd is defined in inches, hence the conversion in the niirs equation below
    gsdw = gsd/(np.sin(elevAngle)**(.25)) #geometric mean of the image plane and ground plane gsds
    
    
    rer = ( np.max([rer1,rer2])*np.min([rer1,rer2])**2.0 )**(1.0/3.0)
    
    niirs = 9.57-3.32*np.log10(gsdw/.0254)+3.32*(1-np.exp(-1.9/snr))*np.log10(rer)-2.0*np.log10(rer)**4.0 - 1.8/snr
    return niirs, gsdw, rer

def giqe5RER(mtf,df,ifovx,ifovy): 
    """Calculates the relative edge response from a 2-D MTF.  This function is 
    primarily for use with the GIQE 5. It
    implements IBSM equations 3-57 and 3-58.  See pybsm.giqeEdgeTerms for the 
    GIQE 3 version.
     
    Parameters 
    ---------- 
    mtf: 
        2-dimensional full system modulation transfer function (unitless).  MTF is 
        the magnitude of the OTF. 
    df :  
        spatial frequency step size (cycles/radian) 
    ifovx:  
        x-direction instantaneous field-of-view of a detector (radians)
    ifovy:  
        y-direction instantaneous field-of-view of a detector (radians)
     
    Returns 
    ------- 
    rer0: 
        relative edge response at 0 degrees orientation (unitless)
    rer90: 
        relative edge response at 90 degrees orientation (unitless)
    """ 
    uslice = sliceotf(mtf,0) 
    vslice = sliceotf(mtf,np.pi/2) 
     
    rer0 = relativeEdgeResponse(uslice,df,ifovx) 
    rer90 = relativeEdgeResponse(vslice,df,ifovy) 
          
    return rer0, rer90


def niirs5(sensor,scenario):
    """returns NIIRS values calculate using GIQE 5 and all intermetiate 
    calculations.  See pybsm.niirs for the GIQE 3 version.  This version of the 
    GIQE replaces the ealier versions and should be used in all
    future analyses

    Parameters
    ---------- 
    sensor:
        an object from the class sensor
    scenario:
        an object from the class scenario
     
    Returns 
    -------- 
    nm: 
        an object containing results of the GIQE calculation along with many
        intermediate calculations
    """
    #initialize the output    
    nm = metrics('niirs ' + sensor.name + ' ' + scenario.name)
    nm.sensor = sensor
    nm.scenario = scenario    
    nm.slantRange = curvedEarthSlantRange(0.0,scenario.altitude,scenario.groundRange)
    
    ##########CONTRAST SNR CALCULATION#########    
    #load the atmosphere model
    nm.atm = loadDatabaseAtmosphere(scenario.altitude,scenario.groundRange,scenario.ihaze)
    
    #crop out out-of-band data (saves time integrating later)    
    nm.atm = nm.atm[nm.atm[:,0] >= nm.sensor.optTransWavelengths[0],:]
    nm.atm = nm.atm[nm.atm[:,0] <= nm.sensor.optTransWavelengths[-1],:]

    if (nm.sensor.optTransWavelengths[0] >= 2.9e-6):
        isEmissive = 1 #toggle the GIQE to assume infrared imaging
        #the next four lines are bookkeeping for interpreting the results since
        #the giqeRadiance function assumes these values anyway         
        nm.scenario.targetReflectance = 0.0
        nm.scenario.backgroundReflectance = 0.0
        nm.scenario.targetTemperature = 282.0
        nm.scenario.targetTemperature = 280.0
        
    else:
        isEmissive = 0
        #more bookkeeping (see previous comment)
        nm.scenario.targetReflectance = 0.15
        nm.scenario.backgroundReflectance = 0.07
    
    #get at aperture radiances for reflective target and background with GIQE type target parameters
    nm.tgtRadiance,nm.bkgRadiance = giqeRadiance(nm.atm,isEmissive)
    nm.radianceWavelengths = nm.atm[:,0]
    
    #now calculate now characteristics ****for a single frame******    
    nm.snr = photonDetectorSNR(sensor, nm.radianceWavelengths,nm.tgtRadiance,nm.bkgRadiance)
    
    #break out photon noise sources (not required for NIIRS but useful for analysis)
    #photon noise due to the scene itself (target,background, and path emissions/scattering)    
    tgtNoise= np.sqrt(np.trapz(photonDetectionRate(nm.snr.tgtFPAirradiance-nm.snr.otherIrradiance,nm.sensor.wx, \
    nm.sensor.wy,nm.radianceWavelengths,nm.snr.qe),nm.radianceWavelengths)*nm.snr.intTime*sensor.ntdi)     
    bkgNoise= np.sqrt(np.trapz(photonDetectionRate(nm.snr.bkgFPAirradiance-nm.snr.otherIrradiance,nm.sensor.wx, \
    nm.sensor.wy,nm.radianceWavelengths,nm.snr.qe),nm.radianceWavelengths)*nm.snr.intTime*sensor.ntdi)    #assign the scene Noise to the larger of the target or background noise    
    sceneAndPathNoise = np.max([tgtNoise,bkgNoise])
    #calculate noise due to just the path scattered or emitted radiation
    scatrate,_,_ = signalRate(nm.radianceWavelengths,nm.atm[:,2]+nm.atm[:,4],nm.snr.optTrans, \
    nm.sensor.D,nm.sensor.f,nm.sensor.wx,nm.sensor.wy,nm.snr.qe,0.0,0.0)
    nm.snr.pathNoise = np.sqrt(scatrate*nm.snr.intTime*sensor.ntdi)
    nm.snr.sceneNoise = np.sqrt(sceneAndPathNoise**2-nm.snr.pathNoise**2)
    #####OTF CALCULATION#######
    
    #cut down the wavelength range to only the regions of interest
    nm.mtfwavelengths =  nm.radianceWavelengths[nm.snr.weights > 0.0]
    nm.mtfweights =  nm.snr.weights[nm.snr.weights > 0.0] 
    
    #setup spatial frequency array    
    nm.cutoffFrequency = sensor.D/np.min(nm.mtfwavelengths)
    urng = np.linspace(-1.0, 1.0, 101)*nm.cutoffFrequency
    vrng = np.linspace(1.0, -1.0, 101)*nm.cutoffFrequency
    nm.uu,nm.vv = np.meshgrid(urng, vrng) #meshgrid of spatial frequencies out to the optics cutoff
    nm.df = urng[1]-urng[0] #spatial frequency step size   
    
    sensor.filterKernel = np.array([1]) #ensures that shapening is turned off.  Not valid for GIQE5
    nm.otf = commonOTFs(sensor,scenario,nm.uu,nm.vv,nm.mtfwavelengths,nm.mtfweights,nm.slantRange,nm.snr.intTime)

    ##CALCULATE NIIRS##############
    nm.ifovx = sensor.px / sensor.f
    nm.ifovy = sensor.py / sensor.f
    nm.gsdx = nm.ifovx*nm.slantRange #GIQE5 assumes all square detectors
    nm.rer0,nm.rer90 = giqe5RER(np.abs(nm.otf.systemOTF),nm.df,nm.ifovx,nm.ifovy)

    #note that NIIRS is calculated using the SNR ***after frame stacking**** if any 
    nm.elevAngle = np.pi/2-nadirAngle(0.0,scenario.altitude,nm.slantRange)
    nm.niirs, nm.gsdw, nm.rer = giqe5(nm.rer0,nm.rer90,nm.gsdx,np.sqrt(sensor.framestacks)*nm.snr.snr,nm.elevAngle)

    return nm



def loadDatabaseAtmosphere(altitude,groundRange,ihaze):
    """linear interpolation of the pre-calculated MODTRAN atmospheres.
    See the original 'loadDatabaseAtmosphere' (now commented out) for more details on the outputs.
    NOTE: This is experimental code.  Linear interpolation between atmospheres
    may not be a good approximation in every case!!!!
    
    Parameters
    ---------- 
    altiude:
        sensor height above ground level in meters
    groundRange:

    ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural extinction with 23 km 
        visibility) or ihaze = 2 (Rural extinction with 5 km visibility)
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
    NOTE: units for columns 1 through 5 are in radiance W/(sr m^2 m)"""   
    
    def altAtmInterp(lowalt,highalt,altitude,groundRange,ihaze):
        #this is an internal function for interpolating atmospheres across altitudes
        lowatm = loadDatabaseAtmosphere_nointerp(lowalt,groundRange,ihaze)
        if lowalt != highalt:
            highatm = loadDatabaseAtmosphere_nointerp(highalt,groundRange,ihaze)
            lowweight = 1 - ((altitude-lowalt) / (highalt-lowalt))
            highweight = ((altitude-lowalt) / (highalt-lowalt))
            atm = lowweight*lowatm + highweight*highatm
        else:
            atm = lowatm
        return atm 

    #define arrays of all possible altitude and ground ranges 
    altarray = np.array([2, 32.55, 75, 150, 225, 500, 1000, 2000, 3000, 4000, \
                         5000, 6000, 7000, 8000, 9000, 10000, 11000,12000,14000,16000,18000,20000]) 
    grangearray = getGroundRangeArray(301e3)
    
    #find the database altitudes and ground ranges that bound the values of interest
    lowalt = altarray[altarray <= altitude][-1]
    highalt = altarray[altarray >= altitude][0]
    lowrng = grangearray[grangearray <= groundRange][-1]
    highrng = grangearray[grangearray >= groundRange][0]
    
    
    #first interpolate across the low and high altitudes
    #then interpolate across ground range
    atm_lowrng = altAtmInterp(lowalt,highalt,altitude,lowrng,ihaze)
    if lowrng != highrng: 
        atm_highrng = altAtmInterp(lowalt,highalt,altitude,highrng,ihaze)
        lowweight = 1 - ((groundRange-lowrng) / (highrng-lowrng))
        highweight = ((groundRange-lowrng) / (highrng-lowrng))
        atm = lowweight*atm_lowrng + highweight*atm_highrng
    else:
        atm = atm_lowrng

    return atm

def xandyUserOTF2(u,v,fname):
    """UPDATE to IBSM Equation 3-32.  Import user-defined x-direction and
    y-direction OTFs and interpolate them onto a 2-dimensional spatial frequency grid.  
    Per ISBM Table 3-3c, the OTF data are ASCII text, space delimited data.  (Note: There
    appears to be a typo in the IBSM documentation - Table 3-3c should represent
    the "x and y" case, not "x or y".).  In the original version, the 2D OTF
    is given as Hx*Hy, the result being that the off-axis OTF is lower than either Hx or Hy.
    The output is now given by the geometric mean.
    
    Parameters
    ----------
    u or v : 
        angular spatial frequency coordinates (rad^-1)
    fname : 
        filename and path to the x and y OTF data.       
        
    Returns
    -------
    H : 
        OTF at spatial frequency (u,v) (unitless)
    
    """
    xandyData = np.genfromtxt(fname)
    
    Hx = np.interp(np.abs(u),xandyData[:,0],xandyData[:,1]) + \
    np.interp(np.abs(u),xandyData[:,0],xandyData[:,2])*1.0j
    
    Hy = np.interp(np.abs(v),xandyData[:,3],xandyData[:,4]) + \
    np.interp(np.abs(v),xandyData[:,3],xandyData[:,5])*1.0j

    H = np.sqrt(Hx*Hy)
    return H

def giqe4(rer,gsd,eho,ng,snr,elevAngle):
    """General Image Quality Equation version 4 from Leachtenauer, et al.,
    "General Image Quality Equation: GIQE," Applied Optics, Vol 36, No 32, 1997.
    I don't endorse the use of GIQE 4 but it is added to pyBSM for historical
    completeness.
     
    Parameters
    ---------- 
    rer : 
        geometric mean relative edge response (unitless)  
    gsd : 
        geometric mean ground sample distance (m) before projection into the ground plane
    eho : 
        geometric mean edge height overshoot (unitless)
    ng :
        noises gain, i.e. increase in noise due to sharpening (unitless). If no sharpening is 
        applied then ng = 1
    snr :
        contrast signal-to-noise ratio (unitless)
    elevangle:
        sensor elevation angle as measured from the target (rad), i.e., pi/2-nadirAngle.
        See pybsm.nadirAngle for more information.  Note that the GIQE4 paper defines
        this angle differently but we stick with this version to be consisent with the GIQE 5 code.
        The outcome is the same either way.
        
    Returns 
    ------- 
    niirs : 
        a National Image Interpretability Rating Scale value (unitless)  
    """
    if rer >= 0.9:
        c1 = 3.32
        c2 = 1.559
    else:
        c1 = 3.16
        c2 = 2.817
    
    gsdgp = gsd/(np.sin(elevAngle)**(0.5))#note that the expondent captures the
    #fact that only one direction in the gsd is distorted by projection into the
    #ground plane
    
    niirs = 10.251 - c1*np.log10(gsdgp/.0254) + c2*np.log10(rer) -.656*eho - 0.334*ng/snr
    #note that, within the GIQE, gsd is defined in inches, hence the conversion
    return niirs, gsdgp

#################################################################
#start defining classes for cameras, scenarios, etc.
################################################################

class sensor(object):
    """Example details of the camera system.  This is not intended to be a
    complete list but is more than adequate for the NIIRS demo (see pybsm.niirs).
    
    Attributes (the first four are mandatory):
    ------------------------------------------
    name :
        name of the sensor (string)
    D : 
        effective aperture diameter (m)
    f :
        focal length (m)
    px and py : 
        detector center-to-center spacings (pitch) in the x and y directions (m)
    optTransWavelengths :
        numpy array specifying the spectral bandpass of the camera (m).  At
        minimum, and start and end wavelength should be specified.
                
    opticsTransmission :
        full system in-band optical transmission (unitless).  Loss due to any
        telescope obscuration should *not* be included in with this optical transmission
        array.
    eta : 
        relative linear obscuration (unitless)        
    wx and wy : 
        detector width in the x and y directions (m)     
    qe :
        quantum efficiency as a function of wavelength (e-/photon)
    qewavelengths :
        wavelengths corresponding to the array qe (m)
    otherIrradiance :
        spectral irradiance from other sources (W/m^2 m).
        This is particularly useful for self emission in infrared cameras.  It may
        also represent stray light.
    darkCurrent :
        detector dark current (e-/s)
    maxN : 
        detector electron well capacity (e-)
    maxFill :
        desired well fill, i.e. Maximum well size x Desired fill fraction
    bitdepth :
        resolution of the detector ADC in bits (unitless)
    ntdi :
        number of TDI stages (unitless)
    coldshieldTemperature :
        temperature of the cold shield (K).  It is a common approximation to assume
        that the coldshield is at the same temperature as the detector array.
    opticsTemperature :
        temperature of the optics (K)
    opticsEmissivity :
        emissivity of the optics (unitless) except for the cold filter.  
        A common approximation is 1-optics transmissivity. 
    coldfilterTransmission :
        transmission through the cold filter (unitless)
    coldfilterTemperature :
        temperature of the cold filter.  It is a common approximation to assume
        that the filter is at the same temperature as the detector array.  
    coldfilterEmissivity :
        emissivity through the cold filter (unitless).  A common approximation 
        is 1-cold filter transmission
    sx and sy :
        Root-mean-squared jitter amplitudes in the x and y directions respectively. (rad)
    dax and day :
        line-of-sight angular drift rate during one integration time in the x and y 
        directions respectively. (rad/s)
    pv : 
        wavefront error phase variance (rad^2) - tip: write as (2*pi*waves of error)^2
    pvwavelength :
        wavelength at which pv is obtained (m)
    Lx and Ly :
        correlation lengths of the phase autocorrelation function.  Apparently,
        it is common to set the Lx and Ly to the aperture diameter.  (m) 
    otherNoise :
        a catch all for noise terms that are not explicitly included elsewhere 
        (read noise, photon noise, dark current, quantization noise are
        all already included)
    filterKernel: 
         2-D filter kernel (for sharpening or whatever).  Note that 
         the kernel is assumed to sum to one.
    framestacks:
         the number of frames to be added together for improved SNR.
    
    """
    def __init__(self,name,D,f,px,optTransWavelengths):
        """Returns a sensor object whose name is *name* and...."""
        self.name = name
        self.D = D
        self.f = f
        self.px = px;
        self.optTransWavelengths = optTransWavelengths
        self.opticsTransmission = np.ones(optTransWavelengths.shape[0])
        self.eta = 0.0
        self.py = px
        self.wx = px
        self.wy = px #initial assumption is 100% fill factor and square detectors
        self.intTime = 1.0
        self.darkCurrent =  0.0
        self.otherIrradiance = 0.0
        self.readNoise = 0.0
        self.maxN = 100.0e6 #initializes to a large number so that, in the absence
        #of better information, it doesn't affect outcomes
        self.maxWellFill = 1.0        
        self.bitdepth=100.0 #initializes to a large number so that, in the absense
        #of better information, it doesn't affect outcomes
        self.ntdi = 1.0
        self.qewavelengths = optTransWavelengths #tplaceholder
        self.qe = np.ones(optTransWavelengths.shape[0]) #placeholder
        self.coldshieldTemperature = 70.0
        self.opticsTemperature = 270.0
        self.opticsEmissivity = 0.0
        self.coldfilterTransmission = 1.0
        self.coldfilterTemperature = 70.0
        self.coldfilterEmissivity = 0.0
        self.sx = 0.0
        self.sy = 0.0
        self.dax = 0.0
        self.day = 0.0
        self.pv = 0.0
        self.pvwavelength = .633e-6 #typical value
        self.Lx = D
        self.Ly = D
        self.otherNoise = np.array([0])
        self.filterKernel = np.array([1])
        self.framestacks = 1
        
class scenario(object):
    """Everything about the target and environment.  NOTE:  if the niirs model
    is called, values for target/background temperature, reflectance, etc. are
    overridden with the NIIRS model defaults.
    ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural extinction with 23 km visibility)
        or ihaze = 2 (Rural extinction with 5 km visibility)    

    altiude:
        sensor height above ground level in meters.  The database includes the following
        altitude options: 2 32.55 75 150 225 500 meters, 1000 to 12000 in 1000 meter steps, 
        and 14000 to 20000 in 2000 meter steps, 24500 
    groundRange:
        distance *on the ground* between the target and sensor in meters.
        The following ground ranges are included in the database at each altitude 
        until the ground range exceeds the distance to the spherical earth horizon: 
        0 100 500 1000 to 20000 in 1000 meter steps, 22000 to 80000 in 2000 m steps, 
        and  85000 to 300000 in 5000 meter steps.
    aircraftSpeed:
        ground speed of the aircraft (m/s)
    targetReflectance:
        object reflectance (unitless)
    targetTemperature:
        object temperature (Kelvin)
    backgroundReflectance:
        background reflectance (unitless)
    backgroundTemperature:
        background temperature (Kelvin)
    haWindspeed: 
        the high altitude windspeed (m/s).  Used to calculate the turbulence profile.
    cn2at1m:
        the refractive index structure parameter "near the ground" (e.g. at h = 1 m).
        Used to calculate the turbulence profile.
    
    """
    
    def __init__(self,name,ihaze,altitude,groundRange):
        self.name = name
        self.ihaze = ihaze
        self.altitude = altitude
        self.groundRange = groundRange
        self.aircraftSpeed = 0.0
        self.targetReflectance = 0.15 #the giqe standard
        self.targetTemperature = 295.0 #282 K is used for GIQE calculation (this value is ignored)
        self.backgroundReflectance = 0.07
        self.backgroundTemperature = 293.0 #280 K used for GIQE calculation (this value is ignored)
        self.haWindspeed = 21.0 #HV 5/7 profile value
        self.cn2at1m = 1.7e-14  #HV 5/7 profile value      
        
class metrics(object):
    """A generic class to fill with any outputs of interest."""
    def __init__(self,name):
        """Returns a sensor object whose name is *name* """
        self.name = name

    

    
