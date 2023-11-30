
"""
This script is an introductory example of how to use pyBSM.  Compare the results 
to Figures 2, 4(a) and 5(a) in: 
LeMaster, Daniel A.; Eismann, Michael T., "pyBSM: A Python package for modeling 
imaging systems", Proc. SPIE 10204 (2017).

This script also includes an example of using pyBSM to simulate imagery.  Be sure
that the test image is in the same folder as this script.
"""
import numpy as np
import matplotlib.pyplot as plt
import pybsm
import copy
import os

dirpath = os.path.dirname(os.path.abspath(__file__))
imgfile = os.path.join(dirpath, 'data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff')

#telescope focal length (m)
f=4                     
# Telescope diameter (m)
D=275e-3               

#detector pitch (m)
p=.008e-3    
       
#Optical system transmission, red  band first (m)
optTransWavelengths = np.array([0.58-.08,0.58+.08])*1.0e-6

#generate the camera object.  Be sure to check out the details of this object.
#There are many camera pararameters available, all of which are initally set
#to default values.  The salient parameters are populated below.
L3 = pybsm.sensor('L32511x',D,f,p,optTransWavelengths)

#guess at the full system optical transmission (excluding obscuration)
L3.opticsTransmission=0.5*np.ones(optTransWavelengths.shape[0]) 

# Relative linear telescope obscuration
L3.eta=0.4 #guess     

#detector width is assumed to be equal to the pitch
L3.wx=p                    
L3.wy=p 
#integration time (s) - this is a maximum, the actual integration time will be
#determined by the well fill percentage
L3.intTime=30.0e-3 

#dark current density of 1 nA/cm2 guess, guess mid range for a silicon camera
L3.darkCurrent = pybsm.darkCurrentFromDensity(1e-5,L3.wx,L3.wy)

#rms read noise (rms electrons)
L3.readNoise=25.0 

#maximum ADC level (electrons)
L3.maxN=96000.0

#bit depth
L3.bitdepth=11.9

#maximum allowable well fill (see the paper for the logic behind this)
L3.maxWellFill = .6

#jitter (radians) - The Olson paper says that its "good" so we'll guess 1/4 ifov rms
L3.sx = 0.25*p/f
L3.sy = L3.sx

#drift (radians/s) - again, we'll guess that it's really good
L3.dax = 100e-6
L3.day = L3.dax

#etector quantum efficiency as a function of wavelength (microns) 
#for a generic high quality back-illuminated silicon array
# https://www.photometrics.com/resources/learningzone/quantumefficiency.php
L3.qewavelengths=np.array([.3, .4, .5, .6, .7, .8, .9, 1.0, 1.1])*1.0e-6
L3.qe=np.array([0.05, 0.6, 0.75, 0.85, .85, .75, .5, .2, 0])


#sensor altitude
altitude=9000.0
#range to target
groundRange = 0.0


#weather model
ihaze = 1
scenario = pybsm.scenario('niceday',ihaze,altitude,groundRange)
scenario.aircraftSpeed = 100.0 #ground speed in m/s


def niirsVsRange(sensor,scenario,groundRange):
    # this function allows us to iterate through each ground range
    myobj=[]    
    for gr in groundRange:
        scenario.groundRange = gr
        metrics = pybsm.niirs(L3,scenario)
        myobj.append(copy.deepcopy(metrics))
    return myobj


####Red band######
#run the NIIRS model for the high visibility case
groundRange = np.arange(0,101e3,10e3)
x23r = niirsVsRange(L3,scenario,groundRange)

#run the NIIRS model for the low visibility case
scenario.ihaze = 2
x5r = niirsVsRange(L3,scenario,groundRange)    

#####NIR band######
#run the NIIRS model for the high visibility case
L3.optTransWavelengths = np.array([0.85-.15,0.85+.15])*1.0e-6
L3.opticsTransmission=0.5*np.ones(optTransWavelengths.shape[0]) #guess
scenario.ihaze = 1
x23n = niirsVsRange(L3,scenario,groundRange)

#run the NIIRS model for the low visibility case
scenario.ihaze = 2    
x5n=niirsVsRange(L3,scenario,groundRange)


#%% #######plot the niirs versus range results#################
groundRange = groundRange/1000
plt.plot(groundRange,[ii.niirs for ii in x23r],groundRange,[ii.niirs for ii in x5r],'b--', \
groundRange,[ii.niirs for ii in x23n],'r',groundRange,[ii.niirs for ii in x5n],'r--')

plt.figure(1)
plt.axis([0, groundRange.max(), 0, 10])
plt.xlabel('ground range (km)')
plt.ylabel('NIIRS')    
plt.legend(['red 23 km','red 5 km','near IR 23 km','near IR 5 km'])
plt.savefig(os.path.join(dirpath,'fig_2.png'))

#%% #######plot noise terms#################
#reproduces figure 4(a) from the paper

pybsm.plotNoiseTerms(x5r[2])
plt.savefig(os.path.join(dirpath,'fig_4a.png'))

#%% #######plot MTF terms####################
#reproduces figure 5(a) from the paper
plt.figure()
pybsm.plotCommonMTFs(x23r[0])
plt.savefig(os.path.join(dirpath,'fig_5a.png'))

#%% #######generate and display synthetic images#################
#select a test image
testimg = plt.imread(imgfile, format='tif')

#pick a mapping between pixel values and reflectance in the scene
#this is a guess.  Try other reflectance ranges to see how this affects noise
pixValues = np.array([testimg.min(),testimg.max()])
refValues = np.array([.05,.5])
#specify the native ground sample distance in the plane of the image
imggsd = 3.19/160.0 #the width of the tank is 319 cm and it spans ~160 pixels in the image

#display sequence of degraded images as a function of ground range
plt.subplots(nrows=1, ncols=4)
idx=1 #index for the subplots
for ii in np.arange(4,groundRange.shape[0],2): #index starts at 4 because the 
#shorter range images would have a GSD less than imggsd
    plt.subplot(1,4,idx)
    idx=idx+1
    plt.title(str(groundRange[ii]) + ' km')
    _, _, noisyblurryImg = pybsm.metrics2image(x23n[ii],testimg,imggsd,pixValues,refValues)
    plt.imshow(noisyblurryImg,cmap='gray')
plt.savefig(os.path.join(dirpath,'fig_showcase.png'))