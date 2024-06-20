Atmosphere Database (Sunny Day) calculated using MODTRAN 5.2.1

Each filename in the database is a number corresponding to the contitions described 
in fileDecoder.csv.  The columns in fileDecoder are given by
    Index (filename), Altitude (km), Ground Range (km), IHAZE (1 or 2)


The database contains two IHAZE options:

    1 - rural extinction, 23 km visibility
    2 - rural extinction, 5 km visibility

Database altitudes include (km):
    .002 .0325 .075 .15 .225 .5, 1 to 12 in 1 km steps, and 14 to 20 in 2 km steps, 24.5 km 

The following ground ranges are included in the database at each altitude until the ground range exceeds the distance to the spherical earth horizon: 
    0 .1 .5 1 to 20 in 1 km steps, 22 to 80 in 2 km steps, and  85 to 300 in 5 km steps

----UPDATE---July 2018----
A nadir look from space option has been added for both IHAZE = 1 and 2
altitude = 500 km, ground range = 0
For the nadir case, aperture radiance will be effectively the same for any orbital altitude.
-------------------------

Each file in the database contains the following columns from the MODTRAN *.7sc file (in this order):

TRANS: total transmission through the defined path.
PTH THRML: radiance component due to atmospheric emission and scattering received at the observer.
SURF EMIS: component of radiance due to surface emission received at the observer.
SOL SCAT: component of scattered solar radiance received at the observer.
GRND RFLT: is the total solar flux impingent on the ground and reflected directly to the sensor from the ground. (direct radiance + diffuse radiance) * surface reflectance
and each of these columns is calculated between the wavelength range of 0.3 to 14 um in .01 um steps (including the end points, e.g. .3, .31, ... 14) 

Therefore, each data base file is 1371 rows by 5 columns.
Note that the total radiance received at the aperture is the sum of these 4 values, 
e.g. TOTAL RAD = PTH THRML + SURF EMIS + SOL SCAT + GRND RFLT
This and the previous column definitions come from the Ontar website.
The units for all radiance columns are: W/(sr cm^2 um) 


For all model runs, the following MODTRAN parameters are fixed (only the most salient shown):

    GNDALT=0.0;        % Ground altitude (km MSL)
    H2=GNDALT;          % target altitude (km MSL)
    PARM1=90;            % azimuth between sensor line of sight and sun
    PARM2=50;           % solar zenith (0 is looking straight up so, in other words the sun is 40 degrees above the horizon)
    
    
    % Define wavelenth range and resolution of simulation
    DV = .01; %spectral sampling (um)
    FWHM=2*DV; % FWHM should be at least 2x sampling
    
    % Atm paramters we typically vary
    VIS=0;             % Visibility = 0 means that visibility is inhereted from IHAZE 
    MODEL=2;            % Atmospheric profile: Mid-Latitude Summer (45 degrees North Latitude)
    H2OSTR=1;           % water vapor scaling constant
    ICLD=0;             % No clouds
    IDAY=236;           % Day of year (August 23rd)
    
    IMULT=1;            % Controls multiple scattering (1=yes, 0=no)
    RAINRT=0;           % No rain
    ANGLEM=0;           % moon phase angle (doesn't matter)
    ISOURC=0;           % Extraterrestrial source is the sun
    
    % Target and background temp/reflectance params
    TPTEMP=299;                 % target temp (K)
    AATEMP=297;                 % background temp (k)
    SURREFt='constant, 15%';     % target reflectance
    SURREFb='Mixed Forest';     % background reflectance

For reference, the .tp5 input file for the 20 km altitude ,300 km ground range (zenith angle 95.155),  IHAZE = 2 case is provided here:
-------------------------------------------------------------------
M   2    2    2    1    0    0    0    0    0    0    0    1    1 299.000 LAMBER
t   8t   0   360.000   1.00000   0.00000 f f f
    2    0    0    0    0    0     0.000     0.000     0.000     0.000   0.00000
    20.000     0.000    95.155     0.000     0.000     0.000    0        0.00000
    2    2  236    0
    90.000    50.000     0.000     0.000     0.000     0.000     0.000     0.000
  0.300000 14.000000  0.010000  0.020000tm        M1AA   
2 297.000
DATA/spec_alb.dat
constant, 15%
Mixed Forest
    0
---------------------------------------------------------------------

Also, the 'Mixed Forest' background spectra is:
wavelength (um)  albedo (unitless)
0.2	0.025
0.5	0.036
0.7	0.032
0.725	0.097
0.75	0.254
0.775	0.314
0.9	0.31
0.925	0.419
1	0.4
1.25	0.384
1.3	0.337
1.35	0.287
1.4	0.236
1.45	0.191
1.5	0.162
1.55	0.19
1.6	0.216
1.65	0.24
1.7	0.265
1.75	0.321
1.8	0.238
1.85	0.185
1.9	0.135
1.95	0.098
2	0.058
2.5	0.05
3	0.057
3.5	0.062
4	0.054
4.5	0.048
5	0.042
5.5	0.036
6	0.034
6.5	0.033
7	0.031
7.5	0.03
8	0.03
8.5	0.03
9	0.03
9.5	0.03
10	0.03
11	0.03
12	0.038
13	0.03
14	0.016
15	0.016
	
	

