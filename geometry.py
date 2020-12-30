################
### Geometry ###
################

'''
Useful geometry functions for astrophysics and astronomy
'''

import numpy as np

def LMCdisk_cart(ra, dec):
    
    '''
    Calculate the position of stars in the LMC disk plane with 
    center at the LMC center in cartesian coordinates (x, y).
    This also calculates the distance to the individual stars.
    
    This follows van der Marel and Cioni 2001 
    
    Input
    - ra: right ascension of stars
    - dec: declination of stars
    
    Output
    - x_m: x coordinate
    - y_m: y coordinate
    - dis: distance to LMC star
    '''
    alph0 = np.radians(82.25) #right ascension of center of LMC
    delt0 = np.radians(-69.50) #declination of center of LMC
    pa = np.radians(149.23+90.00) #146.37 #position angle of line of nodes
    io = np.radians(25.86) #27.81 #inclination of LMC disk
    d0 = 49.90 #distance to center of LMC
    
    #convert to radians
    ra = np.radians(ra)
    dec = np.radians(dec)
    sd = np.sin(delt0)
    cd = np.cos(delt0)
    
    cr = cd*np.cos(dec)*np.cos(ra-alph0)+sd*np.sin(dec)
    srcp = -np.cos(dec)*np.sin(ra-alph0)
    srsp = cd*np.sin(dec) - sd*np.cos(dec)*np.cos(ra-alph0)
    dis = d0*np.cos(io)/(np.cos(io)*cr - np.sin(io)*np.cos(pa)*srsp + np.sin(io)*np.sin(pa)*srcp)
    
    x_m = dis*srcp
    y_m = dis*(np.cos(io)*srsp + np.sin(io)*cr) - d0*np.sin(io)
    
    return x_m, y_m, dis

####################################################################################

def radec2diskcart(ra, dec, ra0 = 0, dec0 = 0, d0 = 0, i0 = 0, pa = 0, galaxy = None):
    '''
    Calculate the position of stars in the disk plane with 
    center at the  center in Cartesian coordinates (x, y).
    This also calculates the distance to the individual stars.
    
    This follows van der Marel and Cioni (2001)
    
    Input:
    -----
    - ra: [deg] right ascension of stars
    - dec: [deg] declination of stars
    - ra0: [deg] center point right ascension
    - dec0: [deg] center point declination
    - d0: [length] distance to (ra0, dec0)
    - i0: [deg] inclinationation of the disk
    - pa: [deg] posistion angle of the line of nodes for the disk
    
    Output:
    ------
    - x_disk: [d0 units] x coordinates of the stars
    - y_disk: [d0 units] y coordinates of the stars
    - dist_disk: [d0 units] distance to individual stars
    '''
    
    if galaxy == 'lmc':
        # all parameters from Choi et al. (2018)
        ra0 = np.radians(82.25) # right ascension of center of LMC
        dec0 = np.radians(-69.50) # declination of center of LMC
        i0 = np.radians(25.86) # inclination of LMC disk
        d0 = 49.90 # distance [kpc] to center of LMC
        pa = np.radians(149.23+90.00) # position angle of line of nodes from North
        
    #convert to radians
    ra = np.radians(ra)
    dec = np.radians(dec)
    sd = np.sin(dec0)
    cd = np.cos(dec0)
    
    cr = cd*np.cos(dec)*np.cos(ra-ra0)+sd*np.sin(dec)
    srcp = -np.cos(dec)*np.sin(ra-ra0)
    srsp = cd*np.sin(dec) - sd*np.cos(dec)*np.cos(ra-ra0)
    dis = d0*np.cos(i0)/(np.cos(i0)*cr - np.sin(i0)*np.cos(pa)*srsp + np.sin(i0)*np.sin(pa)*srcp)
    
    x_disk = dis*srcp
    y_disk = dis*(np.cos(i0)*srsp + np.sin(i0)*cr) - d0*np.sin(i0)
        
    return x_disk, y_disk, dist_disk

####################################################################################

def elliptical_radius(x, y, b_a = 0, psi = 0, galaxy = None):
    '''
    Calculate elliptical radius from Cartesian (x,y)
    
    This comes from Choi et al (2018)
    
    Input:
    -----
        x: [length] x cartesian coordinate
        y: [length] y cartesian coordinate
        psi: [deg] semimajor axis position angle
        
    Output:
    ------
        ell_r: [x units] elliptical radius
    '''
    
    if galaxy == 'lmc':
        # from Choi et al. 2018
        b_a = 0.836 # disk axis ratio
        psi = 227.24 + 90.0 # position angle [deg] of semi major axis from North
        
    cpsi = np.cos(np.radians(psi)) 
    spsi = np.cos(np.radians(psi))
    
    ell_r = np.sqrt(np.square(x*cpsi-y*spsi) + np.square(b_a*(x*spsi + y*cpsi)))
    
    return ell_r

####################################################################################
