#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:10:45 2024
2024-08-18:
-Used to generate distortion profile of SUIT for 2k and 4k images and to 
rectify the distortion.
-This finds the distortion profile from image center to right bottom image
corner and rotates this distortion profile about image center to get the
radial profile of distortion.
-A better method would be using 2D extrapolation and fitting to make the
distortion profile.
-As of now, this works well with SUIT images at VELC aligned position.

2024-08-30: The code is modified to generate 2k and 4k distortion profiles
from SUIT distortion map.

@author: janmejoyarch
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from datetime import date

def make_linear_grad(distortion_profile, imsize):
    '''
    Gradient of distortion along the diagonal of the image.
    To extrapolate the distortion pattern from the zemax simulated data.
    For pixels ranging from frame center to frame corner. Zemax simulates
    for diagonal field angles of 0.39 degrees with a sparse sampling of 101
    points on X and Y. 0.275 deg on either size.
    This model is extrapolated to the full frame size.
    
    The radial pattern of distortion along X and Y axes from frame center to
    bottom right corner (VELC aligned pos) extracted as a 1D array.
    
    Parameters
    ----------
    distortion_profile : 2D array of distortion profile from ZEMAX data.
    To be input separately for X and Y axes.

    Returns
    -------
    1D array- high resolution extrapolated data.

    '''
    #cubic fit along the diagonal of the distortion profile for an axis
    a,b,c,d= np.polyfit(np.arange(101), np.diagonal(distortion_profile), deg=3)
    
    diag_len= round(imsize)
    #The diagonal extrapolation has to be done for diagonal pixels beyond the
    #0.39 deg radius. So the range is extended to 1.414*101= 144 points
    #from -22 to 122 on either side of the diagonal.
    x_new= np.linspace(-22, 122, diag_len)
    #new distortion values from 3rd order fitted polynomial.
    y_new= a*x_new**3+b*x_new**2+c*x_new+d
    #half diagonal of the distortion profile. From center to frame 
    #bottom right corner
    half_y_new= y_new[round(diag_len/2):]
    return(half_y_new)

def make_radial_grad(gradient_1d, imsize):
    '''
    To make a radial gradient by rotating a 1D data about image center.
    Parameters
    ----------
    gradient_1d : 1D numpy array
        A gradient that has to be rotated about frame center to make a radial
        gradient.
    Returns
    -------
    2D numpy array of the radial gradient
    '''
    # Define the size of the 2D array and the center of the radial gradient
    size = imsize  # Size of the 2D array (size x size)
    center = (size // 2, size // 2)  # Center of the radial gradient
    # Create a 2D array and calculate the distance of each point from the center
    y, x = np.indices((size, size))
    distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Normalize the distances to match the length of the 1D gradient array
    distance_normalized = (distance_from_center / distance_from_center.max()) * (len(gradient_1d) - 1)
    # Use the normalized distances to map the 1D gradient to the 2D array
    radial_gradient = gradient_1d[distance_normalized.astype(int)]
    return(radial_gradient)

def prep_header(filename):
    header=fits.Header()
    header['FILENAME']=(filename, "Distortion Correction Matrix")
    header['VERSION']=('v1.0', 'Version name for the Image')
    header['MFG_DATE']=(str(date.today()), 'Manufacturing date for the FITS file')
    header['COMMENT']=("Distortion Correction Matrix")
    return (header)

def save_fits(array, name):
    save_path= os.path.join(sav, name)
    sav_hdu= fits.PrimaryHDU(array, header=prep_header(name))
    sav_hdu.writeto(save_path, overwrite=True)
    
def make_distortion(imsize, SAVE):
    if imsize==4096:
        px= 0.012 #pixels are 12 micron for 4k images
        typ='4k'
    elif imsize==2048:
        px=0.024 ##pixels are 24 micron for 2k images
        typ='2k'
    else:
        print("Invalid image size:", imsize)
    #read ZEMAX distortion profile and convert the table to a 2D array
    file= os.path.join(project_path, 'data/external/distortion_100x100.txt')
    predicted_pos= np.loadtxt(file, skiprows=1, usecols=(5,6))
    real_pos= np.loadtxt(file, skiprows=1, usecols=(7,8))
    shift_pos= real_pos-predicted_pos #to be subtracted from real_positions
    shift_x= np.flip(np.transpose(np.reshape(shift_pos[:,0],(101,101))), axis=(0,1)) #makes gird of shift along x
    shift_y= np.flip(np.transpose(np.reshape(shift_pos[:,1],(101,101))), axis=(0,1)) #makes gird of shift along y
    
    #make extrapolated linear arrays for x and y distortions from image center
    #to right bottom diagonal
    linear_grad_x= make_linear_grad(shift_x, imsize)
    linear_grad_y= make_linear_grad(shift_y, imsize)
    
    # Making radial gradients from linear profiles
    #The xx and yy help to flip the direction of the distortion along an axis.
    #Distortion values are in mm
    xx,yy= np.meshgrid(np.linspace(-1,1, imsize), np.linspace(-1,1, imsize))
    radial_x = make_radial_grad(linear_grad_x, imsize)*xx 
    radial_y = make_radial_grad(linear_grad_y, imsize)*yy
    
    #Distortion matrix converted to pixels and integer values
    radial_x_arr= np.rint(radial_x/px).astype(int)
    radial_y_arr= np.rint(radial_y/px).astype(int)
    if SAVE:
        save_fits(radial_x_arr, f"{typ}_distortion_x_axis.fits")
        save_fits(radial_y_arr, f"{typ}_distortion_y_axis.fits")
    return(radial_x_arr, radial_y_arr)

def test_case(image):
    hdu= fits.open(image)[0]
    imsize= hdu.header['NAXIS1']
    ### IMPLEMENTATION ###
    if imsize==4096:
        bleed_size=300  # +- 300 px bleed size around the image
    elif imsize==2048:
        bleed_size=150  # +- 150 px bleed size around the image
    radial_x_arr, radial_y_arr= make_distortion(imsize, SAVE=True)
    #making a blank matrix to put the distortion corrected values.
    corrected= np.zeros(shape=(imsize+2*bleed_size,imsize+2*bleed_size)) 
    #Distortion correction by shifting pixels
    #suncenter values
    crpix1, crpix2, rsun= hdu.header['CRPIX1'], hdu.header['CRPIX2'], hdu.header['R_SUN']
    image_data= np.flip((hdu.data), axis=(0,1))
    for i in range(imsize):
        for j in range(imsize):
            xshift= radial_x_arr[i,j]
            yshift= radial_y_arr[i,j]
            corrected[bleed_size+i+yshift, bleed_size+j+xshift]= image_data[i,j]
            #This barrel distorts the image to make the sun circular.
            #change to -yshift and -xshift for inducing pincushion distortion.
    #Optional visualization
    plt.figure()
    circle= plt.Circle((crpix1,crpix2), rsun, edgecolor='red', facecolor='none', linewidth=2)
    plt.subplot(1,2,1)
    plt.imshow(np.flip(image_data, axis=(0,1)), origin='lower', vmin= 0, vmax= 3.5e4)
    plt.gca().add_patch(circle)
    #plt.imshow(image_data-corrected[bleed_size:imsize+bleed_size, bleed_size:imsize+bleed_size], origin='lower')
    plt.title("Image")
    circle= plt.Circle((crpix1+bleed_size,crpix2+bleed_size), rsun, edgecolor='red', facecolor='none', linewidth=2)
    plt.subplot(1,2,2)
    plt.imshow(np.flip(corrected, axis=(0,1)), origin='lower', vmin= 0, vmax= 3.5e4)
    plt.gca().add_patch(circle)
    plt.title("Distortion corrected")
    plt.show()


if __name__=='__main__':
    project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/distortion/distortion_correction_project/')
    sav= os.path.join(project_path, 'data/external/')
    radial_x_arr_4k, radial_y_arr_4k= make_distortion(4096, SAVE=True) #make 4k dist profile
    radial_x_arr_2k, radial_y_arr_2k= make_distortion(2048, SAVE=True) #make 2k dist profile
    
    #Uncomment to test on an image
    #test_image= os.path.join(project_path, 'data/raw/SUT_T24_0725_000377_Lev1.0_2024-05-15T23.15.14.966_0971NB05.fits')
    #test_case(test_image)
