#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:10:45 2024

@author: janmejoyarch
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

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

def make_linear_grad(distortion_profile, imsize):
    '''
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
    fit_diag= round(imsize*np.sqrt(2))
    x_new= np.linspace(-22, 122, fit_diag)
    y_new= a*x_new**3+b*x_new**2+c*x_new+d
    return(y_new)

project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/distortion/distortion_correction_project/')
image= os.path.join(project_path, 'data/raw/SUT_T24_0725_000377_Lev1.0_2024-05-15T23.15.14.966_0971NB05.fits')
bleed_size=300 # +- 300 px bleed size around the image
hdu= fits.open(image)[0]
imsize= hdu.header['NAXIS1']

crpix1, crpix2, rsun= hdu.header['CRPIX1'], hdu.header['CRPIX2'], hdu.header['R_SUN']
image_data= np.flip((hdu.data), axis=(0,1))

file= os.path.join(project_path, 'data/external/distortion_100x100.txt')
index= np.loadtxt(file, skiprows=1, usecols=(0,1))
predicted_pos= np.loadtxt(file, skiprows=1, usecols=(5,6))
real_pos= np.loadtxt(file, skiprows=1, usecols=(7,8))
shift_pos= real_pos-predicted_pos #to be subtracted from real_positions

index_grid_x= np.transpose(np.reshape(index[:,0],(101,101)))
index_grid_y= np.transpose(np.reshape(index[:,1],(101,101)))

shift_x= np.flip(np.transpose(np.reshape(shift_pos[:,0],(101,101))), axis=(0,1)) #makes gird of shift along x
shift_y= np.flip(np.transpose(np.reshape(shift_pos[:,1],(101,101))), axis=(0,1)) #makes gird of shift along y

linear_grad_x= make_linear_grad(shift_x, imsize)[2896:]
linear_grad_y= make_linear_grad(shift_y, imsize)[2896:]

xx,yy= np.meshgrid(np.linspace(-1,1, imsize), np.linspace(-1,1, imsize))
radial_x = make_radial_grad(linear_grad_x, imsize)*xx  # Example: Linear gradient from 0 to 1
radial_y = make_radial_grad(linear_grad_y, imsize)*yy  # Example: Linear gradient from 0 to 1

corrected= np.zeros(shape=(imsize+2*bleed_size,imsize+2*bleed_size)) #making a blank matrix to put the distortion corrected values.

for i in range(imsize):
    for j in range(imsize):
        xshift=round(radial_x[i,j]/0.012)
        yshift=round(radial_y[i,j]/0.012)
        corrected[bleed_size+i+yshift, bleed_size+j+xshift]= image_data[i,j]

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

'''
plt.figure()
plt.subplot(2,2,1)
plt.imshow(shift_y)
plt.subplot(2,2,2)
plt.imshow(radial_y)
plt.subplot(2,2,3)
plt.imshow(shift_x)
plt.subplot(2,2,4)
plt.imshow(radial_x)

i= np.arange(101)
I,J= np.meshgrid(i,i) #Its a blank square of side 100

f_x= interp2d(I,J,shift_x, kind='cubic')
f_y= interp2d(J,I, shift_y, kind='cubic') #interchanging the rows and cols to avoid divergent vals. 

#i_new= np.linspace(0, 100, 2837)
i_new= np.linspace(0, 144, 4096)

shift_x_new=f_x(i_new, i_new)/0.012 #dividing by pixel size to get shift in pixels
shift_y_new= np.transpose(f_y(i_new, i_new)/0.012) #dividing by pixel size to get shift in pixels

'''