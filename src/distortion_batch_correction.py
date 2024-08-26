#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:32:18 2024

@author: janmejoyarch
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

def save_fits(array, name):
    sav= os.path.join(project_path, 'products/', name)
    sav_hdu= fits.PrimaryHDU(array, header=HEADER)
    sav_hdu.writeto(sav, overwrite=True)
    
    
    
project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/distortion/distortion_correction_project/')
#image to be corrected
image= os.path.join(project_path, 'data/raw/test.fits')
SAVE=True
VISUALIZE=True
hdu= fits.open(image)[0]
HEADER= hdu.header
imsize= hdu.header['NAXIS1']
if imsize==4096:
    bleed_size=300  # +- 300 px bleed size around the image
    px= 0.012 #pixels are 12 micron for 4k images
    radial_x_arr= fits.open(os.path.join(project_path, 'data/external/4k_distortion_x_axis.fits'))[0].data
    radial_y_arr= fits.open(os.path.join(project_path, 'data/external/4k_distortion_y_axis.fits'))[0].data
elif imsize==2048:
    bleed_size=150  # +- 150 px bleed size around the image
    px=0.024 ##pixels are 24 micron for 2k images
    radial_x_arr= fits.open(os.path.join(project_path, 'data/external/2k_distortion_x_axis.fits'))[0].data
    radial_y_arr= fits.open(os.path.join(project_path, 'data/external/2k_distortion_y_axis.fits'))[0].data
else:
    print("Invalid image size:", imsize)

crpix1, crpix2, rsun= hdu.header['CRPIX1'], hdu.header['CRPIX2'], hdu.header['R_SUN']

image_data= np.flip((hdu.data), axis=(0,1)) #vert and horz mirroring
#making a blank matrix to put the distortion corrected values.
corrected= np.zeros(shape=(imsize+2*bleed_size,imsize+2*bleed_size)) 
#Distortion correction by shifting pixels
for i in range(imsize):
    for j in range(imsize):
        xshift= radial_x_arr[i,j]
        yshift= radial_y_arr[i,j]
        corrected[bleed_size+i+yshift, bleed_size+j+xshift]= image_data[i,j]
        #This barrel distorts the image to make the sun circular.
        #change to -yshift and -xshift for inducing pincushion distortion.

if SAVE: save_fits(np.flip(corrected, axis=(0,1)), os.path.basename(image))
    

if VISUALIZE:
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

    