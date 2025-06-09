#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2025-06-09 05:59:54
@author: janmejoyarch
@hostname: suitpoc1

DESCRIPTION
- Distortion correction in SUIT images.
- Missing pixels are replaced by interpolation.
- OpenCV used.
- Visualization creates problems at times.
- Distortion maps generated using data from Zemax.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor

def save_fits(array, HEADER, name):
    sav= os.path.join(project_path, 'products/', name)
    sav_hdu= fits.PrimaryHDU(array, header=HEADER)
    sav_hdu.writeto(sav, overwrite=True)
project_path= os.path.abspath('..')
image= glob.glob(os.path.join(project_path, 'data/raw/*.fits'))[1]

def visualize(HEADER, corrected_img):
    crpix1, crpix2, rsun= HEADER['CRPIX1'], HEADER['CRPIX2'], HEADER['R_SUN']
    plt.figure()
    plt.imshow(corrected_img, origin='lower')
    circle= plt.Circle((crpix1,crpix2), rsun, edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(circle)
    plt.show()

def run(image):
    print(os.path.basename(image))
    hdu= fits.open(image)[0]
    HEADER= hdu.header
    imsize= HEADER['NAXIS1']
    if imsize==4096:
        bleed_size=300  # +- 300 px bleed size around the image
        radial_x_arr= fits.open(os.path.join(project_path, 'data/external/4k_distortion_x_axis.fits'))[0].data
        radial_y_arr= fits.open(os.path.join(project_path, 'data/external/4k_distortion_y_axis.fits'))[0].data
    elif imsize==2048:
        bleed_size=150  # +- 150 px bleed size around the image
        radial_x_arr= fits.open(os.path.join(project_path, 'data/external/2k_distortion_x_axis.fits'))[0].data
        radial_y_arr= fits.open(os.path.join(project_path, 'data/external/2k_distortion_y_axis.fits'))[0].data
    else:
        print("Invalid image size:", imsize)
    image_data= hdu.data.astype(np.float32)
    corrected= np.zeros(shape=(imsize+2*bleed_size,imsize+2*bleed_size)) 
    _x,_y= np.meshgrid(np.arange(imsize), np.arange(imsize)) # Array giving the corresp. row/ col num
    # The absolute position of the shifted pixels.
    map_x= _x-radial_x_arr 
    map_y= _y-radial_y_arr
    # Convert to 32 bit
    map_x= map_x.astype(np.float32)
    map_y= map_y.astype(np.float32)
    corrected_img = cv2.remap(image_data, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    if SAVE: save_fits(corrected_img, HEADER,os.path.basename(image))
    if VISUALIZE: visualize(HEADER, corrected_img)

if __name__=="__main__":      
    project_path= os.path.abspath('..')
    image_list= glob.glob(os.path.join(project_path, 'data/raw/*.fits'))
    SAVE=True
    VISUALIZE=True
    print("Distortion correction:")
    with ProcessPoolExecutor() as executor:
        executor.map(run, image_list)
