#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2025-06-09 05:59:54
@author: janmejoyarch
@hostname: suitpoc1

DESCRIPTION
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from astropy.io import fits


project_path= os.path.abspath('..')
image= glob.glob(os.path.join(project_path, 'data/raw/*.fits'))[1]

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
#Distortion correction by shifting pixels
_x,_y= np.meshgrid(np.arange(imsize), np.arange(imsize))

map_x= _x-radial_x_arr
map_y= _y-radial_y_arr

map_x= map_x.astype(np.float32)
map_y= map_y.astype(np.float32)

corrected_img = cv2.remap(image_data, map_x, map_y, interpolation=cv2.INTER_LINEAR)

crpix1, crpix2, rsun= HEADER['CRPIX1'], HEADER['CRPIX2'], HEADER['R_SUN']
#Optional visualization
plt.figure()
plt.imshow(corrected_img, origin='lower')
circle= plt.Circle((crpix1,crpix2), rsun, edgecolor='red', facecolor='none', linewidth=2)
plt.gca().add_patch(circle)
plt.show()




