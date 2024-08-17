#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:09:34 2023
Modified on Sat Oct 28 2023

-Generates distortion map and applies correction by shifting pixels on a new array.
-For visualization, pixels are shifted as rounded off integers on a synthetic pattern.
-Partial pixel shifting with flux conservation has to be introduced.
-The distortion model works for X and Y axes.
-For Y axis, the interpolation was diverging, although the data was fine.
So, for interpolation, I interchanged the rows and cols, and after interpolation, 
took a transpose to generate the y distortion map.
-The function `distortion_map` gives the x and y distortion matrices in units of px.

Columns in data file-
# i j  X-Field  Y-Field  R-Field   Predicted X   Predicted Y   Real X   Real Y  Distortion

@author: janmejoy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import os
from astropy.io import fits

def distortion_map(file): #run to create the distortion map
    predicted_pos= np.loadtxt(file, skiprows=1, usecols=(5,6))
    real_pos= np.loadtxt(file, skiprows=1, usecols=(7,8))
    shift_pos= real_pos-predicted_pos #to be subtracted from real_positions
    
    shift_x= np.transpose(np.reshape(shift_pos[:,0],(101,101))) #makes gird of shift along x
    shift_y= np.transpose(np.reshape(shift_pos[:,1],(101,101))) #makes gird of shift along y
    
    i= np.arange(101)
    I,J= np.meshgrid(i,i) #Its a blank square of side 100
    
    f_x= interp2d(I,J,shift_x, kind='cubic')
    f_y= interp2d(J,I, shift_y, kind='cubic') #interchanging the rows and cols to avoid divergent vals. 
    
    i_new= np.linspace(0, 100, 2837)
    
    shift_x_new=f_x(i_new, i_new)/0.012 #dividing by pixel size to get shift in pixels
    shift_y_new= np.transpose(f_y(i_new, i_new)/0.012) #dividing by pixel size to get shift in pixels
    #Taking transpose to reverse the interchange of rows and cols.
    return(shift_x_new[75:-75, 75:-75], shift_y_new[75:-75, 75:-75])

def test_pattern(size):
    zeros= np.zeros(shape=(size, size))
    for n in range(15):
        zeros[2*n*100:(2*n+1)*100]=1
        zeros[:, 2*n*100:(2*n+1)*100]=1
    return(zeros)
    
def checkerboard(n): #Generates pixel level checkerboard. Not being used here.
    chkboard=[]
    for i in range(n):
        chkboard.append(list(np.tile([1,0], int(n/2))) if i%2==0 else list(np.tile(([0,1]), int(n/2))))
    return(np.array(chkboard))


if __name__=='__main__':
    project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/distortion/distortion_correction_project/')
    image= os.path.join(project_path, 'data/raw/SUT_T24_0725_000377_Lev1.0_2024-05-15T23.13.56.502_0971NB04.fits')
    image_data= fits.open(image)[0].data
    
    dist_x, dist_y= distortion_map(os.path.join(project_path, 'data/external/distortion_100x100.txt')) 
    #gives x and y distortion map. Enter file path.
    pattern= test_pattern(dist_x.shape[0])
    corrected= np.zeros(shape=(dist_x.shape)) #making a blank matrix to put the distortion corrected values.

    #Shifting the pixels along x on pattern: Based on dist_x
    for i in range(dist_x.shape[0]):
        for j in range(dist_x.shape[0]):
            xshift=round(dist_x[i,j])
            yshift=round(dist_y[i,j])
            corrected[i-yshift, j-xshift]=image_data[i,j] 

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pattern)
    plt.title("Test Pattern")
    plt.subplot(1,2,2)
    plt.imshow(corrected)
    plt.title("Distortion corrected Test Pattern")
    plt.show()
