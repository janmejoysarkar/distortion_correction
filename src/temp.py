#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:10:45 2024

@author: janmejoyarch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, interp1d
import os
from astropy.io import fits

def make_radial_grad(gradient_1d):
    # Step 2: Define the size of the 2D array and the center of the radial gradient
    size = 4096  # Size of the 2D array (size x size)
    center = (size // 2, size // 2)  # Center of the radial gradient
    
    # Step 3: Create a 2D array and calculate the distance of each point from the center
    y, x = np.indices((size, size))
    distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Normalize the distances to match the length of the 1D gradient array
    distance_normalized = (distance_from_center / distance_from_center.max()) * (len(gradient_1d) - 1)
    
    # Step 4: Use the normalized distances to map the 1D gradient to the 2D array
    radial_gradient = gradient_1d[distance_normalized.astype(int)]

    return(radial_gradient)

def make_linear_grad(distortion_profile):
    a,b,c,d= np.polyfit(np.arange(101), np.diagonal(distortion_profile), deg=3)
    x_new= np.linspace(-22, 122, 5792)
    y_new= a*x_new**3+b*x_new**2+c*x_new+d
    return(y_new)

project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/distortion/distortion_correction_project/')
file= os.path.join(project_path, 'data/external/distortion_100x100.txt')

index= np.loadtxt(file, skiprows=1, usecols=(0,1))
predicted_pos= np.loadtxt(file, skiprows=1, usecols=(5,6))
real_pos= np.loadtxt(file, skiprows=1, usecols=(7,8))
shift_pos= real_pos-predicted_pos #to be subtracted from real_positions

index_grid_x= np.transpose(np.reshape(index[:,0],(101,101)))
index_grid_y= np.transpose(np.reshape(index[:,1],(101,101)))

shift_x= np.flip(np.transpose(np.reshape(shift_pos[:,0],(101,101))), axis=(0,1)) #makes gird of shift along x
shift_y= np.flip(np.transpose(np.reshape(shift_pos[:,1],(101,101))), axis=(0,1)) #makes gird of shift along y

linear_grad_x= make_linear_grad(shift_x)[2896:]
linear_grad_y= make_linear_grad(shift_y)[2896:]

xx,yy= np.meshgrid(np.linspace(-1,1, 4096), np.linspace(-1,1, 4096))
radial_x = make_radial_grad(linear_grad_x)*xx  # Example: Linear gradient from 0 to 1
radial_y = make_radial_grad(linear_grad_y)*yy  # Example: Linear gradient from 0 to 1

plt.subplot(1,2,1)
plt.imshow(shift_y)
plt.subplot(1,2,2)
plt.imshow(radial_y)
'''
i= np.arange(101)
I,J= np.meshgrid(i,i) #Its a blank square of side 100

f_x= interp2d(I,J,shift_x, kind='cubic')
f_y= interp2d(J,I, shift_y, kind='cubic') #interchanging the rows and cols to avoid divergent vals. 

#i_new= np.linspace(0, 100, 2837)
i_new= np.linspace(0, 144, 4096)

shift_x_new=f_x(i_new, i_new)/0.012 #dividing by pixel size to get shift in pixels
shift_y_new= np.transpose(f_y(i_new, i_new)/0.012) #dividing by pixel size to get shift in pixels

'''