import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import os
from astropy.io import fits

project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/distortion/distortion_correction_project/')
image= fits.open(os.path.join(project_path, 'data/raw/SUT_T24_0725_000377_Lev1.0_2024-05-15T22.58.07.105_0972NB03.fits'))[0].data

distort_prof_path= os.path.join(project_path, 'data/external/file.txt')
distortion_profile= np.loadtxt(distort_prof_path, skiprows=1)

predicted_pos= distortion_profile[:,5:7]
real_pos= distortion_profile[:,7:9]

zero_point= real_pos[0]

predicted_pos= (predicted_pos-zero_point)/24e-3
real_pos=(real_pos-zero_point)/24e-3

tps = ski.transform.ThinPlateSplineTransform()
tps.estimate(predicted_pos, real_pos)
warped= ski.transform.warp(image, tps)

plt.imshow(warped, origin='lower')