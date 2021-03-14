import tomopy
import dxchange
import numpy as np
import h5py
import sys
import skimage.feature

##################################### Inputs #########################################################################
ndsets = np.int(sys.argv[1])
theta_start = 0
theta_end = np.int(sys.argv[2])
name = sys.argv[4]
file_name = name+'.h5'
sino_start = 0
sino_end = 2048
flat_field_norm = True
flat_field_drift_corr = False  # Correct the intensity drift
remove_rings = True
binning = np.int(sys.argv[3])
######################################################################################################################


def preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings, FF_drift_corr=flat_field_drift_corr, downsapling=binning):

    if FF_norm:  # dark-flat field correction
        prj = tomopy.normalize(prj, flat, dark)
    if FF_drift_corr:  # flat field drift correction
        prj = tomopy.normalize_bg(prj, air=50)
    prj[prj <= 0] = 1  # check dark<data
    prj = tomopy.minus_log(prj)  # -logarithm
    if remove_rings:  # remove rings
        prj = tomopy.remove_stripe_fw(
             prj, level=7, wname='sym16', sigma=1, pad=True)
        #prj = tomopy.remove_stripe_ti(prj,2)
    if downsapling > 0:  # binning
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)
    return prj


if __name__ == "__main__":
    for k in range(ndsets):
    # read data
        prj, flat, dark, theta = dxchange.read_aps_32id(
            file_name, sino=(sino_start, sino_end), proj=(theta_end*k,theta_end*(k+1)))
        print(theta.shape)
        theta = theta[theta_end*k:theta_end*(k+1)]
        # preprocess
        prj = preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings,
                            FF_drift_corr=flat_field_drift_corr, downsapling=binning)
        print(prj.shape)

        np.save(name+'_bin'+str(binning)+str(k),prj)        
        np.save(name+'_theta'+str(k),theta)  
            