import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign
from tomoalign.utils import find_min_max
from skimage.registration import phase_cross_correlation

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512  # object size n x,y
    nz = 512  # object size in z
    nmodes = 4
    nscan = 13689
    ntheta = 173  # number of angles (rotations)
    data = dxchange.read_tiff_stack(data_prefix+'rec_full_sorted_aligned_check/psiangle'+str(nmodes)+str(nscan)+'/r_00000.tiff',ind=np.arange(0,173)).astype('float32')       
    data = data[:,192:384,128:-128]
    dxchange.write_tiff_stack(data,data_prefix+'rec_full_sorted_aligned_check_crop/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    
