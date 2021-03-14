import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign
from tomoalign.utils import find_min_max
from skimage.registration import phase_cross_correlation
#from silx.image import sift
from pystackreg import StackReg

data_prefix = '/local/data/vnikitin/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512  # object size n x,y
    nz = 512  # object size in z
    nmodes = 4
    nscan = 10000
    ntheta = 173  # number of angles (rotations)
    data = dxchange.read_tiff_stack('sorted/psiangle'+str(nmodes)+str(nscan)+'/r_00000.tiff', ind=np.arange(0,173))    .astype('float32')    
    data_sift = dxchange.read_tiff('siftaligned/data.tif').astype('float32')    
    print(data.shape)
    print(data_sift.shape)
    shift = np.zeros((ntheta,2),dtype='float32')
    for k in range(ntheta):
        shift0, error, diffphase = phase_cross_correlation(data[k], data_sift[k], upsample_factor=1)
        print(shift0)
        data[k]=np.roll(data[k],(-int(shift0[0]),-int(shift0[1])),axis=(0,1))
        shift[k]=shift0
    dxchange.write_tiff_stack(data,'sift_aligned_check/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    np.save('shifts173.npy',shift)

