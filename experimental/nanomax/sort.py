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
    
    data = np.zeros([ntheta,nz,n],dtype='float32')
    theta = np.load('theta173.npy')
    for k in range(0,ntheta):        
        # Load a 3D object
        data[k] = dxchange.read_tiff('psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')              

    ids = np.argsort(theta)
    theta = theta[ids]
    data = data[ids]    
    
    np.save('theta173sorted',theta)
    dxchange.write_tiff_stack(data,'sorted/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)            
