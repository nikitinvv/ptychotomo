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
    n = 256  # object size n x,y
    nz = 192  # object size in z
    nmodes = 4
    nscan = 10000
    ntheta = 173  # number of angles (rotations)
    
    data = np.zeros([ntheta,nz,n],dtype='float32')
    theta = np.load('theta173sorted.npy')

    psi1angle = dxchange.read_tiff('siftalignedcut/data.tif').astype('float32')
    # # print(psi1angle.shape)
    # for center in range(128-16,128+16):
    #     with tomoalign.SolverTomo(theta, ntheta, 1, n, 1, center, 1) as tslv:
    #         print(center)
    #         u = np.zeros([1,n,n],dtype='float32')
    #         u = tslv.cg_tomo_batch(psi1angle[:,100:101], u, 32)
    #         dxchange.write_tiff(u[0],'test_center/r'+str(center)+'.tiff',overwrite=True)
    # exit()
    data = psi1angle        
    
    center = 129
    pnz = 24  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
    stepwin = 1
    ptheta=1
    start_win = nz
    ngpus = 8  # number of gpus
    niter = 180  # number of iterations in the ADMM scheme
    titer = 4  # number of inner ADMM iterations
    res = None

    with tomoalign.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        u = np.zeros([nz,n,n],dtype='float32')
        u = tslv.cg_tomo_batch(data, u, 128)
        dxchange.write_tiff(u,'reccg/r.tiff',overwrite=True)    
    res= tomoalign.admm_of(data, theta, pnz, ptheta,
                            center, ngpus, niter, start_win, stepwin, res, fname='recalignedcut/r')

    
