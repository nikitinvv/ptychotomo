import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign

if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 166  # number of angles (rotations)
    voxelsize = 18.03*1e-7  # object voxel size
    
    # Load a 3D object
    beta = dxchange.read_tiff('lego-real.tiff')    
    obj = np.zeros([nz, n, n], dtype='float32')
    obj[nz//2-beta.shape[0]//2:nz//2+beta.shape[0]//2, n//2-beta.shape[1]//2:n//2 +
        beta.shape[1]//2, n//2-beta.shape[2]//2:n//2+beta.shape[2]//2] = beta
    
    theta = np.load('theta.npy')[:ntheta]
    theta = np.sort(theta)
    print(theta)
    
    with tomoalign.solver_tomo.SolverTomo(theta, ntheta, nz, n, 32, n/2, 1) as tslv:
        data = tslv.fwd_tomo_batch(obj)
    dxchange.write_tiff(data,'data/datainit.tiff',overwrite=True)
    for k in range(1,ntheta):
            s = np.int32((np.random.random(2)-0.5)*16)
            data[k] = np.roll(data[k],(s[0],s[1]),axis=(0,1))     
    cm = np.zeros([ntheta,2])
    for m in range(ntheta):
        cm[m] = ndimage.center_of_mass(data[m])
        #print('b',cm[m])
        data[m] = np.roll(data[m],(-int(cm[m,0]-55),-int(cm[m,1]-64)),axis=(0,1))
        cm[m] = ndimage.center_of_mass(data[m])
    # tomoalign.gen_cyl_data(n, ntheta, pprot, adef)
    dxchange.write_tiff(data,'data/data.tiff',overwrite=True)
    
    center = n/2
    pnz = n  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    ptheta = 166  # number of projections for simultaneus processing by one GPU in the alignment sub-problem
    # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
    stepwin = 1
    start_win = n
    ngpus = 1  # number of gpus
    niter = 128  # number of iterations in the ADMM scheme
    titer = 4  # number of inner ADMM iterations
    alpha = 1e-8
    res = None
    
    res= tomoalign.admm_of(data, theta, pnz, ptheta,
                            center, ngpus, niter, start_win, stepwin,res,fname='rec/r')

    