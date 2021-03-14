import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import ptychotomo
import scipy.ndimage as ndimage
if __name__ == "__main__":

    n = 512
    ntheta = 480
    pprot = 96
    # adef = 10
    # ptychotomo.gen_cyl_data(n, ntheta, pprot, adef)
    data = dxchange.read_tiff('data/deformed_data.tiff')
    data = ndimage.zoom(data,[ntheta//data.shape[0],n//data.shape[1],n//data.shape[2]],order=0)
    [ntheta, nz, n] = data.shape
    theta = np.linspace(0, 4*np.pi, ntheta).astype('float32')    
    center = n/2
    pnz = 32  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    ptheta = 128  # number of projections for simultaneus processing by one GPU in the alignment sub-problem
    # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
    ngpus = 1  # number of gpus
    niteradmm = [5,5,5]  # number of iterations in the ADMM scheme
    startwin = [n,n//2,n//4]
    stepwin = [2,2,2]
    
    fname = 'data/tmp'
    uof = ptychotomo.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)

    dxchange.write_tiff(uof['u'], 'data/of_recon/recon/iter' +
                        str(niteradmm), overwrite=True)
