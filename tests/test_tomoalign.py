import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import ptychotomo

if __name__ == "__main__":

    n = 128
    ntheta = 192
    pprot = 96
    # adef = 10
    # ptychotomo.gen_cyl_data(n, ntheta, pprot, adef)
    data = dxchange.read_tiff('data/deformed_data.tiff')

    theta = np.linspace(0, 4*np.pi, ntheta).astype('float32')
    [ntheta, nz, n] = data.shape

    center = n/2
    pnz = 32  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    ptheta = 16  # number of projections for simultaneus processing by one GPU in the alignment sub-problem
    # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
    ngpus = 1  # number of gpus
    nitercg = 64
    niteradmm = [48,24]  # number of iterations in the ADMM scheme
    startwin = [128,64]
    stepwin = [2,2]

    ucg = ptychotomo.pcg(data, theta, pprot, pnz, center, ngpus, nitercg)
    dxchange.write_tiff(ucg['u'], 'data/cg_recon/recon/iter' +
                        str(nitercg), overwrite=True)

    fname = 'data/tmp'
    uof = ptychotomo.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)

    dxchange.write_tiff(uof['u'], 'data/of_recon/recon/iter' +
                        str(niteradmm), overwrite=True)
