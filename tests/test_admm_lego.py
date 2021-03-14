import numpy as np
import dxchange
import ptychotomo
from random import sample, randint
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
if __name__ == "__main__":

    # read object
    n = 256  # object size n x,y
    nz = 256  # object size in z
    ntheta = 64  # number of angles
    pnz = 32  # partial size for nz
    ptheta = 64  # partial size for ntheta
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    ndet = 128  # detector size
    nprb = 128  # probe size
    center = n/2  # rotation center
    nmodes = 4  # number of probe modes
    nscan = 300  # number of scan positions per each angle
    ngpus = 1  # number of GPUs

    # reconstruction paramters
    recover_prb = True  # recover probe or not
    piter = 8  # ptycho iterations
    titer = 8  # tomo iterations
    diter = 8  # deform iterations
    niter = 128  # admm iterations
    maxshift = 3  # max random shift of projections (for testing)

    # Load a 3D object
    delta = dxchange.read_tiff('data_lego/delta-lego-256.tiff')
    beta = dxchange.read_tiff('data_lego/beta-lego-256.tiff')
    u = delta+1j*beta

    # Load probe
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb_amp = dxchange.read_tiff('data/probes_amp.tiff')[:nmodes]
    prb_ang = dxchange.read_tiff('data/probes_ang.tiff')[:nmodes]
    prb[:] = prb_amp*np.exp(1j*prb_ang)

    # Load scan positions
    scan = np.zeros([2, ntheta, nscan], dtype='float32') - \
        1  # -1 to be skipped in computations
    scan_all = np.load('data_lego/scan.npy')
    for k in range(ntheta):
        scan0 = scan_all[:, k]
        ids = np.where((scan0[0] < n-nprb-maxshift)*(scan0[1] < nz-nprb -
                                                     maxshift)*(scan0[1] >= maxshift)*(scan0[0] >= maxshift))[0]
        ids = ids[sample(range(len(ids)), min(len(ids), nscan))]
        scan[:, k, :len(ids)] = scan0[:, ids]
        plt.plot(scan[0], scan[1], 'r.')
        plt.savefig(f'data_lego/scan{k:03}.png')
        plt.clf()

    # init rotation angles
    theta = np.linspace(0, np.pi, ntheta).astype('float32')

    # Form data
    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        psi = tslv.fwd_tomo_batch(u)
    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi = pslv.exptomo(psi)
        data = pslv.fwd_ptycho_batch(psi, prb, scan)
        data = np.sum(np.abs(data)**2, axis=1)

    # Shift scan positions
    scan[0] += (np.random.random([ntheta, 1])-0.5)*2*maxshift
    scan[1] += (np.random.random([ntheta, 1])-0.5)*2*maxshift

    # Initial guess
    # variable index: 1 - ptycho problem, 2 - regularization (not implemented), 3 - tomo
    # used as in the pdf documents
    h1 = np.ones([ntheta, nz, n], dtype='complex64')
    h3 = np.ones([ntheta, nz, n], dtype='complex64')
    psi1 = np.ones([ntheta, nz, n], dtype='complex64')
    psi3 = np.ones([ntheta, nz, n], dtype='complex64')
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64')
    lamd3 = np.zeros([ntheta, nz, n], dtype='complex64')
    u = np.zeros([nz, n, n], dtype='complex64')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')

    with ptychotomo.SolverAdmm(nscan, theta, center, ndet, voxelsize, energy,
                               ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus) as aslv:
        u, psi1, psi3, flow, prb = aslv.admm(
            data, psi1, psi3, flow, prb, scan,
            h1, h3, lamd1, lamd3,
            u, piter, titer, diter, niter, recover_prb, 'tmp/', dbg_step=1)

    dxchange.write_tiff_stack(
        np.angle(psi1), 'data_lego/rec_admm/psiangle/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.abs(psi1), 'data_lego/rec_admm/psiamp/p', overwrite=True)
    dxchange.write_tiff_stack(u.real, 'data/rec_admm/ure/u', overwrite=True)
    dxchange.write_tiff_stack(u.imag, 'data/rec_admm/uim/u', overwrite=True)
