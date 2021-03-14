import numpy as np
import dxchange
import ptychotomo
from random import sample, randint
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # read object
    n = 256  # object size n x,y
    nz = 256  # object size in z
    pnz = 32
    ntheta = 64  # number of angles (rotations)
    ptheta = 32
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    ndet = 128
    nprb = 128
    center = n/2
    nmodes = 1
    nscan = 600
    ngpus = 1
    recover_prb = False
    piter = 4
    titer = 4
    diter = 4
    niter = 4

    theta = np.linspace(0, np.pi, ntheta).astype('float32')

    # Load a 3D object
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')
    delta = -dxchange.read_tiff('data/delta-chip-256.tiff')
    u = delta+1j*beta

    # Load probe
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb_amp = dxchange.read_tiff('data/probes_amp.tiff')[:nmodes]
    prb_ang = dxchange.read_tiff('data/probes_ang.tiff')[:nmodes]
    prb[:] = prb_amp*np.exp(1j*prb_ang)

    # Load scan positions
    scan = np.zeros([2, ntheta, nscan], dtype='float32') - 1
    for k in range(ntheta):
        scan0 = np.load(f'data/scan/scan128sorted_{k}.npy')
        scan0[1] -= 30
        ids = np.where((scan0[1, 0] < n-nprb)*(scan0[0, 0] <
                                               nz-nprb)*(scan0[0, 0] >= 0)*(scan0[1, 0] >= 0))[0]
        ids = ids[sample(range(len(ids)), min(len(ids), nscan))]
        scan[0, k, :len(ids)] = scan0[1, 0, ids]
        scan[1, k, :len(ids)] = scan0[0, 0, ids]
        #plt.plot(scan[0], scan[1], 'r.')
        # plt.savefig(f'data/scan{k:03}.png')        
        scan[0, k, :len(ids)] += (np.random.rand(1)-0.5)*4
        scan[1, k, :len(ids)] += (np.random.rand(1)-0.5)*4
    # form data
    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        psi = tslv.fwd_tomo_batch(u)
    
    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi = pslv.exptomo(psi)
        data = pslv.fwd_ptycho_batch(psi, prb, scan)
        data = np.sum(np.abs(data)**2, axis=1)


    # Initial guess
    h1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    h3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    # psi1abs = dxchange.read_tiff_stack(data_prefix+'recTrueTrue47000psi1iterabs512/0_00000.tiff',ind=range(0,ntheta))
    # psi1angle = dxchange.read_tiff_stack(data_prefix+'recTrueTrue47000psi1iter512/0_00000.tiff',ind=range(0,ntheta))
    # psi1 = psi1abs*np.exp(1j*psi1angle)
    psi3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    lamd3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    u = np.zeros([nz, n, n], dtype='complex64', order='C')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32', order='C')

    with ptychotomo.SolverAdmm(nscan, theta, center, ndet, voxelsize, energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus) as aslv:
        u, psi3, psi1, flow, prb = aslv.admm(
            data, psi3, psi1, flow, prb, scan,
            h3, h1, lamd3, lamd1,
            u, piter, titer, diter, niter, recover_prb)

    dxchange.write_tiff_stack(
        np.angle(psi1), 'data/rec_admm/psiangle/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.abs(psi1), 'data/rec_admm/psiamp/p', overwrite=True)
    dxchange.write_tiff_stack(u.real, 'data/rec_admm/ure/u', overwrite=True)
    dxchange.write_tiff_stack(u.imag, 'data/rec_admm/uim/u', overwrite=True)