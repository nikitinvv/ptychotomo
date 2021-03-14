import numpy as np
import dxchange
import ptychotomo
from random import sample
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # read object
    n = 256  # object size n x,y
    nz = 256  # object size in z
    pnz = 32
    ntheta = 1  # number of angles (rotations)
    ptheta = 1
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    ndet = 128
    nprb = 128
    center = n/2
    nmodes = 4
    nscan = 1000
    ngpus = 1
    recover_prb = True
    piter = 128

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
        scan0 = np.load(f'data/scan128sorted_{k}.npy')
        scan0[1]-=30
        ids = np.where((scan0[1, 0] < n-nprb)*(scan0[0, 0] <
                                               nz-nprb)*(scan0[0, 0] >= 0)*(scan0[1, 0] >= 0))[0]
        ids = ids[sample(range(len(ids)), min(len(ids), nscan))]
        scan[0, k, :len(ids)] = scan0[1, 0, ids]
        scan[1, k, :len(ids)] = scan0[0, 0, ids]
        #plt.plot(scan[0], scan[1], 'r.')
        # plt.savefig(f'data/scan{k:03}.png')

    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        psi = tslv.fwd_tomo_batch(u)        
    
    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi = pslv.exptomo(psi)
        data = pslv.fwd_ptycho_batch(psi, prb, scan)
        data = np.sum(np.abs(data)**2, axis=1)
    print(f'{data.shape =}, {data.dtype=}')    
    
    psiinit = psi*0 + 1
    prbinit = prb.swapaxes(2,3)
    #nmodes = 2
    prbinit = prb[:,:nmodes]
    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi, prb = pslv.cg_ptycho_batch(data, psiinit, prbinit, scan, piter, recover_prb)
    dxchange.write_tiff(np.angle(psi[0]),'data/rec_ptycho/psiangle',overwrite=True)    
    dxchange.write_tiff(np.abs(psi[0]),'data/rec_ptycho/psiamp',overwrite=True)    
    dxchange.write_tiff_stack(np.angle(prb[0]),'data/rec_ptycho/prbangle',overwrite=True)    
    dxchange.write_tiff_stack(np.abs(prb[0]),'data/rec_ptycho/prbamp',overwrite=True)    
    
    
