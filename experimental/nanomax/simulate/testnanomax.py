import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np

import ptychotomo as pt

if __name__ == "__main__":

    # Model parameters
    n = 256  # object size n x,y
    nz = 256  # object size in z
    ntheta = 174  # number of angles (rotations)
    voxelsize = 18.03*1e-7  # object voxel size
    energy = 12.4  # xray energy
    nprb = 64  # probe size
    det = [128, 128]  # detector size
    noise = False  # apply discrete Poisson noise
    recover_prb = True
    nmodes = 4
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    diter = 4
    niter = 256  # ADMM iterations
    ptheta = 1
    pnz = 16  # number of slice partitions for simultaneous processing in tomography
    # Load a 3D object
    beta = dxchange.read_tiff('model/beta-chip-256.tiff')/4
    delta = -dxchange.read_tiff('model/delta-chip-256.tiff')/4
    obj = cp.zeros([nz, n, n], dtype='complex64')
    obj[nz//2-beta.shape[0]//2:nz//2+beta.shape[0]//2, n//2-beta.shape[1]//2:n//2 +
        beta.shape[1]//2, n//2-beta.shape[2]//2:n//2+beta.shape[2]//2] = cp.array(delta+1j*beta)

    prb = cp.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb_amp = dxchange.read_tiff_stack(
        'model/prbamp_00000.tiff', ind=np.arange(nmodes)).astype('float32')
    prb_ang = dxchange.read_tiff_stack(
        'model/prbangle_00000.tiff', ind=np.arange(nmodes)).astype('float32')
    prb[:] = cp.array(prb_amp*np.exp(1j*prb_ang))[:, 64-nprb //
                                                  2:64+nprb//2, 64-nprb//2:64+nprb//2]/det[0]/100
    theta = cp.load('model/theta.npy')[:ntheta]
    scan = cp.load('model/scan.npy')[:, :ntheta, ::8]
    scan = scan*(n-nprb)/(scan.max())
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz, nmodes)

    # Compute data
    exppsi = slv.exptomo(slv.fwd_tomo_batch(obj))
    for k in range(ntheta):
        s = np.int32((np.random.random(2)-0.5)*10)
        exppsi[k] = cp.roll(exppsi[k],(s[0],s[1]))    
    print(cp.abs(exppsi).max(), (cp.angle(exppsi)).max(), (cp.angle(exppsi)).min())
    data = slv.fwd_ptycho_batch(exppsi, prb, scan)
    #dxchange.write_tiff_stack(data[-1],  'data/proj0', overwrite=True)
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print("max intensity on the detector: ", np.amax(data))

    nmodes = 4
    prb = prb.swapaxes(2,3)
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz, nmodes)
    # Initial guess
    h1 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    h2 = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    h3 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi1 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi2 = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    psi3 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    lamd1 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    lamd2 = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd3 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')    
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32', order='C')

    # ADMM
    u, psi3, psi2, psi1, flow, prb = slv.admm(
        data, psi3, psi2, psi1, flow, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, u, alpha, piter, titer, diter, niter, model, recover_prb)
    # Save result
    name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + 'ntheta' + \
        str(ntheta)+str(model)+str(piter)+str(titer)+str(niter)

    dxchange.write_tiff(u.imag.get(),  'beta/beta'+name, overwrite=True)
    dxchange.write_tiff(-u.real.get(),  'delta/delta'+name,
                        overwrite=True)  # note sign change
    # dxchange.write_tiff(cp.angle(psi).get(),
    #                     'psi/psiangle'+name, overwrite=True)
    # dxchange.write_tiff(cp.abs(psi).get(),  'psi/psiamp'+name, overwrite=True)
    # dxchange.write_tiff(cp.angle(prb).get(),
    #                     'prb/prbangle'+name, overwrite=True)
    # dxchange.write_tiff(cp.abs(prb).get(),  'prb/prbamp'+name, overwrite=True)
