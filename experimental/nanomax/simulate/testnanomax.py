import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np

import ptychotomo as pt

if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 174  # number of angles (rotations)
    voxelsize = 18.03*1e-7  # object voxel size
    energy = 12.4  # xray energy
    nprb = 32  # probe size
    det = [128, 128]  # detector size
    noise = False  # apply discrete Poisson noise
    recover_prb = False
    nmodes = 2
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    niter = 64  # ADMM iterations
    ptheta = 1
    pnz = 32  # number of slice partitions for simultaneous processing in tomography
    # Load a 3D object
    beta = dxchange.read_tiff('model/beta-chip-128.tiff')/8
    delta = -dxchange.read_tiff('model/delta-chip-128.tiff')/8
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
    scan = cp.load('model/scan.npy')[:, :ntheta, ::32]
    scan = scan*(n-nprb)/(scan.max())
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz, nmodes)

    # Compute data
    a = slv.exptomo(slv.fwd_tomo_batch(obj))
    print(cp.abs(a).max(), (cp.angle(a)).max(), (cp.angle(a)).min())
    data = slv.fwd_ptycho_batch(slv.exptomo(
        slv.fwd_tomo_batch(obj)), prb, scan)
    dxchange.write_tiff_stack(data[-1],  'data/proj0', overwrite=True)
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print("max intensity on the detector: ", np.amax(data))

    nmodes = 2
    prb = prb[:, :nmodes].swapaxes(2, 3)
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz, nmodes)
    # Initial guess
    h = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    e = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')

    # ADMM
    u, psi, prb = slv.admm(data, psi, phi, prb, scan, h, e,  lamd,
                           mu, u, alpha, piter, titer, niter, model, recover_prb)
    # Save result
    name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + 'ntheta' + \
        str(ntheta)+str(model)+str(piter)+str(titer)+str(niter)

    dxchange.write_tiff(u.imag.get(),  'beta/beta'+name, overwrite=True)
    dxchange.write_tiff(-u.real.get(),  'delta/delta'+name,
                        overwrite=True)  # note sign change
    dxchange.write_tiff(cp.angle(psi).get(),
                        'psi/psiangle'+name, overwrite=True)
    dxchange.write_tiff(cp.abs(psi).get(),  'psi/psiamp'+name, overwrite=True)
    dxchange.write_tiff(cp.angle(prb).get(),
                        'prb/prbangle'+name, overwrite=True)
    dxchange.write_tiff(cp.abs(prb).get(),  'prb/prbamp'+name, overwrite=True)
