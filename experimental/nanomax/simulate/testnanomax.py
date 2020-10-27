import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np

import ptychotomo as pt
import ptychotomo.util as util

from random import sample 
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":

    # Model parameters
    n = 256+32  # object size n x,y
    nz = 256+32  # object size in z
    ntheta = 166  # number of angles (rotations)
    voxelsize = 18.03*1e-7  # object voxel size
    energy = 12.4  # xray energy
    nprb = 64  # probe size
    det = [128, 128]  # detector size
    noise = False  # apply discrete Poisson noise
    recover_prb = str2bool(sys.argv[1])
    swap_prb =  str2bool(sys.argv[2])
    align =  str2bool(sys.argv[3])    
    shake =  str2bool(sys.argv[4])
    nmodes = int(sys.argv[5])

    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    diter = 4
    niter = 256  # ADMM iterations
    ptheta = 1
    pnz = 32  # number of slice partitions for simultaneous processing in tomography
    # Load a 3D object
    beta = dxchange.read_tiff('model/beta-chip-256.tiff')/4
    delta = -dxchange.read_tiff('model/delta-chip-256.tiff')/4
    obj = cp.zeros([nz, n, n], dtype='complex64')
    obj[nz//2-beta.shape[0]//2:nz//2+beta.shape[0]//2, n//2-beta.shape[1]//2:n//2 +
        beta.shape[1]//2, n//2-beta.shape[2]//2:n//2+beta.shape[2]//2] = cp.array(delta+1j*beta)
    # obj[:52]=0
    
    prb = cp.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb_amp = dxchange.read_tiff_stack(
        'model/prbamp_00000.tiff', ind=np.arange(nmodes)).astype('float32')
    prb_ang = dxchange.read_tiff_stack(
        'model/prbangle_00000.tiff', ind=np.arange(nmodes)).astype('float32')
    prb[:] = cp.array(prb_amp*np.exp(1j*prb_ang))[:, 64-nprb //
                                                  2:64+nprb//2, 64-nprb//2:64+nprb//2]/det[0]/100
    theta = cp.load('model/theta.npy')[:ntheta]
    scan0 = cp.load('model/scan.npy')[:, :ntheta]
    scan = cp.zeros([2,ntheta,1000],dtype='float32')
    for k in range(ntheta):       
        scan[:,k,:] = scan0[:,k,sample(range(13689),1000)]    

    
    scan = scan*(n-nprb)/(scan.max())
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz, nmodes)

    # Compute data
    exppsi = slv.exptomo(slv.fwd_tomo_batch(obj))
    mmin,mmax = util.find_min_max(exppsi.get())
    
    name = str(recover_prb)+str(swap_prb)+str(align)+str(shake)+str(nmodes)
    if(shake):
        for k in range(ntheta):
            s = np.int32((np.random.random(2)-0.5)*8)
            exppsi[k] = cp.roll(exppsi[k],(s[0],s[1]),axis=(0,1))                    

    dxchange.write_tiff_stack(cp.angle(exppsi).get(),
                    'psiinit/psiangle'+name, overwrite=True)

    print(cp.abs(exppsi).max(), (cp.angle(exppsi)).max(), (cp.angle(exppsi)).min())
    data = slv.fwd_ptycho_batch(exppsi, prb, scan)
    #dxchange.write_tiff_stack(data[-1],  'data/proj0', overwrite=True)
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print("max intensity on the detector: ", np.amax(data))

    if(swap_prb):
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
        data, psi3, psi2, psi1, flow, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, u, alpha, mmin, mmax, piter, titer, diter, niter, model, recover_prb, align, name)
