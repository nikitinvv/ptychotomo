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
    ntheta = 1  # number of angles (rotations)
    voxelsize = 18.03*1e-7  # object voxel size
    energy = 12.4 # xray energy
    nprb = 64  # probe size
    det = [128, 128]  # detector size
    recover_prb = False
    nmodes = 4
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    niter = 128  # ADMM iterations
    ptheta = min(29,ntheta)  # number of angular partitions for simultaneous processing in ptychography
    pnz = 32  # number of slice partitions for simultaneous processing in tomography    
    # Load a 3D object
    beta = dxchange.read_tiff('model/beta-chip-128.tiff')/8
    delta = -dxchange.read_tiff('model/delta-chip-128.tiff')/8
    obj = cp.zeros([nz,n,n],dtype='complex64')
    obj[nz//2-64:nz//2+64,n//2-64:n//2+64,n//2-64:n//2+64] = cp.array(delta+1j*beta)
    
    prb = cp.zeros([ntheta, nmodes, nprb, nprb],dtype='complex64')
    prb_amp = dxchange.read_tiff_stack('model/prbamp_00000.tiff', ind=np.arange(nmodes)).astype('float32')
    prb_ang = dxchange.read_tiff_stack('model/prbangle_00000.tiff', ind=np.arange(nmodes)).astype('float32')    
    prb[:] = cp.array(prb_amp*np.exp(1j*prb_ang))[:,64-nprb//2:64+nprb//2,64-nprb//2:64+nprb//2]/det[0]/100
    theta = cp.load('model/theta.npy')[:ntheta]
    scan = cp.load('model/scan.npy')[:,:ntheta,::32]
    scan = scan*(n-nprb)/(scan.max())/2
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz,nmodes)
    # Compute data
    prb[0,1,15,10]=cp.abs(prb).max()
    prb/=10
    psi0 = slv.exptomo(slv.fwd_tomo_batch(obj))
    print('phase max,min ',(cp.angle(psi0)).max(),(cp.angle(psi0)).min())
    # data = slv.fwd_ptycho_batch(psi0,prb,scan)

    prb0 = prb[:,1]

    r = 1/cp.sqrt(cp.sum(cp.abs(prb)))/np.sqrt(scan.shape[2])*np.sqrt(nprb)/cp.sqrt(cp.abs(prb).max())
    print(r)
    fpsi0 = slv.fwd_ptycho(psi0, prb0, scan)*r
    print(cp.linalg.norm(fpsi0))
    afpsi0 = slv.adj_ptycho(fpsi0, prb0, scan)*r
    a = cp.sum(fpsi0*cp.conj(fpsi0))
    b = cp.sum(psi0*cp.conj(afpsi0))
    c = cp.sum(psi0*cp.conj(psi0))
    d = cp.sum(afpsi0*cp.conj(afpsi0))
    print('Adjoint', a,b,a/b)
    print('Unity', b,c,b/c)
    print('Unity', b,d,b/d)


    # # psi0=10
    # psi0[0,nz//2,n//2]=cp.abs(psi0).max()*4

    # r = 1/cp.mean(cp.abs(psi0))/np.sqrt(scan.shape[2])
    # print(r)
    # fprb0 = slv.fwd_ptycho(psi0, prb0, scan)*r
    # print(cp.linalg.norm(fprb0))
    # afpsi0 = slv.adj_ptycho_prb(fprb0, psi0, scan)*r
    # a = cp.sum(fprb0*cp.conj(fprb0))
    # b = cp.sum(prb0*cp.conj(afpsi0))
    # c = cp.sum(prb0*cp.conj(prb0))
    # d = cp.sum(afpsi0*cp.conj(afpsi0))
    # print('Adjoint', a,b,a/b)
    # print('Unity', b,c,b/c)
    # print('Unity', b,d,b/d)


    