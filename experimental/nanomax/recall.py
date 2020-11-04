import os
import signal
import sys
import h5py
import cupy as cp
import dxchange
import numpy as np
import matplotlib.pyplot as plt
import ptychotomo as pt
from random import sample 
import scipy.ndimage as ndimage

data_prefix = '/local/data/vnikitin/nanomax/'



if __name__ == "__main__":
    
    align = True
    ngpus = 8
    ntheta = 160
    nscan = 3500
   
    n = 512
    nz = 512
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = False
    
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 32  # ptychography iterations
    titer = 32  # tomography iterations
    diter = 32
    niter = 257  # ADMM iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography
    pnz = 8  # number of slice partitions for simultaneous processing in tomography
    center = 256.0
    
    nmodes = 4
    
    data = np.zeros([ntheta, nscan, det[0], det[1]], dtype='float32')
    scan = np.zeros([2, ntheta, nscan], dtype='float32')-1
    theta = np.zeros(ntheta, dtype='float32')
    
    for k in range(ntheta):
        print(k)
        ids = sample(range(13689),nscan)
        data0 = np.load(data_prefix+'data/data128_'+str(k)+'.npy')        
        scan0 = np.load(data_prefix+'data/scan128_'+str(k)+'.npy')        
        data[k] = data0[ids]        
        scan[:,k:k+1,:] = scan0[:,:,ids]
        theta[k] = np.load(data_prefix+'data/theta128_'+str(k)+'.npy')                         
        
    # Load a 3D object
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64',order='C')
    prb[:] = np.load(data_prefix+'data/prb128.npy')
    
    ids = np.argsort(theta)
    theta= theta[ids]
    scan = scan[:,ids]
    data = data[ids]

    shift = np.load(data_prefix+'data/shifts.npy')#[::160//ntheta]
    for k in range(ntheta):  
        print(shift[k])      
        scan[0,k]-=np.round(shift[k,1])
        scan[1,k]-=np.round(shift[k,0])
        scan[0,k]-=(268-256)
        ids = np.where((scan[0,k]>n-1-nprb)+(scan[1,k]>nz-1-nprb)+(scan[0,k]<0)+(scan[1,k]<0))[0]
        scan[0,k,ids]=-1
        scan[1,k,ids]=-1
        data[k,ids] = 0 #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    # Initial guess
    h1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1#+1e-10j    
    h2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    h3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1#+1e-10j
    

    psi1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1#+1e-10j    

    # psi1abs = dxchange.read_tiff_stack(data_prefix+'recTrueTrue47000psi1iterabs512/0_00000.tiff',ind=range(0,ntheta))
    # psi1angle = dxchange.read_tiff_stack(data_prefix+'recTrueTrue47000psi1iter512/0_00000.tiff',ind=range(0,ntheta))
    # psi1 = psi1abs*np.exp(1j*psi1angle)
    psi2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    psi3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1#+1e-10j
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    lamd2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')    
    u = np.zeros([nz, n, n], dtype='complex64', order='C')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32', order='C')


    data = np.fft.fftshift(data, axes=(2, 3))
    # Class gpu solver
    slv = pt.Solver(nscan, theta, center, det, voxelsize,
                    energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus)
    name = data_prefix+'rec'+str(recover_prb)+str(align)+str(nmodes)+str(scan.shape[2])
    
    # ADMM
    u, psi3, psi2, psi1, flow, prb = slv.admm(
        data, psi3, psi2, psi1, flow, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, u, alpha, piter, titer, diter, niter, model, recover_prb, align, name)

