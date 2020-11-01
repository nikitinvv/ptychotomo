import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np
import scipy.ndimage as ndimage
import ptychotomo as pt
import ptychotomo.util as util

from random import sample 
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 166  # number of angles (rotations)
    voxelsize = 18.03*1e-7  # object voxel size
    energy = 12.4  # xray energy
    nprb = 64  # probe size
    det = [128, 128]  # detector size
    
    recover_prb = str2bool(sys.argv[1])
    swap_prb =  str2bool(sys.argv[2])
    align =  str2bool(sys.argv[3])    
    shake =  str2bool(sys.argv[4])
    noise = str2bool(sys.argv[5])  # apply discrete Poisson noise
    nmodes = int(sys.argv[6])

    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    diter = 4
    niter = 257  # ADMM iterations
    ptheta = 1
    pnz = 128  # number of slice partitions for simultaneous processing in tomography
    # Load a 3D object
    beta = dxchange.read_tiff('model/lego-imag.tiff')+1e-12
    delta = dxchange.read_tiff('model/lego-real.tiff')/2+1e-12
    obj = np.zeros([nz, n, n], dtype='complex64')
    obj[nz//2-beta.shape[0]//2:nz//2+beta.shape[0]//2, n//2-beta.shape[1]//2:n//2 +
        beta.shape[1]//2, n//2-beta.shape[2]//2:n//2+beta.shape[2]//2] = delta+1j*beta
    # obj[:52]=0
    
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb_amp = dxchange.read_tiff_stack(
        'model/prbamp_00000.tiff', ind=np.arange(nmodes)).astype('float32')
    prb_ang = dxchange.read_tiff_stack(
        'model/prbangle_00000.tiff', ind=np.arange(nmodes)).astype('float32')    
    prb[:] = (prb_amp*np.exp(1j*prb_ang))[:, 64-nprb //
                                                  2:64+nprb//2, 64-nprb//2:64+nprb//2]/det[0]/100#/4
    theta = np.load('model/theta.npy')[:ntheta]
    scan0 = np.load('model/scan.npy')[:, :ntheta]
    scan = np.zeros([2,ntheta,1024],dtype='float32')
    for k in range(ntheta):       
        scan[:,k,:] = scan0[:,k,sample(range(13689),1024)]    

    
    scan = (scan)*(n-nprb)/(scan.max())
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz, nmodes)

    # Compute data
    psi = slv.fwd_tomo_batch(obj)
    exppsi = slv.exptomo(psi)
    name = 'lego'+str(recover_prb)+str(swap_prb)+str(align)+str(shake)+str(noise)+str(nmodes)+str(scan.shape[2])
    

    if(shake):
        # s = np.zeros([ntheta,2],dtype='float32')
        # for k in range(ntheta):
        #     s[k] = np.int32((np.random.random(2)-0.5)*8)
        # np.save('s',s)
        s = np.int32(np.load('s.npy'))
        for k in range(ntheta):
            exppsi[k] = np.roll(exppsi[k],(s[k,0],s[k,1]),axis=(0,1))                    
    dxchange.write_tiff_stack(np.angle(exppsi),
                    'psiinit/psiangle'+name, overwrite=True)
    
    print(np.abs(exppsi).max(), (np.angle(exppsi)).max(), (np.angle(exppsi)).min())
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
    h1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j    
    h2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    h3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j


    psi1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    # psi1 = exppsi.copy()#np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j
    psi2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    # psi3 = exppsi.copy()#np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j
    psi3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    lamd2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')    

    u = np.zeros([nz, n, n], dtype='complex64', order='C')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32', order='C')


    psi1, prb = slv.cg_ptycho_batch(
                data, psi1, prb, scan, h1, lamd1, None, 32, model, recover_prb)
    lpsi1 = slv.logtomo(psi1)
    dxchange.write_tiff_stack(lpsi1.real,
                                          'lspci/psi', overwrite=True)                
    for k in range(ntheta):        
        # a = np.abs(np.angle(psi1[k])                    )        
        #a[a<0]=0
        # cm = ndimage.center_of_mass(a)   
        # print('1',cm)                          
        a = lpsi1[k].real
        a[a<3e-5]=0        
        cm = ndimage.center_of_mass(a)   
        # print('2',cm)                          
        scan[0,k]-=np.round(cm[1]-n//2+0.5)
        scan[1,k]-=np.round(cm[0]-nz//2+0.5)
        ids = np.where((scan[0,k]>n-1-nprb)+(scan[1,k]>nz-1-nprb)+(scan[0,k]<0)+(scan[1,k]<0))[0]
        scan[0,k,ids]=-1
        scan[1,k,ids]=-1
        data[k,ids] = 0
        # print(cm)
        # print((cm[0]-nz//2+0.5),(cm[1]-n//2+0.5))
        # print(psi1[k].shape)
        lpsi1[k] = np.roll(lpsi1[k],(-int(cm[0]-nz//2+0.5),-int(cm[1]-n//2+0.5)),axis=(0,1))      
    # print(np.min(scan))
    # print(np.max(scan))
    
    dxchange.write_tiff_stack(lpsi1.real,
                                          'clspci/psi', overwrite=True)                      
        
    psi1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1                    
    # ADMM
    u, psi3, psi2, psi1, flow, prb = slv.admm(
        data, psi3, psi2, psi1, flow, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, u, alpha, piter, titer, diter, niter, model, recover_prb, align, name)
