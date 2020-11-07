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
import flowvis

data_prefix = '/local/data/vnikitin/nanomax/'

if __name__ == "__main__":
    
    n = 512
    nz = 512
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = True
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 8  # ptychography iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography
    pnz = 1
    nmodes = 4
    ngpus = 1
    nscan = 10000
    center = 268

    id_theta = int(sys.argv[1])
    ntheta = int(sys.argv[2])
    data = np.zeros([ntheta, nscan, det[0], det[1]], dtype='float32')
    scan = np.zeros([2, ntheta, nscan], dtype='float32')-1
    theta = np.zeros(ntheta, dtype='float32')
    
    theta=np.load('theta173.npy')
    idstheta = np.argsort(theta)
    
    for k in range(ntheta):
        ids = sample(range(13689),nscan)
        data0 = np.load(data_prefix+'data/data128_'+str(idstheta[id_theta]+k)+'.npy')        
        scan0 = np.load(data_prefix+'data/scan128_'+str(idstheta[id_theta]+k)+'.npy')        
        data[k] = data0[ids]        
        scan[:,k:k+1,:] = scan0[:,:,ids]
        
    
    shifts = np.load('shifts173.npy')[id_theta]
    scan[0]-=np.round(shifts[1])
    scan[1]-=np.round(shifts[0])
    
    for k in range(ntheta):
        ids = np.where((scan[0,k]>n-1-nprb)+(scan[1,k]>nz-1-nprb)+(scan[0,k]<0)+(scan[1,k]<0))[0]
        scan[0,k,ids]=-1
        scan[1,k,ids]=-1
        data[k,ids] = 0     

    # Load a 3D object
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64',order='C')
    prb[:] = np.load(data_prefix+'data/prb128.npy')
    
    # Initial guess
    psi = np.ones([ntheta, nz, n], dtype='complex64', order='C')*1    
    data = np.fft.fftshift(data, axes=(2, 3))
    # Class gpu solver
    slv = pt.Solver(nscan, theta, center, det, voxelsize,
                    energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus)
    name = data_prefix+'rec'+str(recover_prb)+str(nmodes)+str(scan.shape[2])
    psi, prb = slv.cg_ptycho_batch(
        data/det[0]/det[1], psi, prb, scan, None, -1, piter, model, recover_prb)

    # Save result
    for k in range(ntheta):
        dxchange.write_tiff(np.angle(psi[k]),  'rec/psiangle'+str(nmodes)+str(nscan)+'/r'+str(id_theta+k), overwrite=True)
        dxchange.write_tiff(np.abs(psi[k]),  'rec/psiamp'+str(nmodes)+str(nscan)+'/r'+str(id_theta+k), overwrite=True)
        for m in range(4):
            dxchange.write_tiff(np.angle(prb[:,m]),  'rec/prbangle'+str(k)+'/r'+str(id_theta+k), overwrite=True)
            dxchange.write_tiff(np.abs(prb[:,m]),  'rec/prbamp'+str(k)+'/r'+str(id_theta+k), overwrite=True)
        