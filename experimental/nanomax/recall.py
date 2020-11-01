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

def read_data(id_data):
    try:
        h5file = h5py.File(
            '/local/data/nanomax/files/scan_000'+str(id_data)+'.h5', 'r')
        data = h5file['measured/diffraction_patterns'][:].astype('float32')
        positions = h5file['measured/positions_um'][:].astype('float32')
        mask = h5file['measured/mask'][:].astype('float32')
        data *= mask
        data = np.roll(data,(2,1),axis=(1,2))    
        theta = h5file['measured/angle_deg'][()].astype('float32')/180*np.pi
        scan = ((-positions)*1e3+4.02*1e3)/18.03
        scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
    except:
        scan = None
        theta = None
        data = None
    return data, scan, theta

def read_rec(id_data):
    h5file = h5py.File(
            '/local/data/nanomax/files/scan'+str(id_data)+'_DM_1000.ptyr', 'r')
    psi = h5file['content/obj/Sscan00G00/data'][:]
    probe = h5file['content/probe/Sscan00G00/data'][:]
    positions = h5file['content/positions/Sscan00G00'][:]
    
    scan = ((positions)*1e9+4.02*1e3)/18.03    
    scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
        
    return psi, probe, scan

if __name__ == "__main__":
    
    ntheta = 166
    nscan = 2048
    theta = np.load('theta.npy')
    scan = np.load('scan.npy')[:,:ntheta,::2]
    data = np.load('data.npy')[:ntheta,::2]#,32:96,32:96]

    psirec,prbrec,scanrec = read_rec(210)    
    n = 512+64
    nz = 512+64
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = True
    align = True
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    diter = 4
    niter = 256  # ADMM iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography
    pnz = 4  # number of slice partitions for simultaneous processing in tomography
    nmodes = int(sys.argv[1])    
    
    # Load a 3D object
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64',order='C')
    prb[:] = np.array(prbrec[:nmodes])/det[0]    
    

    psiangle = dxchange.read_tiff_stack('tmp/psiangle'+str(nmodes)+'__00000.tiff',ind = np.arange(ntheta))
    psiamp = dxchange.read_tiff_stack('tmp/psiamp'+str(nmodes)+'__00000.tiff',ind = np.arange(ntheta))

#ALIGN to com 
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, len(theta), nz, n, nprb, ptheta, pnz, nmodes)
    psi1 = psiamp*np.exp(1j*psiangle)
    lpsi1 = slv.logtomo(psi1)
    dxchange.write_tiff_stack(lpsi1.real,'lspci/psi', overwrite=True)                
    for k in range(ntheta):        
        a = lpsi1[k].real
        a[a<7e-5]=0        
        cm = ndimage.center_of_mass(a)   
        print(cm)                          
        scan[0,k]-=np.round(cm[1]-n//2+0.5)
        scan[1,k]-=np.round(cm[0]-nz//2+0.5)
        ids = np.where((scan[0,k]>n-1-nprb)+(scan[1,k]>nz-1-nprb)+(scan[0,k]<0)+(scan[1,k]<0))[0]
        scan[0,k,ids]=-1
        scan[1,k,ids]=-1
        data[k,ids] = 0 #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        lpsi1[k] = np.roll(lpsi1[k],(-int(cm[0]-nz//2+0.5),-int(cm[1]-n//2+0.5)),axis=(0,1))      
    print(np.min(scan))
    print(np.max(scan))
    dxchange.write_tiff_stack(lpsi1.real,'clspci/psi', overwrite=True)          

    # Initial guess
    h1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j    
    h2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    h3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j


    psi1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j    
    psi2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    psi3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')+1+1e-10j
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    lamd2 = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd3 = np.zeros([ntheta, nz, n], dtype='complex64', order='C')    
    u = np.zeros([nz, n, n], dtype='complex64', order='C')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32', order='C')


    data = np.fft.fftshift(data, axes=(2, 3))
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes)
    name = 'rec'+str(recover_prb)+str(align)+str(nmodes)+str(scan.shape[2])
    
    # ADMM
    u, psi3, psi2, psi1, flow, prb = slv.admm(
        data, psi3, psi2, psi1, flow, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, u, alpha, piter, titer, diter, niter, model, recover_prb, align, name)

