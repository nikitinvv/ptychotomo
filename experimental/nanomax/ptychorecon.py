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
    # data = np.zeros([ntheta, nscan, 128, 128], dtype='float32')-1
    # scan = np.zeros([2, ntheta, nscan], dtype='float32')-1
    # theta = np.zeros(ntheta, dtype='float32')
    # kk = 0
    # for k in range(134, 424, 1):
    #     print(k)
    #     data0, scan0, theta0 = read_data(k)
    #     if(scan0 is not None):
    #         ids = np.array(sample(range(13689),nscan))+(13689-13689)  
    #         scan[0, kk, :] = scan0[1,0,ids]
    #         scan[1, kk, :] = scan0[0,0,ids]
    #         theta[kk] = theta0
    #         data[kk] = data0[ids, 64-data.shape[2]//2:64+data.shape[2] //
    #                          2, 64-data.shape[3]//2:64+data.shape[3]//2]            
    #         print(kk,scan[:,kk].max(), scan[:,kk].min())   
    #         kk += 1
            
    # # ids = np.argsort(theta)
    # # theta = theta[ids]
    # # scan = scan[:,ids]
    # # data = data[ids]
    # np.save('theta',theta)
    # np.save('scan',scan)
    # np.save('data',data)
    # exit()
    theta = np.load('theta.npy')
    scan = np.load('scan.npy')[:,:ntheta,::2]
    # print(scan.max())   
    # print(scan.min())   
    # exit()
    data = np.load('data.npy')[:ntheta,::2]#,32:96,32:96]

    psirec,prbrec,scanrec = read_rec(210)    
    # prbrec = prbrec[:,32:96,32:96]
    n = 512+64
    nz = 512+64
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = True
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 128  # ptychography iterations
    titer = 4  # tomography iterations
    niter = 128  # ADMM iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography
    pnz = 32  # number of slice partitions for simultaneous processing in tomography
    nmodes = int(sys.argv[1])    
    
    # Load a 3D object
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64',order='C')
    prb[:] = np.array(prbrec[:nmodes])/det[0]    
    
    # dxchange.write_tiff(data,  'data', overwrite=True)
    # dxchange.write_tiff_stack(np.angle(prb),  'tmp/prb/prbangleinit', overwrite=True)
    # dxchange.write_tiff_stack(np.abs(prb),  'tmp/prb/prbampinit', overwrite=True)
    # dxchange.write_tiff_stack(np.angle(psirec),  'tmp/psi/psiangleinit', overwrite=True)
    # dxchange.write_tiff_stack(np.abs(psirec),  'tmp/psi/psiampinit', overwrite=True)
    # # exit()
    # Initial guess
    h = np.ones([ntheta, nz, n], dtype='complex64', order='C')
    psi = np.ones([ntheta, nz, n], dtype='complex64', order='C')*1    
    e = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = np.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = np.zeros([nz, n, n], dtype='complex64', order='C')
    data = np.fft.fftshift(data, axes=(2, 3))
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes)
    rho = None
    psi, prb = slv.cg_ptycho_batch(
        data/det[0]/det[1], psi, prb, scan, h, lamd, rho, piter, model, recover_prb)

    # Save result
    dxchange.write_tiff_stack(np.angle(psi),  'tmp/psiangle'+str(nmodes)+'_', overwrite=True)
    dxchange.write_tiff_stack(np.abs(psi),  'tmp/psiamp'+str(nmodes)+'_', overwrite=True)
    dxchange.write_tiff_stack(np.angle(prb[0]),  'tmp/prbangle'+str(nmodes)+'_', overwrite=True)
    dxchange.write_tiff_stack(np.abs(prb[0]),  'tmp/prbamp'+str(nmodes)+'_', overwrite=True)
