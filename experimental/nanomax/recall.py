import os
import signal
import sys
import h5py
import cupy as cp
import dxchange
import numpy as np
import matplotlib.pyplot as plt
import ptychotomo as pt


def read_data(id_data):
    try:
        h5file = h5py.File(
            '/local/data/nanomax/files/scan_000'+str(id_data)+'.h5', 'r')
        data = h5file['measured/diffraction_patterns'][:].astype('float32')
        positions = h5file['measured/positions_um'][:].astype('float32')
        mask = h5file['measured/mask'][:].astype('float32')
        data *= mask
        theta = h5file['measured/angle_deg'][()].astype('float32')/180*np.pi
        scan = ((-positions)*1e3+4.02*1e3)/18.03
        scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
    except:
        scan = None
        theta = None
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
    print(psi.shape, data.dtype)
    print(probe.shape, probe.dtype)
    return psi, probe, scan

if __name__ == "__main__":
    kk = 0
    data = np.zeros([1, 13689, 128, 128], dtype='float32')-1
    scan = np.zeros([2, 1, 13689], dtype='float32')-1
    theta = np.zeros(1, dtype='float32')
    ids = int(sys.argv[2])
    for k in range(ids, ids+1):
        data0, scan0, theta0 = read_data(k)
        if(scan0 is not None):
            scan[0, kk:kk+1, :scan0.shape[2]] = scan0[1]
            scan[1, kk:kk+1, :scan0.shape[2]] = scan0[0]
            theta[kk] = theta0
            data[kk] = data0[:, 64-data.shape[2]//2:64+data.shape[2] //
                             2, 64-data.shape[3]//2:64+data.shape[3]//2]            
            kk += 1
        else:
            exit()
    psirec,prbrec,scanrec = read_rec(210)    
    
    n = 512+128+64
    nz = 512+128+64
    det = [128, 128]
    ntheta = 1  # number of angles (rotations)
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
    ptheta = ntheta  # number of angular partitions for simultaneous processing in ptychography
    pnz = 64  # number of slice partitions for simultaneous processing in tomography
    nmodes = int(sys.argv[1])

    data = data/det[0]/det[1]
    
    prb = cp.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb[:] = cp.array(prbrec[:nmodes])/det[0]/det[1]
        
    # Initial guess
    h = cp.ones([ntheta, nz, n], dtype='complex64', order='C')
    psi = cp.ones([ntheta, nz, n], dtype='complex64', order='C')*1
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')
    scan = cp.array(scan[:, :, :])
    data = np.fft.fftshift(data[:, :], axes=(2, 3))
    theta = cp.array(theta)
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes)
    print('id', ids)
    print("max intensity on the detector: ", np.amax(data))
    
    rho = 0.5
    psi, prb = slv.cg_ptycho_batch(
        data, psi, prb, scan, h, lamd, rho, piter, model, recover_prb)

    # Save result
    dxchange.write_tiff(cp.angle(psi[0]).get(),  'psiall/psiangle'+str(nmodes)+'_'+str(ids), overwrite=True)
    dxchange.write_tiff(cp.abs(psi[0]).get(),  'psiall/psiamp'+str(nmodes)+'_'+str(ids), overwrite=True)
    dxchange.write_tiff_stack(cp.angle(prb[0]).get(),  'prball/prbangle'+str(nmodes)+'_'+str(ids), overwrite=True)
    dxchange.write_tiff_stack(cp.abs(prb[0]).get(),  'prball/prbamp'+str(nmodes)+'_'+str(ids), overwrite=True)
