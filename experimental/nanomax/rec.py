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
    for k in range(210, 211):
        data0, scan0, theta0 = read_data(k)
        if(scan0 is not None):
            scan[0, kk:kk+1, :scan0.shape[2]] = scan0[1]
            scan[1, kk:kk+1, :scan0.shape[2]] = scan0[0]
            theta[kk] = theta0
            data[kk] = data0[:, 64-data.shape[2]//2:64+data.shape[2] //
                             2, 64-data.shape[3]//2:64+data.shape[3]//2]            
            kk += 1
    dxchange.write_tiff(data,  'data', overwrite=True)
    
    psirec,prbrec,scanrec = read_rec(210)    
    
    # dxchange.write_tiff(np.angle(psirec),  'psirec/psiangle', overwrite=True)
    # dxchange.write_tiff(np.abs(psirec),  'psirec/psiamp', overwrite=True)
    # dxchange.write_tiff_stack(np.angle(prbrec),  'prbrec/prbangle', overwrite=True)
    # dxchange.write_tiff_stack(np.abs(prbrec),  'prbrec/prbamp', overwrite=True)
    # # plt.plot(scan[0,0],scan[1,0],'b.',markersize=3)
    # # plt.plot(scanrec[0,0],scanrec[1,0],'r.',markersize=3)
    # # plt.show()
    # print(scanrec.shape)
    # print(scan[1,0,:10])
    # print(scanrec[0,0,::4])
    # print(np.max(np.abs(scan[0,0,:]-scanrec[1,0,::4])))
    # print(np.max(np.abs(scan[0,0,:]-scanrec[1,0,1::4])))
    # print(np.max(np.abs(scan[0,0,:]-scanrec[1,0,2::4])))
    # print(np.max(np.abs(scan[0,0,:]-scanrec[1,0,3::4])))

    # exit()
    
    n = 512+64
    nz = 512+64
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
    
    # Load a 3D object
    prb = cp.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    # a = cp.array(pt.probesquare(nprb, 1, rin=0.2*32/nprb, rout=0.6*32/nprb))
    prb[:] = cp.array(prbrec[:nmodes])/det[0]/det[1]
    # prb=prb.swapaxes(2,3)
    # data =data.swapaxes(2,3)
    
    dxchange.write_tiff_stack(cp.angle(prb).get(),  'prb/prbangleinit', overwrite=True)
    dxchange.write_tiff_stack(cp.abs(prb).get(),  'prb/prbampinit', overwrite=True)
    
    # Initial guess
    h = cp.ones([ntheta, nz, n], dtype='complex64', order='C')
    psi = cp.ones([ntheta, nz, n], dtype='complex64', order='C')*1
    e = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')
    scan = cp.array(scan[:, :, :])
    data = np.fft.fftshift(data[:, :], axes=(2, 3))
    theta = cp.array(theta)
    # Class gpu solver
    slv = pt.Solver(scan, theta, det, voxelsize,
                    energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes)
    dxchange.write_tiff(np.fft.fftshift(data[0],axes=(1,2)),  'data', overwrite=True)
    data1 = slv.fwd_ptycho_batch(psi,prb,scan)
    data1 = data1/det[0]/det[1]
    dxchange.write_tiff(np.fft.fftshift(data1[0],axes=(1,2)),  'datasim', overwrite=True)
    print("max intensity on the detector: ", np.amax(data1))
    print("sum: ", np.sum(data1[0,0]))
    print("max intensity on the detector: ", np.amax(data))
    print("sum: ", np.sum(data[0,0]))
    
    rho = 0.5
    psi, prb = slv.cg_ptycho_batch(
        data, psi, prb, scan, h, lamd, rho, piter, model, recover_prb)

    # Save result
    dxchange.write_tiff(cp.angle(psi[0]).get(),  'psig2/psiangle'+str(nmodes)+'_', overwrite=True)
    dxchange.write_tiff(cp.abs(psi[0]).get(),  'psig2/psiamp'+str(nmodes)+'_', overwrite=True)
    dxchange.write_tiff_stack(cp.angle(prb[0]).get(),  'prbg2/prbangle'+str(nmodes)+'_', overwrite=True)
    dxchange.write_tiff_stack(cp.abs(prb[0]).get(),  'prbg2/prbamp'+str(nmodes)+'_', overwrite=True)
