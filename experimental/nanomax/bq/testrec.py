import os
import signal
import sys
import h5py
import cupy as np
import dxchange
import numpy as np
import matplotlib.pyplot as plt
import ptychotomo as pt
from skimage.registration import phase_cross_correlation

def read_data(id_data):
    try:
        h5file = h5py.File(
            'scan_000'+str(id_data)+'.h5', 'r')
        data = h5file['measured/diffraction_patterns'][:].astype('float32')
        positions = h5file['measured/positions_um'][:].astype('float32')
        mask = h5file['measured/mask'][:].astype('float32')
        data *= mask
        theta = h5file['measured/angle_deg'][()].astype('float32')/180*np.pi
        positions[:,0]+=(128/2)*1e-3*18.03    
        positions[:,1]+=(128/2)*1e-3*18.03    
        positions[:,0]+=(-5.156591848745634)
        positions[:,1]+=(-4.651750688728579)
        scan = ((-positions)*1e3)/18.03                
        scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
    except:
        scan = None
        theta = None
    return data, scan, theta

def read_rec(id_data):
    h5file = h5py.File(
            'scan'+str(id_data)+'_DM_1000.ptyr', 'r')
    psi = h5file['content/obj/Sscan00G00/data'][:]
    probe = h5file['content/probe/Sscan00G00/data'][:]
    positions = h5file['content/positions/Sscan00G00'][::4]
    grids = h5file['content/obj/Sscan00G00/grids'][:]
    #print(positions.shape)
    positions[:,0]-=(128/2)*1e-9*18.03    
    positions[:,1]-=(128/2)*1e-9*18.03        
    print(np.min(grids[0]))
    print(np.min(grids[1]))
    positions[:,0]-=np.min(grids[0])
    positions[:,1]-=np.min(grids[1])
    scan = ((positions)*1e9)/18.03    
    scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
    return psi, probe, scan

kk = 0
data = np.zeros([1, 13689, 128, 128], dtype='float32')-1
scan = np.zeros([2, 1, 13689], dtype='float32')-1
theta = np.zeros(1, dtype='float32')
for k in range(210, 211):
    data0, scan0, theta0 = read_data(k)
    psirec,prbrec,scanrec = read_rec(210)    
    print(np.max((scanrec[:,:,:]-scan0[:,:,:])))
    print(scanrec[:,:,:4])
    print(scan0[:,:,:4])
    print(scan0[:,:,0])
    if(scan0 is not None):
        scan[0, kk:kk+1, :scan0.shape[2]] = scan0[1]
        scan[1, kk:kk+1, :scan0.shape[2]] = scan0[0]
        theta[kk] = theta0
        data[kk] = data0[:, 64-data.shape[2]//2:64+data.shape[2] //
                            2, 64-data.shape[3]//2:64+data.shape[3]//2]            
        kk += 1
dxchange.write_tiff(data,  'dataall', overwrite=True)
psirec,prbrec,scanrec = read_rec(210)    

# print(np.linalg.norm(scan[0,0,:]-scanrec[1,0,:]))
# print(np.linalg.norm(scan[1,0,:]-scanrec[0,0,:]))
# exit()
#print(np.min(scan[0]))
#print(np.min(scan[1]))
#print(np.max(scan[0])+128)
#print(np.max(scan[1])+128)

n = 517
nz = 572
nscan = 1
ngpus = 1
det = [128, 128]
center = 256
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
idscan = 5346

data = data/det[0]/det[1]

# Load a 3D object
prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
prbrec = prbrec[:nmodes]
prb[:] = np.array(prbrec)/det[0]



psi = np.array(psirec)
scan = np.array(scan[:, :, idscan:idscan+1])
data = np.fft.fftshift(data[:, idscan:idscan+1], axes=(2, 3))

dxchange.write_tiff_stack(np.angle(prb),  'prb/prbangleinit', overwrite=True)
dxchange.write_tiff_stack(np.abs(prb),  'prb/prbampinit', overwrite=True)
dxchange.write_tiff_stack(np.angle(psi),  'psi/psiangleinit', overwrite=True)
dxchange.write_tiff_stack(np.abs(psi),  'psi/psiampinit', overwrite=True)# scan = np.floor(scan)


psiamp = dxchange.read_tiff('rec/psiamp413689/r15.tiff')
psiangle = dxchange.read_tiff('rec/psiangle413689/r15.tiff')
psi[:] = psiamp*np.exp(1j*psiangle)


data = np.roll(data,(2,1),axis=(2,3))
# Class gpu solver
slv = pt.Solver(nscan, theta, center, det, voxelsize,
                energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus)
dxchange.write_tiff(np.fft.fftshift(data[0],axes=(1,2)),  'data', overwrite=True)
data1 = slv.fwd_ptycho_batch(psi,prb,scan)
data1 = data1/det[0]/det[1]
res = np.zeros([3,nscan, 128, 128],dtype='float32')
res[0] = np.fft.fftshift(data[0],axes=(1,2))
res[1] = np.fft.fftshift(data1[0],axes=(1,2))
res[2] = np.fft.fftshift(data1[0]-data[0],axes=(1,2))
dxchange.write_tiff(np.fft.fftshift(data[0],axes=(1,2)),  'data', overwrite=True)
dxchange.write_tiff(np.fft.fftshift(data1[0],axes=(1,2)),  'datasim', overwrite=True)
dxchange.write_tiff(np.fft.fftshift(data1[0]-data[0],axes=(1,2)),  'datasimdif', overwrite=True)
dxchange.write_tiff(res,  'res', overwrite=True)
print(np.linalg.norm(np.sqrt(data1)-np.sqrt(data)))
# shift, error, diffphase = phase_cross_correlation(data1[0,0], data[0,0], upsample_factor=1)
# print(shift)
# # print("max intensity on the detector: ", np.amax(data1))
# # print("sum: ", np.sum(data1[0,0]))
# print("max intensity on the detector: ", np.amax(data)*det[0]*det[1])
# print("max intensity on the detector: ", np.amax(np.abs(prb))*det[0])


