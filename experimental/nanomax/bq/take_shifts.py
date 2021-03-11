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

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'

if __name__ == "__main__":
    
   
    n = 512-128
    nz = 512-192
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = True
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 256  # ptychography iterations
    nmodes = 4
    ngpus = 1
    nscan = 4000
    
    
    # Load a 3D object
    prb = np.zeros([1, nmodes, nprb, nprb], dtype='complex64',order='C')
    prb[:] = np.load(data_prefix+'datanpy/prb128.npy')

    #scan0 = np.load(data_prefix+'datanpy/scan128sorted_'+str(id_theta)+'.npy')   
    #print(scan0.shape)
    shifts = np.load(data_prefix+'/datanpy/shifts.npy')
    shiftspart = np.load(data_prefix+'/datanpy/shiftscrop.npy')
    shiftssum = np.load(data_prefix+'/datanpy/shiftssum.npy')
    print(shifts)
    plt.figure(figsize=(20,6))    
    sx = shifts[:,0]+shiftspart[:,0]+shiftssum[:,0]
    sy = shifts[:,1]+shiftspart[:,1]+shiftssum[:,1]
    np.save('sx',-sx)
    np.save('sy',-sy)
    plt.plot(sx,'r.')
    # plt.plot(sy,'b.')
    plt.grid()
    plt.savefig(data_prefix+'/shifts')
    
    theta_azat = np.load('nm_theta_sorted.npy',allow_pickle=True)
    shifts_azat =  np.load('nm_align_param_sorted_new.npy',allow_pickle=True)#*41.45/18.03
    # a=np.load('/data/staff/tomograms/vviknik/nanomax/datanpy/shifts.npy')
    # print(a.shape)
    # plt.plot(a[:,0],'r.')
    # plt.plot(a[:,1],'b.')
    # plt.savefig('/data/staff/tomograms/vviknik/nanomax/azat.png')
    plt.clf()
    plt.figure(figsize=(20,8))
    plt.plot(-sx,'r.',label='sift shifts')
    plt.plot(shifts_azat[0],'b.',label='stxm shifts')    
    plt.legend()
    plt.grid()
    plt.savefig('/data/staff/tomograms/vviknik/nanomax/shiftsx.png')

    plt.figure(figsize=(20,8))
    plt.plot(-sy,'r.',label='sift shifts')
    plt.plot(shifts_azat[1],'b.',label='stxm shifts')    
    plt.legend()
    plt.grid()
    plt.savefig('/data/staff/tomograms/vviknik/nanomax/shiftsy.png')
    # print(shifts_azat)
    # exit()
    # #exit()