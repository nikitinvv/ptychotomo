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


data_prefix = '/local/data/vnikitin/nanomax/'
def read_data(id_data):
    try:
        h5file = h5py.File(
            data_prefix+'/scan_000'+str(id_data)+'.h5', 'r')
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
            data_prefix+'/scan'+str(id_data)+'_DM_1000.ptyr', 'r')
    psi = h5file['content/obj/Sscan00G00/data'][:]
    probe = h5file['content/probe/Sscan00G00/data'][:]
    positions = h5file['content/positions/Sscan00G00'][:]
    
    scan = ((positions)*1e9+4.02*1e3)/18.03    
    scan = scan[np.newaxis, ...].astype(
            'float32').swapaxes(0, 1).swapaxes(0, 2)
        
    return psi, probe, scan


data_prefix = '/local/data/vnikitin/nanomax/'
if __name__ == "__main__":
   
    kk = 0
    for k in range(134, 450, 1):
        print(kk, k)
        data0, scan0, theta0 = read_data(k)
        if(scan0 is not None):
            #ids = np.array(sample(range(13689),nscan))+(13689-13689)  
            scan = scan0*0
            scan[0, :] = scan0[1,0,:]
            scan[1, :] = scan0[0,0,:]
            theta = theta0
            data = data0
            # print(k,scan.max(), scan.min())   
            np.save(data_prefix+'data/theta128_'+str(kk),theta)
            np.save(data_prefix+'data/scan128_'+str(kk),scan)
            np.save(data_prefix+'data/data128_'+str(kk),data0)
            kk += 1
    # psirec,prbrec,scanrec = read_rec(210)    
    # # Load a 3D object
    # prb = prbrec/prbrec.shape[2]             
    # np.save(data_prefix+'data/prb128',prb)
    exit()            
            
    