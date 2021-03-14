import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from timing import tic,toc
import gc
import scipy.ndimage as ndimage
import skimage.feature
import cv2 
matplotlib.use('Agg')
centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_721prj_180deg_1s_170': 1211,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167': 1224,
}
ngpus = 4
def apply_shift(psi, p):
    """Apply shift for all projections."""
    [nz,n] = psi.shape
    tmp = np.zeros([2*nz, 2*n], dtype='float32')
    tmp[nz//2:3*nz//2, n//2:3*n//2] = psi
    [x,y] = np.meshgrid(np.fft.rfftfreq(2*n),np.fft.fftfreq(2*nz))
    shift = np.exp(-2*np.pi*1j*(x*p[1]+y*p[0]))
    res0 = np.fft.irfft2(shift*np.fft.rfft2(tmp))
    res = res0[nz//2:3*nz//2, n//2:3*n//2]
    return res
        
def find_min_max(data):
    # s = np.std(data,axis=(1,2))    
    # m = np.mean(data,axis=(1,2))
    # mmin = m-2*s
    # mmax = m+2*s
    mmin = np.zeros(data.shape[0],dtype='float32')
    mmax = np.zeros(data.shape[0],dtype='float32')
    
    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:],1000)
        stend = np.where(h>np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]        
        mmin[k] = e[st]
        mmax[k] = e[end+1]
     
    return mmin,mmax

def pad(data,ne,n):
    datae = np.zeros([data.shape[0],nz,ne],dtype='float32')
    datae[:,:,ne//2-n//2:ne//2+n//2]=data
    datae[:,:,:ne//2-n//2]=datae[:,:,ne//2-n//2:ne//2-n//2+1]
    datae[:,:,ne//2+n//2:]=datae[:,:,ne//2+n//2-1:ne//2+n//2]
    return datae

def unpad(data,ne,n):
    return data[:,:,ne//2-n//2:ne//2+n//2]


if __name__ == "__main__":
    binning=2
    [ntheta,nz,n] = [1,1024//pow(2,binning),1024//pow(2,binning)]
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')    
    data=np.zeros([ntheta,nz,n],dtype='float32')
    data[:,256//pow(2,binning):480//pow(2,binning),256//pow(2,binning):480//pow(2,binning)]=1
    psi=np.zeros([ntheta,nz,n],dtype='float32')    
    psi[:,(256+4*8)//pow(2,binning):(480+4*8)//pow(2,binning),(256+4*8)//pow(2,binning):(480+4*8)//pow(2,binning)]=1
    pars = [0.5,1, n, 4, 5, 1.1,4]
       
    with dc.SolverDeform(ntheta, nz, n, ntheta) as dslv:
        for k in range(128):
            flow = dslv.registration_flow_batch(
                    psi,data, np.array([0]), np.array([1]), flow, pars, 1) 
            
            psin = dslv.apply_flow_gpu_batch(psi, flow)               
            print('a',pars[2],np.linalg.norm(data-psin))
            #print(np.min(flow[0]),np.std(flow[0]))
            pars[2]-=4#np.int(pars[2]*pars[0])
            
        plt.subplot(131)
        plt.imshow(dc.flowvis.flow_to_color(flow[0]), cmap='gray')
        #plt.show()
        plt.subplot(132)
        plt.imshow(data[0]-psi[0])
        plt.subplot(133)
        plt.imshow(data[0]-psin[0])
        
        plt.savefig(
        '/data/staff/tomograms/vviknik/tomoalign_vincent_data//flowfw.png')