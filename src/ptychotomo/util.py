from scipy import ndimage
import numpy as np
import time

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
       return time.time() - startTime_for_tictoc
       
def find_min_max(data):
    """Find min and max values according to histogram"""
    
    mmin = np.zeros(data.shape[0],dtype='float32')
    mmax = np.zeros(data.shape[0],dtype='float32')
    
    for k in range(data.shape[0]):
        h, e = np.histogram(np.angle(data[k][:]),1000)
        stend = np.where(h>np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]        
        mmin[k] = e[st]
        mmax[k] = e[end+1]
     
    return mmin,mmax


def find_mass_center_shifts(psi):
    shifts = np.zeros([psi.shape[0],2],dtype='float32')
    c = np.array([psi.shape[1]//2,psi.shape[2]//2])
    for k in range(psi.shape[0]):
        cm = ndimage.measurements.center_of_mass(np.abs(np.angle(psi[k])))
        import dxchange
        dxchange.write_tiff(np.abs(np.angle(psi[k])),'p/p')
        shifts[k] = cm-c
    return shifts        