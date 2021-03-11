import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign
from tomoalign.utils import find_min_max
from skimage.feature import register_translation

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512-128
    nz = 512-192
    nmodes = 4
    nscan = 13689
    ntheta = 174  # number of angles (rotations)
    shift = np.zeros((ntheta,2),dtype='float32')
    ssum = np.zeros([ntheta,nz],dtype='float32')
    cm = np.zeros(ntheta,dtype='float32')
    psiangle = np.zeros([ntheta,nz,n],dtype='float32')       
    psiamp = np.zeros([ntheta,nz,n],dtype='float32')       

    
    for k in range(ntheta):        
        psiangle[k] = dxchange.read_tiff(data_prefix+'rec_crop/psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')  
        psiamp[k] = dxchange.read_tiff(data_prefix+'rec_crop/psiamp'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')  
        psiangle0 = psiangle[k].copy()
        psiangle0[psiangle0<0] = 0
        ssum[k] = np.sum(psiangle0,axis=1)
    for k in range(0,ntheta):
        shift0, error, diffphase = register_translation(ssum[k], ssum[0], upsample_factor=1)
        print(shift0)
        ssum[k] = np.roll(ssum[k],-int(shift0))        
        psiangle[k] = np.roll(psiangle[k],-int(shift0),axis=0)    
        psiangle0 = psiangle[k].copy()
        psiangle0[psiangle0<0] = 0
        cm = ndimage.measurements.center_of_mass(psiangle0)
        psiangle[k] = np.roll(psiangle[k],-int(cm[1]-n//2),axis=1)    
        psiamp[k] = np.roll(psiamp[k],-int(cm[1]-n//2),axis=1)    
        print(cm)
        shift[k,0] = shift0
        shift[k,1] = cm[1]-n//2

    np.save(data_prefix+'datanpy/shiftssum',shift)        
    dxchange.write_tiff_stack(psiangle,data_prefix+'rec_crop_sum_check/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    dxchange.write_tiff_stack(psiamp,data_prefix+'rec_crop_sum_check/psiamp'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    
