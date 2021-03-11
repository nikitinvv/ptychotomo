import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign
from tomoalign.utils import find_min_max
from skimage.registration import phase_cross_correlation

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    # Model parameters
    n = 512  # object size n x,y
    nz = 512  # object size in z
    nmodes = 4
    nscan = 13689
    ntheta = 174  # number of angles (rotations)
    shift = np.zeros((ntheta,2),dtype='float32')
    ssum = np.zeros([ntheta,nz],dtype='float32')
    cm = np.zeros(ntheta,dtype='float32')
    data = np.zeros([ntheta,nz,n],dtype='float32')       
    
    data = dxchange.read_tiff_stack(data_prefix+'recfull_sorted/psiangle'+str(nmodes)+str(nscan)+'/r_00000.tiff', ind=np.arange(0,173))  
    for k in range(ntheta):        
        data0 = data[k].copy()
        data0[data0<0] = 0
        ssum[k] = np.sum(data0,axis=1)
    for k in range(0,ntheta):
        shift0, error, diffphase = phase_cross_correlation(ssum[k], ssum[0], upsample_factor=1)
        print(shift0)
        ssum[k] = np.roll(ssum[k],-int(shift0))        
        data[k] = np.roll(data[k],-int(shift0),axis=0)    
        data0 = data[k].copy()
        data0[data0<0] = 0
        cm = ndimage.measurements.center_of_mass(data0)
        data[k] = np.roll(data[k],-int(cm[1]-n//2),axis=1)    
        print(cm)



    
    
    
    #exit()
    plt.imshow(ssum)
    plt.savefig(data_prefix+'/png/ssum.png')
    plt.show()

    #     print(shift0)
    #     data[k]=np.roll(data[k],(-int(shift0[0]),-int(shift0[1])),axis=(0,1))
    #     shift[k]=shift0
    dxchange.write_tiff_stack(data,data_prefix+'rec_align_sum/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    # np.save(data_prefix+'/datanpy/shiftscrop.npy',shift)

