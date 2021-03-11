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
    n = 384  # object size n x,y
    nz = 320  # object size in z
    nmodes = 4
    nscan = 13689
    ntheta = 173  # number of angles (rotations)
    shift = np.zeros((ntheta,2),dtype='float32')
    ssum = np.zeros([ntheta,nz],dtype='float32')
    cm = np.zeros(ntheta,dtype='float32')
    data = np.zeros([ntheta,nz,n],dtype='float32')       
    
    for k in range(ntheta):
        data[k] = dxchange.read_tiff(data_prefix+'reccrop/psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')  
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
        shift[k,0] = shift0
        shift[k,1] = cm[1]-n//2

    np.save(data_prefix+'datanpy/shiftssum',shift)


    
    
    
    #exit()
    plt.imshow(ssum)
    plt.savefig(data_prefix+'/png/ssum.png')
    plt.show()

    #     print(shift0)
    #     data[k]=np.roll(data[k],(-int(shift0[0]),-int(shift0[1])),axis=(0,1))
    #     shift[k]=shift0
    dxchange.write_tiff_stack(data,data_prefix+'reccrop_align_sum/psiangle'+str(nmodes)+str(nscan)+'/r.tiff',overwrite=True)                
    # np.save(data_prefix+'/datanpy/shiftscrop.npy',shift)

