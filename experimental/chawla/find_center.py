import dxchange
import numpy as np
import sys
import tomoalign
import matplotlib.pyplot as plt
if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    fname = sys.argv[3]
    center = np.float(sys.argv[4])

    binning = 1
    data = np.zeros([ndsets*nth, 1,
                     2448//pow(2, binning)], dtype='float32')
    theta = np.zeros(ndsets*nth, dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'fw_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')[:,1224//pow(2,binning):1224//pow(2,binning)+1]
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')    
    data[np.isnan(data)] = 0
    data=data[:400]
    theta=theta[:400]
    ngpus = 1
    pprot = data.shape[0]
    nitercg = 32
    pnz = 1    
    res = tomoalign.pcg(data, theta, pprot, pnz, center/pow(2,binning), ngpus, nitercg)
    dxchange.write_tiff(
        res['u'][0], fname+'/results_cg_centersp'+str(data.shape[0])+'/u/r'+str(center), overwrite=True)
