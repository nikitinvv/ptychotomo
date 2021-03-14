import dxchange
import numpy as np
import sys
import tomoalign

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    fname = sys.argv[3]
    center = np.float(sys.argv[4])

    binning = 0
    data = np.zeros([ndsets*nth, 1,
                     1120//pow(2, binning)], dtype='float32')
    theta = np.zeros(ndsets*nth, dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')[:,350//pow(2,binning):350//pow(2,binning)+1]
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0
    #data -= np.mean(data)

    ngpus = 1
    pprot = nth*ndsets
    nitercg = 32
    pnz = 1    
    data = np.ascontiguousarray(data)
    theta = np.ascontiguousarray(theta)
    res = tomoalign.pcg(data, theta, pprot, pnz, center/pow(2,binning), ngpus, nitercg)
    dxchange.write_tiff(
        res['u'][0], fname+'/results_cg_centers/u/r'+str(center), overwrite=True)
