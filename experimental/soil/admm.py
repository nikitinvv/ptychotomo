import dxchange
import numpy as np
import sys
import tomoalign
import scipy.ndimage as ndimage
import cupy as cp

centers = {
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T10_22micro_0001': 633,# 1500, 700, 1120
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T5_36micro_0002': 624, # 1500, 700, 1120 
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_33micro_0001': 622, # 1500, 700, 1120 
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T15_31micro_0001':  551, #1500, 700, 1120
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_24macro_0001': 560, # 1500, 700, 1120     
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T10_31macro_0001': 1300,# 1500, 1100, 2560
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T15_38macro': 1300, # 1500, 2160, 2560
    #'/data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T5_28macro_0002': , # 1500, 1100, 1120
}

if __name__ == "__main__":

    fname = sys.argv[1]
    binning = 0
    ndsets = 1
    nth = 1500

    data= np.load(fname+'_bin' +str(binning)+str(0)+'.npy').astype('float32')
    theta = np.load(fname+'_theta' +str(0)+'.npy').astype('float32')
    data[np.isnan(data)] = 0
    data_new = np.zeros([data.shape[0],720,data.shape[2]],dtype='float32')
    data_new[:,10:-10] = data
    data = data_new
    ngpus = 4
    pnz = 4
    ptheta = 10

    print(data.shape)
    
    startwin = [256, 128, 64]
    niteradmm = [96, 48, 24]
    stepwin = [2, 2, 2]
    center = (centers[fname])/pow(2, binning)

    fname += '/dense_new'+'_'+str(binning)

    res = tomoalign.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)

    dxchange.write_tiff_stack(
        res['u'], fname+'/results_admm_new/u/r', overwrite=True)
    dxchange.write_tiff_stack(
        res['psi'], fname+'/results_admm_new/psi/r', overwrite=True)
    np.save(fname+'/results_admm_new/flow.npy', res['flow'])
