import dxchange
import numpy as np
import sys
import tomoalign
import scipy.ndimage as ndimage
centers = {
    '/local/data/vnikitin/Kenan_ZP_8keV_interlaced_5000prj_3s_001': 1200,#1277,
    '/local/data/vnikitin/Kenan_ZP_ROI2_8keV_interlaced_5000prj_3s_002': 1250,
    '/local/data/vnikitin/Kenan_ZP_ROI3_8keV_interlaced_5000prj_2s_003': 1200,
}

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    fname = sys.argv[3]
    part = int(sys.argv[4])
    binning = 0
    data = np.zeros([ndsets*nth, 2048//pow(2, binning),
                     2448//pow(2, binning)], dtype='float32')
    theta = np.zeros(ndsets*nth, dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0    
    data=data[:,256:-256]
    # w = np.linspace(0,np.pi/2,256)
    # data[:,:,:256]*=np.sin(w)
    # data[:,:,-256:]*=np.cos(w)
    dxchange.write_tiff_stack(data,fname+'/data/d')
    
    #data-=np.mean(data)
    ngpus = 8
    pnz = 4
    ptheta = 10
    niteradmm = [96,48,24]  # number of iterations in the ADMM scheme
    # startwin = [512,256,128]
    # niteradmm = [2,2,2]  # number of iterations in the ADMM scheme
    startwin = [512,256,128]
    
    # startwin = [512]
    # step for decreasing the window size in optical flow estimtion
    stepwin = [4,4,4]
    center = (centers[fname])/pow(2, binning)




    fname += '/dense'+'_'+str(binning)+'01'#str(part)
    
    # data = np.ascontiguousarray(data[part::2].astype('float32'))
    # theta = np.ascontiguousarray(theta[part::2].astype('float32'))
    
    res = tomoalign.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)
    
    dxchange.write_tiff_stack(
        res['u'].swapaxes(0,1), fname+'/results_admm/u/r', overwrite=True)
    dxchange.write_tiff_stack(
        res['psi'], fname+'/results_admm/psi/r', overwrite=True)
    np.save(fname+'/results_admm/flow.npy', res['flow'])
