import dxchange
import numpy as np
import sys
import tomoalign
import scipy.ndimage as ndimage

centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_721prj_180deg_1s_170': 1211,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167': 1224,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_4800prj_720deg_166': 1224,
}

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    fname = sys.argv[3]

    binning = 1
    data = np.zeros([ndsets*nth, 2048//pow(2, binning),
                     2448//pow(2, binning)], dtype='float32')
    theta = np.zeros(ndsets*nth, dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0
    data -= np.mean(data)
    ngpus = 4
    pnz = 4
    ptheta = 10
    niteradmm = [96, 48, 24]  # number of iterations in the ADMM scheme
    # starting window size in optical flow estimation
    startwin = [256, 128, 64]
    # step for decreasing the window size in optical flow estimtion
    stepwin = [2, 2, 2]
    center = (centers[fname])/pow(2, binning)

    fname += '/revision_noflow'+'_'+str(binning)

    data = np.ascontiguousarray(data.astype('float32'))
    theta = np.ascontiguousarray(theta.astype('float32'))

    #dxchange.write_tiff_stack(
        #data, fname+'/data/d', overwrite=True)
    pprot = 720
    res = tomoalign.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)

    dxchange.write_tiff_stack(
        res['u'], fname+'/results_admm/u/r', overwrite=True)
    dxchange.write_tiff_stack(
        res['psi'], fname+'/results_admm/psi/r', overwrite=True)
    np.save(fname+'/results_admm/flow.npy', res['flow'])
