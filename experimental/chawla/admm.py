import dxchange
import numpy as np
import sys
import tomoalign
centers = {
    '/data/staff/tomograms/vviknik/chawla/Cu_spheres_16nmZP_8keV_interlaced_1500prj_2s_004': 1224,
    '/data/staff/tomograms/vviknik/chawla/C2_roomT_8200eV_Interlaced_1201prj_145': 1190,
    '/data/staff/tomograms/vviknik/chawla/C2_roomT_2nd_450C_8200eV_Interlaced_1201prj_030': 1224,
    '/data/staff/tomograms/vviknik/chawla/C2_2nd_500C_8200eV_Interlaced_1201prj_070': 1217,
    '/data/staff/tomograms/vviknik/chawla/C2_2nd_500C_8200eV_Interlaced_1201prj_080': 1213,
}

ngpus = 4

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    fname = sys.argv[3]   
    
    binning = 1
    data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'fw_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta'+str(k)+'.npy').astype('float32')
    data[np.isnan(data)]=0           
    # data = data[:,512:]
    ngpus = 4
    pnz = 16
    ptheta = 20
    pprot = 200
    niteradmm = [96, 48, 24]  # number of iterations in the ADMM scheme
    # starting window size in optical flow estimation
    startwin = [256, 128, 64]
    # step for decreasing the window size in optical flow estimtion
    stepwin = [2, 2, 2]
    center = centers[fname]/pow(2,binning)    
    res = tomoalign.admm_of_levels_p(data, theta, pprot, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)
    dxchange.write_tiff_stack(res['u'], fname+'/results_admm_prealigned/u/r', overwrite=True)
    