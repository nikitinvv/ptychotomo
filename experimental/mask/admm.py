import dxchange
import numpy as np
import sys
import tomoalign
centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run1_8keV_phase_interlaced_100prj_per_rot_1201prj_1s_006': 1204,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run21_40min_8keV_phase_interlaced_1201prj_1s_012': 1187,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_PBI_new_ROI_8keV_phase_interlaced_2000prj_1s_042':  1250,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_ROI2_8keV_phase_interlaced_1201prj_0.5s_037':  1227,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_PBI_new_ROI_8keV_phase_interlaced_1201prj_0.5s_041': 1248,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024': 1202,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_PAN_8keV_phase_721prj_0.5s_045': 1244,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_PAN_8keV_phase_interlaced_1201prj_1s_046': 1241,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_3h_8keV_phase_interlaced_1201prj_0.5s_047_049': 1226,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_3h_ROI2_8keV_phase_interlaced_1201prj_0.5s_047_050': 1209,
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
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta'+str(k)+'.npy').astype('float32')
    data[np.isnan(data)]=0           
    data-=np.mean(data)
    ngpus = 4
    pnz = 64
    ptheta = 14
    niteradmm = [96, 48, 25]  # number of iterations in the ADMM scheme
    startwin = [256, 512, 1024]  # starting window size in optical flow estimation
    # step for decreasing the window size in optical flow estimtion
    stepwin = [0, 0, 0]
    center = centers[fname]/pow(2,binning)    
    
  
    data0 = data.copy()
    data = np.ascontiguousarray(data0[0::2])
    theta = np.ascontiguousarray(theta[0::2])
    fname+='nondense_r1'    
    res = tomoalign.admm_of_levels(data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)
    dxchange.write_tiff_stack(res['u'], fname+'/results_admm/u/r', overwrite=True)
    dxchange.write_tiff_stack(res['psi'], fname+'/results_admm/psi/r', overwrite=True)
    np.save(fname+'/results_admm/flow.npy',res['flow'])
        

    # data = data[0::2]    
    # fname+='nondense_r1'    
    # res = tomoalign.admm_of_levels(data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)
    # dxchange.write_tiff_stack(res['u'], fname+'/results_admm/u/r', overwrite=True)
    # dxchange.write_tiff_stack(res['psi'], fname+'/results_admm/psi/r', overwrite=True)
    # np.save(fname+'/results_admm/flow.npy',res['flow'])