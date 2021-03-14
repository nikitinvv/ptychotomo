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
    print(data.shape)
    f = open(fname+'/bin/data.bin', 'w+b')
    binary_format = bytearray(data)
    f.write(binary_format)
    f.close()
    f = open(fname+'/bin/theta.bin', 'w+b')
    binary_format = bytearray(theta)
    f.write(binary_format)
    f.close()