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
    data = np.zeros([ndsets*nth, 2048//pow(2, binning),
                     2448//pow(2, binning)], dtype='float32')
    theta = np.zeros(ndsets*nth, dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'fw_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0
    # data = data[1200,512:]
    # theta=theta[:1200]
    #print(len(np.unique(theta[:])))
    #exit()
    # # data = data[:, 512:]
    # for k in range(15):
    #     print(theta[k*100:(k+1)*100])
    # exit()

    # #theta=np.sort(theta%(2*np.pi))
    
    # print(np.min(theta[1::2]-theta[0::2]))
    # print(np.max(theta[1::2]-theta[0::2]))

    # exit()

    ngpus = 4
    pprot = 1200
    nitercg = 64
    pnz = 8
    center = centers[fname]/2

    res = tomoalign.pcg(data, theta, pprot, pnz, center, ngpus, nitercg)
    dxchange.write_tiff_stack(
        res['u'], fname+'/results_pcgr_new1200/u/r', overwrite=True)
