import dxchange
import numpy as np
import sys
import tomoalign

centers = {
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple1_Phase_1201prj_interlaced_1s_010': 1227,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple2_Phase_1201prj_interlaced_1s_011': 1237,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple3_Phase_1201prj_1s_009': 1217,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple4_Phase_1201prj_interlaced_1s_012': 1183,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple5_Phase_1201prj_interlaced_1s_013': 1202,
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
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0
    data -= np.mean(data)

    ngpus = 4
    pprot = 1200
    nitercg = 64
    pnz = 8
    center = (centers[fname])/pow(2, binning)


    data = np.ascontiguousarray(data)
    theta = np.ascontiguousarray(theta)
    res = tomoalign.pcg(data, theta, pprot, pnz, center, ngpus, nitercg)
    dxchange.write_tiff_stack(
        res['u'], fname+'/results_cg/u/r', overwrite=True)
