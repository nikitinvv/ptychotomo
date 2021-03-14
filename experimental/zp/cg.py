import dxchange
import numpy as np
import sys
import tomoalign
centers = {
    '/local/data/vnikitin/Kenan_ZP_8keV_interlaced_5000prj_3s_001': 1200,#1277,
    '/local/data/vnikitin/Kenan_ZP_ROI3_8keV_interlaced_5000prj_2s_003': 1200,
}
if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    fname = sys.argv[3]
    binning = int(sys.argv[4])
    part = int(sys.argv[5])
    data = np.zeros([ndsets*nth, 2048//pow(2, binning),
                     2448//pow(2, binning)], dtype='float32')
    theta = np.zeros(ndsets*nth, dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0    
    theta = theta[part::2]
    data = data[part::2]
    # data=data[:,256//pow(2,binning):-256//pow(2,binning),456//pow(2,binning):-456//pow(2,binning)]
    #ids = np.where((theta>-np.pi*mrange/180.0)*(theta<=np.pi*mrange/180.0))
    #theta = theta[ids]
    #data = data[ids]

    
    ngpus = 4
    pprot = len(theta)
    nitercg = 64
    pnz = 4
    center = centers[fname]/pow(2,binning)

    data = np.ascontiguousarray(data)
    theta = np.ascontiguousarray(theta)
    res = tomoalign.pcg(data, theta, pprot, pnz, center, ngpus, nitercg)
    dxchange.write_tiff_stack(
        res['u'].swapaxes(0,1), fname+'/results_pcg'+str(part)+'_'+str(nitercg)+str(data.shape[2])+'/u/r', overwrite=True)
