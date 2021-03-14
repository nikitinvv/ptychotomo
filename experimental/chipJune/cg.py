import dxchange
import numpy as np
import sys
import tomoalign


if __name__ == "__main__":

    nsets = np.int(sys.argv[1])
    ndsets = np.int(sys.argv[2])
    nth = np.int(sys.argv[3])
    start = np.int(sys.argv[4])
    name = sys.argv[5]

    binning = 0
    data = np.zeros([nsets*ndsets*nth, (2048-512)//pow(2, binning),
                     (2448-400)//pow(2, binning)], dtype='float32')
    theta = np.zeros(nsets*ndsets*nth, dtype='float32')
    strs = ['098','099','100']
    for j in range(nsets):                
        name0 = name[:-3]+strs[j]#name[:-2]+str(np.int(name[-2:])+j)
        print(name0)
        for k in range(ndsets):
            print(j,k)
            idstart = j*ndsets*nth+k*nth
            data[idstart:idstart+nth] = np.load(name0+'ti_bin'+str(binning)+str(k)+'.npy')[:,256:-256,200:-200].astype('float32')                                   
            theta[idstart:idstart+nth] = np.load(name0+'_theta'+str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0
    data = np.ascontiguousarray(data[start::2])
    theta = np.ascontiguousarray(theta[start::2])
    ngpus = 4
    pnz = 8
    nitercg = 32
    center = 1256-200
    res = tomoalign.cg(data, theta, pnz, center, ngpus, nitercg, padding=True)
    name+='/'+str(len(theta))
    dxchange.write_tiff_stack(res['u'], name+'/results_cg'+str(start)+'/u/r', overwrite=True)
    
    
            