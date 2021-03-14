import dxchange
import numpy as np
import sys
import tomoalign
centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167': 1224,
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