import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import ptychotomo as pt

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'
if __name__ == "__main__":

    n = 512-128
    nz = 512-192
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    titer = 32
    nmodes = 4
    ngpus = 1
    nscan = 13689
    ntheta = 174
    
    theta = np.zeros([ntheta],dtype='float32')
    
    psi = np.zeros([ntheta, nz, n], dtype='complex64', order='C')
    #psiangle0 = dxchange.read_tiff_stack(data_prefix+'reccrop_align_sum/psiangle'+str(nmodes)+str(nscan)+'/r_00000.tiff', ind=np.arange(0,173))
    for k in range(ntheta):
        psiangle = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiangle'+str(nmodes)+str(nscan)+'/r_%05d.tiff' % k)
        psiamp = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiamp'+str(nmodes)+str(nscan)+'/r_%05d.tiff' % k)
        print(np.linalg.norm(psiamp))
        # psiangle = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiangle'+str(nmodes)+str(nscan)+'/r%d.tiff' % k)
        # psiamp = dxchange.read_tiff(data_prefix+'rec_crop_sum_check/psiamp'+str(nmodes)+str(nscan)+'/r%d.tiff' % k)

        psi[k] = psiamp*np.exp(1j*psiangle)
        theta[k] = np.load(data_prefix+'datanpy/theta128sorted_'+str(k)+'.npy')
    
    
    # Class gpu solver
    psi3 = psi[:,130:131].astype('complex64')
    psi2 = np.zeros([3,1,n,n],dtype='complex64')   
    lamd2 = np.zeros([3,1,n,n],dtype='complex64')   
    lamd3 = np.zeros([ntheta,1,n],dtype='complex64')   
    
    # # print(psi1angle.shape)
    for center in range(185,200):
        print(center)
        slv = pt.Solver(nscan, theta, center, det, voxelsize,
                    energy, ntheta, 1, n, nprb, 1, 1, nmodes, ngpus)
        xi0,xi1,K,pshift = slv.takexi(psi3,psi2,lamd3,lamd2,1,1)        
        # xi0 = np.angle(psi3)+1j*0#.copy()
        # K = K*0+1                    
        u = np.zeros([1,n,n],dtype='complex64')
        u = slv.cg_tomo_batch(xi0, xi1, K, u, 1, -1, titer)
        dxchange.write_tiff(u[0].real,data_prefix+'test_center2/r'+str(center)+'.tiff',overwrite=True)
        dxchange.write_tiff(u[0].imag,data_prefix+'test_center2/i'+str(center)+'.tiff',overwrite=True)
    