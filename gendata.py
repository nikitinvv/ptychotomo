import ptychotomo as pt
import dxchange
import numpy as np
import cupy as cp
import signal
import sys
import os
import sys
import time

if __name__ == "__main__":
    
    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use
    print("gpu id:",igpu)
    
    # Model parameters
    n = 512
    nz = 512
    voxelsize = 1e-6/2  # object voxel size
    energy = 8.8  # xray energy
    maxint = [3,0.3,0.03,0.003]  # maximal probe intensity
    alpha = [8e-9,1e-8,2e-8,4e-8]
    maxint = maxint[igpu]
    alpha = alpha[igpu]
    
    prbsize = 16 # probe size
    prbshift = 12  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [128, 128] # detector size
    ntheta = 128*3//2  # number of angles (rotations)
    noise = True  # apply discrete Poisson noise
    
    ptheta = 4
    pnz = 16
    beta = dxchange.read_tiff('../data/beta-pad.tiff')
    delta = -dxchange.read_tiff('../data/delta-pad.tiff')
    obj = cp.array(delta+1j*beta)
    

    prb = cp.array(pt.objects.probe(prbsize, maxint))
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    scan = cp.array(pt.objects.scanner3(theta, obj.shape, prbshift,
                                    prbshift, prbsize, spiral=0, randscan=True, save=False)) 
    #tomoshape = [len(theta), obj.shape[0], obj.shape[2]]
    # Class gpu solver 
    slv = pt.solver.Solver(prb, scan, theta, det, voxelsize, energy, ntheta, nz, n, ptheta, pnz)
    # Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Compute data
    psi = slv.exptomo(slv.fwd_tomo_batch(obj))
    
    # Compute data
    data = slv.fwd_ptycho_batch(slv.exptomo(slv.fwd_tomo_batch(obj)))
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print("max intensity on the detector: ", np.amax(data))


    name = 'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)


    np.save('/data/staff/tomograms/viknik/gendata/data'+name,data)    
    np.save('/data/staff/tomograms/viknik/gendata/coordinates'+name,scan.get())    
    np.save('/data/staff/tomograms/viknik/gendata/theta'+name,theta.get())    
    np.save('/data/staff/tomograms/viknik/gendata/prb'+name,prb.get())    
    
       
