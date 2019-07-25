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
    
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    # Model parameters
    n = 128
    nz = 128
    ntheta = 3*n//2
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    maxint = 0.1
    alpha = 3e-7
        
    prbsize = 16 # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [64, 64] # detector size
    noise = True  # apply discrete Poisson noise
    
    ptheta = 32
    pnz = 32
    beta = dxchange.read_tiff('data/beta-chip-128.tiff')
    delta = -dxchange.read_tiff('data/delta-chip-128.tiff')
    obj = cp.array(delta+1j*beta)
    

    prb = cp.array(pt.objects.probe(prbsize, maxint))
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    scan = cp.array(pt.objects.scanner3(theta, obj.shape, prbshift,
                                    prbshift, prbsize, spiral=0, randscan=True, save=False)) 
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
        for k in range(0,data.shape[0]):    
            data[k] = np.random.poisson(data[k]).astype('float32')
    print("max intensity on the detector: ", np.amax(data))


    name = 'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)


    np.save('/data/staff/tomograms/viknik/gendata/data'+name,data)    
    np.save('/data/staff/tomograms/viknik/gendata/coordinates'+name,scan.get())    
    np.save('/data/staff/tomograms/viknik/gendata/theta'+name,theta.get())    
    np.save('/data/staff/tomograms/viknik/gendata/prb'+name,prb.get())    
    
       
