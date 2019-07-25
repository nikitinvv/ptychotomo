import ptychotomo as pt
import dxchange
import numpy as np
import cupy as cp
import signal
import sys
import os
import sys


if __name__ == "__main__":
    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use    
    print("gpu id:",igpu)


    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    # Model parameters
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
 
    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 3*n//2  # number of angles (rotations)
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    maxint = 0.1  # maximal probe intensity
    prbsize = 16  # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [64, 64]  # detector size
    noise = True  # apply discrete Poisson noise

    # Reconstrucion parameters
    model = 'poisson'  # minimization funcitonal (poisson,gaussian)
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    NITER = 300  # ADMM iterations
    alpha = 3e-7

    ptheta = 32
    pnz = 32
    
    name = 'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)

    scan = cp.array(np.load('/data/staff/tomograms/viknik/gendata/coordinates'+name+'.npy'))
    data = np.load('/data/staff/tomograms/viknik/gendata/data'+name+'.npy')
    prb = cp.array(np.load('/data/staff/tomograms/viknik/gendata/prb'+name+'.npy'))
    theta = cp.array(np.load('/data/staff/tomograms/viknik/gendata/theta'+name+'.npy'))

    print("max data = ", np.amax(data))      

    # Class gpu solver 
    slv = pt.solver.Solver(prb, scan, theta, det, voxelsize, energy, ntheta, nz, n, ptheta, pnz)
    # Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)



    # Initial guess
    h = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    e = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')
    
    # ADMM
    u, psi = slv.admm(data, h, e, psi, phi, lamd,
                            mu, u, alpha, piter, titer, NITER, model)
    # Save result
    name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)+str(model)+str(piter)+str(titer)+str(NITER)

    dxchange.write_tiff(u.imag.get(),  '/data/staff/tomograms/viknik/beta/beta_'+name)
    dxchange.write_tiff(u.real.get(),  '/data/staff/tomograms/viknik/delta/delta_'+name)
    # if not os.path.exists('/data/staff/tomograms/viknik/lagr'):
    #     os.makedirs('/data/staff/tomograms/viknik/lagr')
    # np.save('/data/staff/tomograms/viknik/lagr/lagr'+name,lagr.get())
    


