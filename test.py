import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np

import ptychotomo as pt

if __name__ == "__main__":

    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use
    # use cuda managed memory in cupy
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 128*3//2  # number of angles (rotations)
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    maxint = 0.3  # maximal probe intensity
    prbsize = 16  # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [64, 64]  # detector size
    noise = False  # apply discrete Poisson noise

    # Reconstrucion parameters
    model = 'poisson'  # minimization funcitonal (poisson,gaussian)
    alpha = 3*1e-7  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    niter = 100  # ADMM iterations
    ptheta = 32  # number of angular partitions for simultaneous processing in ptychography
    pnz = 32  # number of slice partitions for simultaneous processing in tomography

    # Load a 3D object
    beta = dxchange.read_tiff('data/beta-chip-128.tiff')
    delta = -dxchange.read_tiff('data/delta-chip-128.tiff')
    obj = cp.array(delta+1j*beta)

    # init probe, angles, scanner
    prb = cp.array(pt.probe(prbsize, maxint))
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    scan = cp.array(pt.scanner3(theta, obj.shape, prbshift,
                                prbshift, prbsize, spiral=0, randscan=True, save=True))
    # Class gpu solver
    slv = pt.Solver(prb, scan, theta, det, voxelsize,
                    energy, ntheta, nz, n, ptheta, pnz)

    def signal_handler(sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Compute data
    data = slv.fwd_ptycho_batch(slv.exptomo(slv.fwd_tomo_batch(obj)))
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print("max intensity on the detector: ", np.amax(data))

    # Initial guess
    h = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    e = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')

    # ADMM
    u, psi, lagr = slv.admm(data, h, e, psi, phi, lamd,
                            mu, u, alpha, piter, titer, niter, model)

    # subtract background in delta
    u.real -= u[0].real

    # Save result
    name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta' + \
        str(ntheta)+str(model)+str(piter)+str(titer)+str(niter)

    dxchange.write_tiff(u.imag.get(),  'beta/beta'+name)
    dxchange.write_tiff(-u.real.get(),  'delta/delta'+name)  # note sign change
    dxchange.write_tiff(cp.angle(psi).get(),  'psi/psiangle'+name)
    dxchange.write_tiff(cp.abs(psi).get(),  'psi/psiamp'+name)
    if not os.path.exists('lagr'):
        os.makedirs('lagr')
    np.save('lagr/lagr'+name, lagr.get())
