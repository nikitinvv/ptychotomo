import objects
import solver_gpu
import dxchange
import tomopy
import numpy as np
import signal
import sys

if __name__ == "__main__":
    rho = 1
    tau = 1
    alpha = tau*1e-7/2  
    gamma = 0.25
    eta = 0.25
    piter = 4
    titer = 4
    NITER = 20
    maxint = 0.1
    voxelsize = 1e-6
    energy = 5
    
    for ni in range(6,7):
        n = 2**ni
        nangles = 3*n//2
        noise = False
        # Load a 3D object
        delta = np.float32(np.random.random([n,n,n]))*1e-5
        beta = np.float32(np.random.random([n,n,n]))*1e-6
    
        obj = objects.Object(beta, delta, voxelsize)
        prb = objects.Probe(objects.gaussian(16, rin=0.8, rout=1.0), maxint=maxint)
        det = objects.Detector(n, n)
        theta = np.linspace(0, 2*np.pi, nangles).astype('float32')
        scanax, scanay = objects.scanner3(theta, beta.shape, 8, 8, prb.size, spiral=1, randscan=True, save=False)    
        tomoshape = [len(theta), obj.shape[0], obj.shape[2]]
        slv = solver_gpu.Solver(prb, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)
        def signal_handler(sig, frame):
            print('Remove class and free gpu memory')
            slv = []
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTSTP, signal_handler)

        data = np.abs(slv.fwd_ptycho(
            slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
        print("sigma = ", np.amax(np.sqrt(data*det.x*det.y)))

        if (noise == True):
            data = np.random.poisson(
                data*det.x*det.y).astype('float32')/(det.x*det.y)

        # Initial guess
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        e = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        lamd = np.zeros(tomoshape, dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x, psi, res = slv.admm(data, h, e, psi, phi, lamd, mu, x, rho, tau, alpha,
                            gamma, eta, piter, titer, NITER)

    