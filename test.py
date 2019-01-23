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
    NITER = 200
    maxint = 0.1
    voxelsize = 1e-6
    energy = 5
    nangles = 100
    noise = False
    # Load a 3D object
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')[::2, ::2, ::2]
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')[::2, ::2, ::2]
    
    # Create object.
    obj = objects.Object(beta, delta, voxelsize)
    # Create probe
    prb = objects.Probe(objects.gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
    # Detector parameters
    det = objects.Detector(63, 63)
    # Define rotation angles
    theta = np.linspace(0, 2*np.pi, nangles).astype('float32')
    # Scanner positions
    scanax, scanay = objects.scanner3(theta, beta.shape, 8, 8, prb.size, spiral=1, randscan=True, save=True)    
    # tomography data shape
    tomoshape = [len(theta), obj.shape[0], obj.shape[2]]
    # Class solver
    slv = solver_gpu.Solver(prb, scanax, scanay,
                            theta, det, voxelsize, energy, tomoshape)
    def signal_handler(sig, frame):
        print('Remove class and free gpu memory')
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # # Adjoint test
    # f = slv.exptomo(slv.fwd_tomo(obj.complexform))
    # g = slv.fwd_ptycho(f)
    # ff = slv.adj_ptycho(g)
    # s1 = np.sum(f*np.conj(ff))
    # s2 = np.sum(g*np.conj(g))
    # print(s1,s2)
    # print((s1-s2)/s1)
    

    # Compute data  |FQ(exp(i\nu R x))|^2,
    data = np.abs(slv.fwd_ptycho(
        slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
    print("sigma = ", np.amax(np.sqrt(data*det.x*det.y)))

    # Apply Poisson noise (warning: Poisson distribution is discrete, so the resulting values are integers)
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

    # Save result
    dxchange.write_tiff(x.beta,  'beta/beta')
    dxchange.write_tiff(x.delta,  'delta/delta')
    dxchange.write_tiff(psi.real,  'psi/psi')
    np.save('residuals', res)
