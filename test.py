import objects
import solver_gpu
import dxchange
import tomopy
import numpy as np
import signal
import sys


if __name__ == "__main__":

    # Parameters.
    rho = 0.5
    gamma = 0.5
    eta = 0.25
    piter = 4
    titer = 4
    NITER = 50
    maxint = 0.1
    voxelsize = 1e-6
    energy = 5

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
    theta = np.linspace(0, 2*np.pi, 400).astype('float32')
    # Scanner positions
    scanax, scanay = objects.scanner3(theta, beta.shape, 10, 10, margin=[
        prb.size, prb.size], offset=[0, 0], spiral=1)
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

    # Compute data  |FQ(exp(i\nu R x))|^2,
    data = np.abs(slv.fwd_ptycho(
        slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
    print("sigma = ", np.amax(np.sqrt(data*det.x*det.y)))

    # Apply Poisson noise (warning: Poisson distribution is discrete, so the resulting values are integers)
    data = np.random.poisson(data*det.x*det.y).astype('float32')/(det.x*det.y)

    # Initial guess
    h = np.ones(tomoshape, dtype='complex64', order='C')
    psi = np.ones(tomoshape, dtype='complex64', order='C')
    lamd = np.zeros(tomoshape, dtype='complex64', order='C')
    x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
        obj.shape, dtype='float32', order='C'), voxelsize)

    # ADMM
    x, psi, res = slv.admm(data, h, psi, lamd, x, rho,
                           gamma, eta, piter, titer, NITER)

    # Save result
    dxchange.write_tiff(x.beta,  'beta/beta')
    dxchange.write_tiff(x.delta,  'delta/delta')
    dxchange.write_tiff(psi.real,  'psi/psi')
    np.save('residuals', res)
