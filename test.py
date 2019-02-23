import objects
import solver_gpu
import dxchange
import tomopy
import numpy as np
import signal
import sys

if __name__ == "__main__":

    alpha = 1e-7
    piter = 4
    titer = 4
    NITER = 50
    maxint = 1
    voxelsize = 1e-6
    energy = 5
    nangles = 400
    noise = True

    # Load a 3D object
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')#[:30:2, ::2, ::2]
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')#[:30:2, ::2, ::2]

    # Create object, probe, detector, angles, scan positions
    obj = objects.Object(beta, delta, voxelsize)
    prb = objects.Probe(objects.gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
    det = objects.Detector(63, 63)
    theta = np.linspace(0, 2*np.pi, nangles).astype('float32')
    scan = objects.scanner3(theta, beta.shape, 10, 10,
                            prb.size, spiral=1, randscan=False, save=False)
    # tomography data shape
    tomoshape = [len(theta), obj.shape[0], obj.shape[2]]
    # Class solver
    slv = solver_gpu.Solver(prb, scan, theta, det,
                            voxelsize, energy, tomoshape)

    def signal_handler(sig, frame):
        print('Remove class and free gpu memory')
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Compute data  |FQ(exp(i\nu R x))|^2, with using normalized operators
    data = np.abs(slv.fwd_ptycho(
        slv.exptomo(slv.fwd_tomo(obj.complexform))))**2/slv.coefdata
    # Convert to integers
    data = np.round(data)
    print("max data = ", np.amax(data))

    # Apply Poisson noise
    if (noise == True):
        data = np.random.poisson(data).astype('float32')

    # Initial guess
    h = np.ones(tomoshape, dtype='complex64', order='C')
    psi = np.ones(tomoshape, dtype='complex64', order='C')
    e = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    lamd = np.zeros(tomoshape, dtype='complex64', order='C')
    mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
        obj.shape, dtype='float32', order='C'), voxelsize)

    model = 'poisson'
    # ADMM
    x, psi, res = slv.admm(data, h, e, psi, phi, lamd,
                           mu, x, alpha, piter, titer, NITER, model)

    # Subtract background value for delta
    x.delta -= -1.4e-5

    # Save result
    name = 'reg'+np.str(alpha)+'noise'+np.str(noise)+np.str(model)
    dxchange.write_tiff(x.beta,  'beta/beta'+name)
    dxchange.write_tiff(x.delta,  'delta/delta'+name)
    dxchange.write_tiff(psi.real,  'psi/psi'+name)
    np.save('residuals'+name, res)
