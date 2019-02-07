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
    gamma = 0.25
    eta = 0.25
    piter = 4
    titer = 4
    NITER = 100
    maxint = 0.3
    voxelsize = 1e-6
    energy = 5
    nangles = 400
    noise = True
    shift=10
    # Load a 3D object
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')[:30:2, ::2, ::2]
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')[:30:2, ::2, ::2]
    

    alphaa = 1e-7*2**np.arange(-3,3)
    shifta = [8,10,12,15]
    maxinta = [0.2,0.3,0.4,0.5,0.6]

    for imaxint in range(0,len(maxinta)):
    # Create object.
    obj = objects.Object(beta, delta, voxelsize)
    # Create probe
    prb = objects.Probe(objects.gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
    # Detector parameters
    det = objects.Detector(63, 63)
    # Define rotation angles
    theta = np.linspace(0, 2*np.pi, nangles).astype('float32')
    # Scanner positions
    scanax, scanay = objects.scanner3(theta, beta.shape, shift, shift, prb.size, spiral=1, randscan=False, save=False)    
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
        slv.exptomo(slv.fwd_tomo(obj.complexform))))**2*det.x*det.y
    data = np.round(data)
    print("sigma = ", np.amax(np.sqrt(data)))

    # Apply Poisson noise (warning: Poisson distribution is discrete, so the resulting values are integers)
    if (noise == True):
        data = np.random.poisson(data).astype('float32')
    
    data/=(det.x*det.y)

    for ialpha in range(0,5):
    
    alpha = tau*1e-7/2*2**(ialpha-2)    
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
                           gamma, eta, piter, titer, NITER, 'grad')

    # Save result
    name='gr'+'_max'+np.str(maxint)+'_nang'+np.str(nangles)+'_shift'+np.str(shift)+'_alpha'+np.str(alpha)+'_noise'+np.str(noise)
    dxchange.write_tiff(x.beta,  'beta/beta'+name)
    dxchange.write_tiff(x.delta,  'delta/delta'+name)
    dxchange.write_tiff(psi.real,  'psi/psi'+name)
    np.save('residuals'+name, res)


    # ML
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
                           gamma, eta, piter, titer, NITER, 'ml')

    # Save result
    name='ml'+'_max'+np.str(maxint)+'_nang'+np.str(nangles)+'_shift'+np.str(shift)+'_alpha'+np.str(alpha)+'_noise'+np.str(noise)
    dxchange.write_tiff(x.beta,  'beta/beta'+name)
    dxchange.write_tiff(x.delta,  'delta/delta'+name)
    dxchange.write_tiff(psi.real,  'psi/psi'+name)
    np.save('residuals'+name, res)
