import objects
import solver_gpu
import dxchange
import tomopy
import numpy as np
import signal
import sys

if __name__ == "__main__":
    rho = 0.5
    tau = 1e-7
    alpha = tau*1e-7/2*1e-4
    gamma = 0.5
    eta = 0.5
    piter = 4
    titer = 4
    NITER = 10
    maxint = 4
    voxelsize = 1e-6
    energy = 5
    nangles = 400
    noise = False
    # Load a 3D object
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')#[:30:2, ::2, ::2]
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')#[:30:2, ::2, ::2]

    # Create object.
    obj = objects.Object(beta, delta, voxelsize)
    # Create probe
    prb = objects.Probe(objects.gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
    # Detector parameters
    det = objects.Detector(128, 128)
    # Define rotation angles
    theta = np.linspace(0, 2*np.pi, nangles).astype('float32')
    # Scanner positions
    scanax, scanay = objects.scanner3(theta, beta.shape, 10, 10, prb.size, spiral=1, randscan=False, save=False)    
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
        slv.exptomo(slv.fwd_tomo(obj.complexform))))**2*det.x*det.y*maxint
    # Convert to integers        
    data = np.round(data)
    print("sigma = ", np.amax(np.sqrt(data)))

    dxchange.write_tiff(
             np.fft.fftshift(data[360][25]),  'data_'+str(maxint))