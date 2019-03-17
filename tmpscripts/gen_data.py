import solver
import dxchange
import objects
import numpy as np
import cupy as cp
import signal
import sys


if __name__ == "__main__":

    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use
    print("gpu id:",igpu)

    
    # Model parameters
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    maxinta = [1,0.1,0.01,0.0001]  # maximal probe intensity
    prbsize = 16 # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [64, 64] # detector size
    ntheta = 256*3//2  # number of angles (rotations)
    noise = True  # apply discrete Poisson noise

    # Reconstrucion parameters
    modela = ['gaussian']  # minimization funcitonal (poisson,gaussian)
    alphaa = [1e-11] # tv regularization penalty coefficient
    piter = 200  # ptychography iterations
    titer = 200  # tomography iterations
    NITER = 1  # ADMM iterations

    ptheta = 4 # NEW: number of angular partitions for simultaneous processing in ptychography
    
    # Load a 3D object
    beta0 = dxchange.read_tiff('data/beta-pad2-256.tiff')[42:42+16]
    delta0 = -dxchange.read_tiff('data/delta-pad2-256.tiff')[42:42+16]
    beta = np.zeros([2*prbsize+beta0.shape[0],beta0.shape[1],beta0.shape[2]],dtype='float32')
    delta = np.zeros([2*prbsize+beta0.shape[0],beta0.shape[1],beta0.shape[2]],dtype='float32')    
    beta[prbsize:-prbsize] = beta0
    delta[prbsize:-prbsize] = delta0

    maxint = maxinta[igpu]
    obj = cp.array(delta+1j*beta)
    prb = cp.array(objects.probe(prbsize, maxint))
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    scan = cp.array(objects.scanner3(theta, obj.shape, prbshift,
                                    prbshift, prbsize, spiral=0, randscan=True, save=False)) 
    tomoshape = [len(theta), obj.shape[0], obj.shape[2]]

    # Class gpu solver 
    slv = solver.Solver(prb, scan, theta, det, voxelsize, energy, tomoshape, ptheta)
    # Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Compute data
    psi = slv.exptomo(slv.fwd_tomo(obj))
    data = np.zeros(slv.ptychoshape, dtype='float32')
    for k in range(0, ptheta):  # angle partitions in ptyocgraphy
        ids = np.arange(k*ntheta//ptheta, (k+1)*ntheta//ptheta)
        slv.cl_ptycho.setobj(scan[:, ids].data.ptr, prb.data.ptr)
        data[ids] = (cp.abs(slv.fwd_ptycho(psi[ids]))**2/slv.coefdata).get()
    print("max data = ", np.amax(data))      
    if (noise == True):# Apply Poisson noise
        data = np.random.poisson(data).astype('float32')

    # Save one angle
    dxchange.write_tiff(np.fft.fftshift(
        data[ntheta//2]), 'gendata/data'+str(maxint),overwrite=True)

  