import solver
import dxchange
import objects
import numpy as np
import cupy as cp
import signal
import sys
import os


if __name__ == "__main__":

    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use
    print("gpu id:",igpu)

    
    # Model parameters
    voxelsize = 1e-7  # object voxel size
    energy = 5  # xray energy
    maxinta = [3,3,0.3,0.03]  # maximal probe intensity
    prbsize = 16 # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [64, 64] # detector size
    ntheta = 512*3//2  # number of angles (rotations)
    
    noise = True# apply discrete Poisson noise
    if (igpu==0)
        noise = False  
    maxint = maxinta[igpu]
    
    # Reconstrucion parameters
    modela = ['poisson','gaussian']  # minimization funcitonal (poisson,gaussian)
    alphaa = [1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5] # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    NITER = 250  # ADMM iterations

    ptheta = 32 # NEW: number of angular partitions for simultaneous processing in ptychography
    
    # Load a 3D object
    beta = dxchange.read_tiff('data/betacirc-512.tiff')[256:256+16].astype('float32')
    delta = dxchange.read_tiff('data/deltacirc-512.tiff')[256:256+16].astype('float32')
    #beta = np.zeros([2*prbsize+beta0.shape[0],beta0.shape[1],beta0.shape[2]],dtype='float32')
    #delta = np.zeros([2*prbsize+beta0.shape[0],beta0.shape[1],beta0.shape[2]],dtype='float32')    
    #beta[prbsize:-prbsize] = beta0
    #delta[prbsize:-prbsize] = delta0

    
    obj = cp.array(delta+1j*beta)
    prb = cp.array(objects.probe(prbsize, maxint))
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    scan = cp.array(objects.scanner3(theta, obj.shape, prbshift,
                                    prbshift, prbsize, spiral=0, randscan=True, save=True)) 
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

    for imodel in range(len(modela)):
        model = modela[imodel]
        for ialpha in range(len(alphaa)):            
            alpha = alphaa[ialpha]

            # Initial guess
            h = cp.zeros(tomoshape, dtype='complex64', order='C')+1
            psi = cp.zeros(tomoshape, dtype='complex64', order='C')+1
            e = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
            phi = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
            lamd = cp.zeros(tomoshape, dtype='complex64', order='C')
            mu = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
            u = cp.zeros(obj.shape, dtype='complex64', order='C')

            # ADMM
            u, psi, lagr = slv.admm(data, h, e, psi, phi, lamd,
                                    mu, u, alpha, piter, titer, NITER, model)
             # Save result
            name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
                str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)+str(model)+str(piter)+str(titer)+str(NITER)

            dxchange.write_tiff(u.imag.get(),  'beta/beta'+name)
            dxchange.write_tiff(u.real.get(),  'delta/delta'+name)
            dxchange.write_tiff(u[u.shape[0]//2].imag.get(),  'betap/beta'+name)
            dxchange.write_tiff(u[u.shape[0]//2].real.get(),  'deltap/delta'+name)
            dxchange.write_tiff(cp.angle(psi).get(),  'psi/psiangle'+name)
            dxchange.write_tiff(cp.abs(psi).get(),  'psi/psiamp'+name)    
            if not os.path.exists('lagr'):
                os.makedirs('lagr')
            np.save('lagr/lagr'+name,lagr.get())
