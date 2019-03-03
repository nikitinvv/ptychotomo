import solver
import dxchange
import objects
import numpy as np
import cupy as cp
import signal
import sys


if __name__ == "__main__":

    piter = 10
    titer = 10
    NITER = 300
    voxelsize = 1e-6
    energy = 5
    noise = True
    
    cp.cuda.Device(0).use()  # gpu id to use
    modela = ['gaussian','poisson']
    maxinta = [0.01,0.05,0.1,1]#,0.01,0.1,1]#,0.001,0.1]#,0.30.5,1,3]        
    nthetaa = [384]    
    prbshifta = [8]
    alphaa = [1e-8,5e-8,2.5e-8,7.5e-8,1e-7]

    for imodel in range(0,len(modela)):
        for imaxint in range(0,len(maxinta)):
            for intheta in range(0,len(nthetaa)):
                for iprbshift in range(0,len(prbshifta)):  
                    model = modela[imodel]              
                    maxint = maxinta[imaxint]              
                    ntheta = nthetaa[intheta]              
                    prbshift = prbshifta[iprbshift]
                    # Load a 3D object
                    beta = dxchange.read_tiff(
                        'data/BETA256.tiff').astype('float32')[32:80]
                    delta = dxchange.read_tiff(
                        'data/DELTA256.tiff').astype('float32')[32:80]
                    
                    # Create object, probe, detector, angles, scan positions
                    obj = cp.array(delta+1j*beta)
                    prb = cp.array(objects.probe(16, maxint))    
                    det = [64, 64]
                    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
                    scan = cp.array(objects.scanner3(theta, obj.shape, prbshift,
                                                    prbshift, prb.shape[0], spiral=0, randscan=True, save=False))    
                    # tomography data shape
                    tomoshape = [len(theta), obj.shape[0], obj.shape[2]]

                    # Class solver
                    slv = solver_gpu.Solver(prb, scan, theta, det,
                                            voxelsize, energy, tomoshape)

                    def signal_handler(sig, frame):
                        slv = []
                        sys.exit(0)
                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTSTP, signal_handler)

                    # Compute data
                    psi = slv.exptomo(slv.fwd_tomo(obj))
                    # generate by angles partitions
                    data = np.zeros(slv.ptychoshape, dtype='float32')
                    for k in range(0, 16):
                        ids = np.arange(k*ntheta//16, (k+1)*ntheta//16)
                        slv.cl_ptycho.setobj(scan[:, ids].data.ptr, prb.data.ptr)
                        data[ids] = (
                            cp.abs(slv.fwd_ptycho(psi[ids]))**2/slv.coefdata).get()                    
                    # Apply Poisson noise
                    if (noise == True):
                        data = np.random.poisson(data).astype('float32')
                    for ialpha in range(0,len(alphaa)):
                        alpha=alphaa[ialpha]
                        print("max data, maxint, ntheta, prbshift, alpha", np.amax(data),maxint, ntheta, prbshift, alpha)
                        # Initial guess
                        h = cp.zeros(tomoshape, dtype='complex64', order='C')+1
                        psi = cp.zeros(tomoshape, dtype='complex64', order='C')+1
                        e = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
                        phi = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
                        lamd = cp.zeros(tomoshape, dtype='complex64', order='C')
                        mu = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
                        u = cp.zeros(obj.shape, dtype='complex64', order='C')    
                        # ADMM
                        u, psi, res = slv.admm(data, h, e, psi, phi, lamd,
                                                mu, u, alpha, piter, titer, NITER, model)
                        # Save result
                        name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
                            str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)+str(model)
                        dxchange.write_tiff(u.imag.get(),  'beta/beta'+name)
                        dxchange.write_tiff(u.real.get(),  'delta/delta'+name)
                        dxchange.write_tiff(u[u.shape[0]//2].imag.get(),  'betap/beta'+name)
                        dxchange.write_tiff(u[u.shape[0]//2].real.get(),  'deltap/delta'+name)
                        dxchange.write_tiff(psi.real.get(),  'psi/psir'+name)
                        dxchange.write_tiff(psi.imag.get(),  'psi/psii'+name)
                        np.save('lagr/lagr'+name, res.get())

