import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np

import ptychotomo as pt

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])

    cp.cuda.Device(igpu).use()  # gpu id to use
    # use cuda managed memory in cupy
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Model parameters
    n = 600  # object size n x,y
    nz = 276  # object size in z
    ntheta = 256  # number of angles (rotations)
    theta = np.linspace(0, np.pi, ntheta).astype('float32') # angles

    titer = 16  # tomography iterations
    pnz = 39  # number of slice partitions for simultaneous processing in tomography

    # Load object
    beta = dxchange.read_tiff('model/beta-chip-276_600_600.tiff')
    delta = dxchange.read_tiff('model/delta-chip-276_600_600.tiff')
    u0 = delta+1j*beta
    
    # Class gpu solver
    slv = pt.SolverTomo(theta, ntheta, nz, n, pnz)  
    # generate data                     
    data = slv.fwd_tomo_batch(u0)        
    # adjoint test
    u1 = slv.adj_tomo_batch(data)        
    print('Adjoint test: ', np.sum(data*np.conj(data)),'=?',np.sum(u0*np.conj(u1)))
    dxchange.write_tiff(data.real, 'data', overwrite=True)

    # Initial guess
    u = u0*0
    # Solve tomography problem
    u = slv.cg_tomo_batch(data, u, titer)    

    # Save result
    name = 'ntheta' + str(ntheta)+str(titer)
    dxchange.write_tiff(u.imag,  'rec/beta'+name,overwrite=True)
    dxchange.write_tiff(u.real,  'rec/delta'+name,overwrite=True)  # note sign change
    
    # plot result
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))
    plt.subplot(1, 2, 1)
    plt.title('object real')
    plt.imshow(u[nz//4].real, cmap='gray')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('object imag')
    plt.imshow(u[nz//4].imag, cmap='gray')
    plt.colorbar()
    plt.savefig('result_tomo.png', dpi=600)
    print("See result_tomo.png and tiff files in rec/ folder")
