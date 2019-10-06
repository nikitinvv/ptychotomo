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
    ntheta = 32  # number of angles (rotations)
    voxelsize = 1e-7  # object voxel size
    energy = 8.8  # xray energy
    nprb = 128  # probe size
    nscan = 1000
    det = [128, 128]  # detector size
    noise = False  # apply discrete Poisson noise
    retrieve_prb = False

    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 3*1e-15  # tv regularization penalty coefficient
    piter = 8  # ptychography iterations
    titer = 32  # tomography iterations
    niter = 1  # ADMM iterations
    ptheta = 4  # number of angular partitions for simultaneous processing in ptychography
    pnz = 16  # number of slice partitions for simultaneous processing in tomography

    beta = dxchange.read_tiff('model/beta-chip-276_600_600.tiff')
    delta = dxchange.read_tiff('model/delta-chip-276_600_600.tiff')
    obj = cp.array(delta+1j*beta)
    # read probe
    prb0 = cp.zeros([ntheta, nprb, nprb], dtype='complex64')
    prbamp = cp.array(dxchange.read_tiff(
        'model/prbamp.tiff').astype('float32'))
    prbang = cp.array(dxchange.read_tiff(
        'model/prbang.tiff').astype('float32'))
    prb0[0] = prbamp*cp.exp(1j*prbang)
    # read scan positions
    scan = cp.ones([2, ntheta, nscan], dtype='float32')
    scan[:, 0] = cp.load('model/coords.npy')[:, :nscan].astype('float32')
    # repeat
    for k in range(ntheta):
        scan[:, k] = scan[:, 0]
        prb0[k] = prb0[0]
    # set theta
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    # Class gpu solver
    slv = pt.Solver(nscan, nprb, theta, det, voxelsize,
                    energy, ntheta, nz, n, ptheta, pnz)

    data = slv.fwd_ptycho_batch(slv.exptomo(
        slv.fwd_tomo_batch(obj)), scan, prb0)
    # dxchange.write_tiff(slv.exptomo(slv.fwd_tomo_batch(obj)).real.get(), 'datapsi', overwrite=True)
    # dxchange.write_tiff(data, 'data', overwrite=True)
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print("max intensity on the detector: ", np.amax(data))
    # Initial guess
    h = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    e = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')
    prb = prb0.copy()
    # ADMM
    u, psi, prb = slv.admm(data, h, e, psi, scan, prb, phi, lamd,
                      mu, u, alpha, piter, titer, niter, model, retrieve_prb)

    # subtract background in delta
    # u.real -= np.mean(u[4:8].real)

    # Save result
    name = 'reg'+str(alpha)+'noise'+str(noise) + 'ntheta' + \
        str(ntheta)+str(model)+str(piter)+str(titer)+str(niter)

    print(cp.linalg.norm(psi))
    dxchange.write_tiff(u.imag.get(),  'beta/beta'+name)
    dxchange.write_tiff(u.real.get(),  'delta/delta'+name)  # note sign change
    dxchange.write_tiff_stack(cp.angle(psi).get(),  'psi/psiangle'+name)
    dxchange.write_tiff_stack(cp.abs(psi).get(),  'psi/psiamp'+name)


# plot result
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))
    plt.subplot(2, 2, 1)
    plt.title('scan positions')
    plt.plot(scan[0, 0, :].get(), scan[1, 0, :].get(),
             '.', markersize=1.5, color='blue')
    plt.xlim([0, n])
    plt.ylim([0, nz])
    plt.gca().invert_yaxis()
    plt.subplot(2, 4, 1)
    plt.title('object real')
    plt.imshow(u[n//2].real.get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 2)
    plt.title('object imag')
    plt.imshow(u[n//2].imag.get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 3)
    plt.title('retrieved probe phase')
    plt.imshow(cp.angle(prb[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 4)
    plt.title('retrieved probe amplitude')
    plt.imshow(cp.abs(prb[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.title('object phase')
    plt.imshow(cp.angle(psi[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.title('object amplitude')
    plt.imshow(cp.abs(psi[0]).get(), cmap='gray')
    plt.colorbar()
    plt.savefig('result.png', dpi=600)
    print("See result.png and tiff files in rec/ folder")
