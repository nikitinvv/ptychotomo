import objects
import solver_gpu
import dxchange
import tomopy
import numpy as np


def gaussian(size, rin=0.8, rout=1):
    r, c = np.mgrid[:size, :size] + 0.5
    rs = np.sqrt((r - size/2)**2 + (c - size/2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def scanner3(theta, shape, sx, sy, margin=[0, 0], offset=[0, 0], spiral=0):
    a = spiral
    scan = []
    lenmaxx = 0
    lenmaxy = 0

    for m in range(len(theta)):
        s = objects.Scanner(shape, sx, sy, margin, offset=[
                            offset[0], np.mod(offset[1]+a, sy)])
        scan.append(s)
        a += spiral
        lenmaxx = max(lenmaxx, len(s.x))
        lenmaxy = max(lenmaxy, len(s.y))

    scanax = -1+np.zeros([len(theta), lenmaxx], dtype='int32')
    scanay = -1+np.zeros([len(theta), lenmaxy], dtype='int32')

    for m in range(len(theta)):
        scanax[m, :len(scan[m].x)] = scan[m].x
        scanay[m, :len(scan[m].y)] = scan[m].y

    return scan, scanax, scanay


if __name__ == "__main__":

    # Parameters.
    rho = 0.5
    tau= 1e3*1e3
    alpha = 1e-2*1e3*5*2
    gamma = 0.25
    eta = 0.25/720/64/1e5*5
    piter = 1
    titer = 1
    NITER = 256
    maxint = 0.1
    voxelsize = 1e-6
    energy = 5

    # Load a 3D object.
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')
    # Create object.
    obj = objects.Object(beta, delta, voxelsize)
    # Create probe.
    prb = objects.Probe(gaussian(15, rin=0.8, rout=1.0), maxint=maxint)  
    # Detector parameters.
    det = objects.Detector(128, 128)
    # Define rotation angles.
    theta = np.linspace(0, 2*np.pi, 720).astype('float32')
    # Raster scan parameters for each rotation angle.
    scan, scanax, scanay = scanner3(theta, beta.shape, 12, 12, margin=[
                                    prb.size, prb.size], offset=[0, 0], spiral=1)
    tomoshape = [len(theta), obj.shape[1], obj.shape[2]]

    # class solver
    slv = solver_gpu.Solver(prb, scanax, scanay,
                            theta, det, voxelsize, energy, tomoshape)

    # Project
    psis = slv.fwd_tomo(obj.complexform)
    psis = slv.exptomo(psis)
    # Propagate
    data = slv.fwd_ptycho(psis)
    data = np.abs(data)
    data = data**2*det.x*det.y
    data = np.int(data)
    data = np.random.poisson(data).astype('float32')
    print(np.amax(np.sqrt(data)))
    data=data/(det.x*det.y)

    dxchange.write_tiff(data[360][25],  'data_noise/data_'+str(np.int(np.amax(data))))
    
    exit()
    
    h = np.ones(psis.shape, dtype='complex64', order='C')
    psi = np.ones(psis.shape, dtype='complex64', order='C')
    lamd = np.zeros(psi.shape, dtype='complex64', order='C')
    phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
        obj.shape, dtype='float32', order='C'), voxelsize)
   
    # ADMM
    x = slv.admm(data, h, psi, phi, lamd, mu, x, rho, tau, gamma, eta, alpha, piter, titer, NITER)

    # Save result
    dxchange.write_tiff(x.beta[32],  'beta/beta2')
    dxchange.write_tiff(x.delta[32],  'delta/delta2')

   