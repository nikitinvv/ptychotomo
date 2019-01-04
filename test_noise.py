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

    # Parameters fixed
    rho = 0.5
    gamma = 0.25
    eta = 0.25/720/128/1e5*5
    piter = 1
    titer = 1
    NITER = 1024
    voxelsize = 1e-6
    energy = 5

    # Load a 3D object.
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')#[::2, ::2, ::2]
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')#[::2, ::2, ::2]
    
    obj = objects.Object(beta, delta, voxelsize)
    det = objects.Detector(127, 127)
    theta = np.linspace(0, 2*np.pi, 720).astype('float32')
    tomoshape = [len(theta), obj.shape[1], obj.shape[2]]

    maxint = 10
    prb = objects.Probe(gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
    scan, scanax, scanay = scanner3(theta, beta.shape, 12, 12, margin=[
                                    prb.size, prb.size], offset=[0, 0], spiral=1)
    slv = solver_gpu.Solver(prb, scanax, scanay,
                            theta, det, voxelsize, energy, tomoshape)
    # data
    psis = slv.fwd_tomo(obj.complexform)
    data = np.abs(slv.fwd_ptycho(slv.exptomo(psis)))**2

    # # rec
    tau = 1e-16
    alpha = 1e-16
    h = np.ones(psis.shape, dtype='complex64', order='C')
    psi = np.ones(psis.shape, dtype='complex64', order='C')
    lamd = np.zeros(psi.shape, dtype='complex64', order='C')
    phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
        obj.shape, dtype='float32', order='C'), voxelsize)
    x = slv.admm(data, h, psi, phi, lamd, mu, x, rho, tau,
                 gamma, eta, alpha, piter, titer, NITER)
    dxchange.write_tiff(x.beta[64],  'beta/beta_joint_20over')
    dxchange.write_tiff(x.delta[64],  'delta/delta_joint_20over')

    # rec tv
    tau = 1e3*1e3
    alpha = 1e-2*1e3*5
    h = np.ones(psis.shape, dtype='complex64', order='C')
    psi = np.ones(psis.shape, dtype='complex64', order='C')
    lamd = np.zeros(psi.shape, dtype='complex64', order='C')
    phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
    x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
        obj.shape, dtype='float32', order='C'), voxelsize)
    x = slv.admm(data, h, psi, phi, lamd, mu, x, rho, tau,
                 gamma, eta, alpha, piter, titer, NITER)
    dxchange.write_tiff(x.beta[40],  'beta/beta_joint_tv_20over_10maxint')
    dxchange.write_tiff(x.delta[64],  'delta/delta_joint_tv_20over_10maxint')


    # Denoise for different intensities
    maxinta = [100, 10, 1, 0.1]
    for k in range(0,1):
        maxint = maxinta[k]
        prb = objects.Probe(gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
        scan, scanax, scanay = scanner3(theta, beta.shape, 12, 12, margin=[
            prb.size, prb.size], offset=[0, 0], spiral=1)
        slv = solver_gpu.Solver(prb, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)
        # data
        data = np.abs(slv.fwd_ptycho(
            slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
        data = np.random.poisson(
            data*det.x*det.y).astype('float32')/(det.x*det.y)

        # rec
        tau = 1e-16
        alpha = 1e-16
        h = np.ones(psis.shape, dtype='complex64', order='C')
        psi = np.ones(psis.shape, dtype='complex64', order='C')
        lamd = np.zeros(psi.shape, dtype='complex64', order='C')
        phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)
        x = slv.admm(data, h, psi, phi, lamd, mu, x, rho, tau,
                     gamma, eta, alpha, piter, titer, NITER)
        dxchange.write_tiff(
            x.beta[40],   'beta/beta_joint_20over_' + str(maxint)+'_maxint_noise')
        dxchange.write_tiff(
            x.delta[64],  'delta/delta_joint_20over_'+str(maxint)+'_maxint_noise')

        for itau in range(0, 5):
            for ialpha in range(0, 5):
               # rec tv
                tau = 1e3*1e3*2**(itau-2)
                alpha = 1e-2*1e3*5*2**(ialpha-2)
                h = np.ones(psis.shape, dtype='complex64', order='C')
                psi = np.ones(psis.shape, dtype='complex64', order='C')
                lamd = np.zeros(psi.shape, dtype='complex64', order='C')
                phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
                mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
                x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
                    obj.shape, dtype='float32', order='C'), voxelsize)
                x = slv.admm(data, h, psi, phi, lamd, mu, x, rho, tau,
                         gamma, eta, alpha, piter, titer, NITER)
                dxchange.write_tiff(x.beta[40],   'beta/beta_joint_tv_' + str(
                    itau)+'_'+str(ialpha)+'_20over_'+str(maxint)+'_maxint_noise')
                dxchange.write_tiff(x.delta[64],  'delta/delta_joint_tv_'+str(
                    itau)+'_'+str(ialpha)+'_20over_'+str(maxint)+'_maxint_noise')
