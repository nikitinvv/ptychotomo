import objects
import solver_gpu
import dxchange
#import tomopy
import numpy as np
import sys

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
    gamma = 0.25
    eta = 0.25
    voxelsize = 1e-6
    energy = 5

    # Load a 3D object.
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')[::2, ::2, ::2]
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')[::2, ::2, ::2]
    # Create object.
    obj = objects.Object(beta, delta, voxelsize)
    # Detector parameters.
    det = objects.Detector(63, 63)
    # Define rotation angles.
    theta = np.linspace(0, 2*np.pi, 720).astype('float32')

    if(sys.argv[1] == "True"):

        print('compute regular')
        maxint = 100
        rho = 1e-10
        piter = 250
        titer = 250
        NITER = 1
        # Create probe.
        prb = objects.Probe(gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
        # Raster scan parameters for each rotation angle.
        scan, scanax, scanay = scanner3(theta, beta.shape, 12, 6, margin=[
                                        prb.size, prb.size], offset=[0, 0], spiral=1)
        tomoshape = [len(theta), obj.shape[0], obj.shape[2]]

        # class solver
        slv = solver_gpu.Solver(prb, scan, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)

        # data
        data = np.abs(slv.fwd_ptycho(
            slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
        print('sigma=', np.sqrt(np.amax(np.abs(data))*det.x*det.y))

        # rec
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(psi.shape, dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x = slv.admm(data, h, psi, lamd, x, rho,
                     gamma, eta, piter, titer, NITER)


        # Save result
        dxchange.write_tiff(x.beta,  '../rec_ptycho/beta_s/beta_st_over12')
        dxchange.write_tiff(
            x.delta,  '../rec_ptycho/delta_s/delta_st_over12')


        rho = 0.5
        piter = 1
        titer = 1
        NITER = 250
        # rec
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(psi.shape, dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x = slv.admm(data, h, psi, lamd, x, rho,
                     gamma, eta, piter, titer, NITER)

        # Save result
        dxchange.write_tiff(x.beta,  '../rec_ptycho/beta_s/beta_joint_over12')
        dxchange.write_tiff(
            x.delta,  '../rec_ptycho/delta_s/delta_joint_over12')



    maxinta = [100,10,1,0.1]
    for k in len(maxinta):
        maxint = maxinta[k]
        # Create probe.
        prb = objects.Probe(gaussian(15, rin=0.8, rout=1.0), maxint=maxint)
        # Raster scan parameters for each rotation angle.
        scan, scanax, scanay = scanner3(theta, beta.shape, 6, 6, margin=[
                                        prb.size, prb.size], offset=[0, 0], spiral=1)
        tomoshape = [len(theta), obj.shape[0], obj.shape[2]]

        # class solver
        slv = solver_gpu.Solver(prb, scan, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)

        # data
        data = np.abs(slv.fwd_ptycho(
            slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
        print('sigma=', np.sqrt(np.amax(np.abs(data))*det.x*det.y))
        data = np.random.poisson(
            data*det.x*det.y).astype('float32')/(det.x*det.y)


        rho = 1e-10
        piter = 250
        titer = 250
        NITER = 1
        # rec
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(psi.shape, dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x = slv.admm(data, h, psi, lamd, x, rho,
                     gamma, eta, piter, titer, NITER)

        dxchange.write_tiff(
            x.beta,   '../rec_ptycho/beta/beta_st_over12_' + str(maxint)+'_maxint_noise')
        dxchange.write_tiff(
            x.delta,  '../rec_ptycho/delta/delta_st_over12_'+str(maxint)+'_maxint_noise')


        rho = 0.5
        piter = 1
        titer = 1
        NITER = 250
        # rec
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(psi.shape, dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)

        # ADMM
        x = slv.admm(data, h, psi, lamd, x, rho,
                     gamma, eta, piter, titer, NITER)

        dxchange.write_tiff(
            x.beta,   '../rec_ptycho/beta/beta_joint_over12_' + str(maxint)+'_maxint_noise')
        dxchange.write_tiff(
            x.delta,  '../rec_ptycho/delta/delta_joint_over12_'+str(maxint)+'_maxint_noise')
