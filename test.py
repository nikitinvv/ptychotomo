import objects
# import solver
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
    tau = 1e-3
    gamma = 0.25
    eta = 0.25
    piter = 1
    titer = 1
    maxint = 100
    voxelsize = 1e-6
    energy = 5

    # Load a 3D object.
    beta = dxchange.read_tiff('data/lego-imag.tiff').astype('float32')
    delta = dxchange.read_tiff('data/lego-real.tiff').astype('float32')
    #print(beta.shape)
    # print(np.amax(beta))
    # print(np.amax(delta))
    # exit()
    beta = tomopy.misc.phantom.shepp3d(size=256, dtype=u'float32')*1e-6
    delta = tomopy.misc.phantom.shepp3d(size=256, dtype=u'float32')*1e-5*5

    # Create object.
    obj = objects.Object(beta, delta, voxelsize)
    # Create probe.
    weights = gaussian(15, rin=0.8, rout=1.0)
    prb = objects.Probe(weights, maxint=maxint)
    # Detector parameters.
    det = objects.Detector(63,63)
    # Define rotation angles.
    theta = np.linspace(0, np.pi, 60).astype('float32')
    # Raster scan parameters for each rotation angle.
    scan, scanax, scanay = scanner3(theta, beta.shape, 6, 6, margin=[
                                    prb.size, prb.size], offset=[0, 0], spiral=1)
    tomoshape = [len(theta), obj.shape[1], obj.shape[2]]
    # class solver
    slv = solver_gpu.Solver(prb, scan, scanax, scanay,
                            theta, det, voxelsize, energy, tomoshape)

    # Adjoint and normalization test
    # a = slv.fwd_reg(obj.complexform)
    # b = slv.adj_reg(a)
    # aa = slv.fwd_reg(b)
    # s1 = np.sum(np.complex64(a)*np.conj(np.complex64(aa)))
    # s2 = np.sum(np.complex64(b)*np.conj(np.complex64(b)))
    # s3 = np.sum(np.complex64(aa)*np.conj(np.complex64(aa)))

    # print("Adjoint and normalization test gr: "+str([s1,s2,(s1-s2)/s1,s1/s3]))
    # exit()
    # r = 1/np.sqrt(len(theta)*obj.shape[2]/2)
    # a = obj.complexform
    # b = slv.fwd_tomo(a)*r
    # aa = slv.adj_tomo(b)*r
    # s1 = np.sum(np.complex64(a)*np.conj(np.complex64(aa)))
    # s2 = np.sum(np.complex64(b)*np.conj(np.complex64(b)))
    # s3 = np.sum(np.complex64(aa)*np.conj(np.complex64(aa)))
    # print("Adjoint and normalization test tomo: "+str([s1,s2,(s1-s2)/s1,s1/s3]))
    # #r = 1/np.sqrt(len(theta)*obj.shape[2])
    # a = slv.fwd_tomo(obj.complexform)
    # a = slv.exptomo(a)
    # b = slv.fwd_ptycho(a)
    # aa = slv.adj_ptycho(b,obj.complexform)
    # s1 = 0+1j*0
    # s3 = 0+1j*0

    # s1 = np.sum(np.complex64(a)*np.conj(np.complex64(aa)))
    # for k in range(len(theta)):
    #     #print([a[k].shape,aa[k].shape])
    #     s2+=np.sum(np.complex64(b[k])*np.conj(np.complex64(b[k])))
    # for k in range(len(theta)):
    #     s3+=np.sum(np.complex64(aa[k])*np.conj(np.complex64(aa[k])))

    # print("Adjoint and normalization test ptycho: "+str([s1,s2,(s1-s2)/s1,s1/s3]))

    # Project.
    psis = slv.fwd_tomo(obj.complexform)
    psis = slv.exptomo(psis)
    # Propagate.
    data = slv.fwd_ptycho(psis)
    data = np.abs(data)**2
    #data = np.random.poisson(data).astype('float32')

# Init.
    tau=1e-12
    reg_term=0
    h = np.ones(psis.shape, dtype='complex64')
    psi = np.ones(psis.shape, dtype='complex64')
    lamd = np.zeros(psi.shape, dtype='complex64')
    y = np.zeros([3,*obj.shape], dtype='complex64')
    mu = np.zeros([3,*obj.shape], dtype='complex64')
    x = objects.Object(np.zeros(obj.shape, dtype='float32'), np.zeros(
        obj.shape, dtype='float32'), voxelsize)
    slv.admm(data, h, psi, y, lamd, x, rho, mu, tau, gamma, eta, piter, titer,reg_term)

 # Init.
    tau=1e-2
    reg_term=0
    h = np.ones(psis.shape, dtype='complex64')
    psi = np.ones(psis.shape, dtype='complex64')
    lamd = np.zeros(psi.shape, dtype='complex64')
    y = np.zeros([3,*obj.shape], dtype='complex64')
    mu = np.zeros([3,*obj.shape], dtype='complex64')
    x = objects.Object(np.zeros(obj.shape, dtype='float32'), np.zeros(
        obj.shape, dtype='float32'), voxelsize)
    slv.admm(data, h, psi, y, lamd, x, rho, mu, tau, gamma, eta, piter, titer,reg_term)
 
 # Init.
    tau=1e-2
    reg_term=1
    h = np.ones(psis.shape, dtype='complex64')
    psi = np.ones(psis.shape, dtype='complex64')
    lamd = np.zeros(psi.shape, dtype='complex64')
    y = np.zeros([3,*obj.shape], dtype='complex64')
    mu = np.zeros([3,*obj.shape], dtype='complex64')
    x = objects.Object(np.zeros(obj.shape, dtype='float32'), np.zeros(
        obj.shape, dtype='float32'), voxelsize)
    slv.admm(data, h, psi, y, lamd, x, rho, mu, tau, gamma, eta, piter, titer,reg_term)
