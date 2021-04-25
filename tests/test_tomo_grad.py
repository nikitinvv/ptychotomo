import numpy as np
import dxchange
import ptychotomo

if __name__ == "__main__":
    
    # read object
    u = dxchange.read_tiff('data/init_object.tiff')
    nz, n, _ = u.shape

    # parameters
    center = n/2
    ntheta = 384
    ne = 3*n//2
    ngpus = 1
    pnz = nz//4
    theta = np.linspace(0, 4*np.pi, ntheta).astype('float32')
    # cg
    niter = 32
    init = u*0
    pprot = 192
    # simulate data
    with ptychotomo.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        data = tslv.fwd_tomo_batch(u)
        u = tslv.grad_tomo_batch(data, init, niter)

    dxchange.write_tiff(u, 'data/cg/u',overwrite=True)


    res  = ptychotomo.pcg(data, theta, pprot, pnz, center, ngpus, niter, padding=False)
    dxchange.write_tiff(res['u'], 'data/cg/upcg',overwrite=True)
