import objects
import solver
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
    for m in range(len(theta)):
        s = objects.Scanner(shape, sx, sy, margin, offset=[offset[0], np.mod(offset[1]+a, sy)])
        scan.append(s)
        a += spiral
    return scan


if __name__ == "__main__":

    # Parameters.
    rho = 0.5
    gamma = 0.25
    eta = 0.25
    piter = 1
    titer = 1
    maxint = 1
    voxelsize = 1e-6
    energy = 5

    # Load a 3D object.
#    beta = dxchange.read_tiff('data/lego-imag.tiff')
#    delta = dxchange.read_tiff('data/lego-real.tiff')
    beta = tomopy.misc.phantom.shepp3d(64)*1e-3
    delta = tomopy.misc.phantom.shepp3d(64)*1e-3



    # Create object.
    obj = objects.Object(beta, delta, voxelsize)

    # Create probe.
    weights = gaussian(15, rin=0.8, rout=1.0)
    prb = objects.Probe(weights, maxint=maxint)

    # Detector parameters.
    #det = objects.Detector(63, 63)
    det = objects.Detector(31, 31)


    # Define rotation angles.
    theta = np.linspace(0, np.pi, 180)

    # Raster scan parameters for each rotation angle.
    scan = scanner3(theta, beta.shape, 6, 6, margin=[prb.size, prb.size], offset=[0, 0], spiral=1)
    # Project.
    #psis = solver.project(obj, theta, energy=5)
    psis = solver.fwd_tomo(obj.complexform, theta)*voxelsize
    psis = np.exp(1j * solver.wavenumber(energy) * psis)

    # Propagate.
    data = solver.propagate3(prb, psis, scan, theta, det)

    # Init.
    hobj = np.ones(psis.shape, dtype='complex')
    psi = np.ones(psis.shape, dtype='complex')
    lamd = np.zeros(psi.shape, dtype='complex')
    recobj = objects.Object(np.zeros(obj.shape), np.zeros(obj.shape), voxelsize)

    solver.admm(data,prb,scan,hobj,psi,lamd,recobj,theta,voxelsize,energy,rho,gamma,eta,piter,titer)

