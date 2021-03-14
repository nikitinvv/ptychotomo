from .solver_deform import SolverDeform
from .solver_tomo import SolverTomo

import numpy as np
import dxchange
import elasticdeform
import concurrent.futures as cf
from scipy import ndimage
from functools import partial


def gen_ang(numproj, nProj_per_rot):
    """Generate interlaced angles"""

    prime = 3
    pst = 0
    pend = 360
    seq = []
    i = 0
    sgn = 1  # for switching direction
    while len(seq) < numproj:
        b = i
        i += 1
        r = 0
        q = 1 / prime
        while (b != 0):
            a = np.mod(b, prime)
            r += (a * q)
            q /= prime
            b = np.floor(b / prime)
        r *= ((pend-pst) / nProj_per_rot)
        k = 0
        while (np.logical_and(len(seq) < numproj, k < nProj_per_rot)):
            if(sgn == 1):
                seq.append(pst + (r + k * (pend-pst) / nProj_per_rot))
            else:
                seq.append(pend-((1-r) + k * (pend-pst) / nProj_per_rot))
            k += 1
    return seq


def deform_data(u, theta, displacement0, k):
    """Deform object with respect to displacement0*(1-exp(t[k])) displacement,
    and computes its projection data"""

    print('deforming 3d frame', k)
    [nz,ne] = u.shape[:2]   
    displacement = displacement0*(1-np.exp(np.linspace(0, 1, len(theta))[k]))
    ud = elasticdeform.deform_grid(
        u, displacement, order=1, mode='mirror', crop=None, prefilter=True, axis=None)
    with SolverTomo(theta[k:k+1], 1, nz, ne, 32, ne/2, 1) as tslv:
        data = tslv.fwd_tomo_batch(ud)
    return data


def deform_data_batch(u, theta, displacement0):
    """Continuos deformation of the object in time t with respect to displacement0*(1-exp(t)) displacement,
    and compute projection data for each state"""

    [nz,ne] = u.shape[:2]
    res = np.zeros([len(theta), nz, ne], dtype='float32')
    with cf.ThreadPoolExecutor(32) as e:
        shift = 0
        for res0 in e.map(partial(deform_data, u, theta, displacement0), range(0, len(theta))):
            res[shift] = res0
            shift += 1
    return res


def cyl(r, c, h, rx, ry, rz, n):
    """Generate cylinders in 3D volume"""

    [x, y] = np.meshgrid(np.arange(-n//2, n//2), np.arange(-n//2, n//2))
    x = x/n*2
    y = y/n*2
    circ1 = ((x-c[1])**2+(y-c[0])**2 < r).astype('float32')
    circ2 = ((x-c[1])**2+(y-c[0])**2 < r-0.9/n).astype('float32')
    f = np.zeros([n, n, n], dtype='float32')
    f[n//2-h//2:n//2+h//2] = circ1-circ2
    f = ndimage.rotate(f, rx, axes=(1, 0), reshape=False, order=3)
    f = ndimage.rotate(f, ry, axes=(1, 2), reshape=False, order=3)
    f = ndimage.rotate(f, rz, axes=(2, 0), reshape=False, order=3)
    return f


def gen_cyl_data(n, ntheta, pprot, adef, noise=False):
    """Generate cylinders in 3D volume, deform them, compute projections, save to disk"""

    print('Start data generation')
    nz = n  # vertical object size
    ne = 3*n//2  # padded size

    # generate cylinders
    f = cyl(0.01, [0.1, 0.2], n, 30, 45, 30, n)
    f = f+1*cyl(0.01, [-0.1, -0.2], n, -30, -15, 30, n)
    f = f+1*cyl(0.01, [-0.3, -0.3], n, -30, -95, -40, n)
    f = f+cyl(0.01, [-0.4, 0.4], n, 15, 30, 90, n)
    f = f+1*cyl(0.01, [0.4, -0.45], n, 90, 30, 90, n)
    f = f+1*cyl(0.01, [0.2, -0.25], n, -90, 30, 15, n)
    f = f+1*cyl(0.01, [0.3, -0.15], n, -10, 110, -15, n)
    f[f > 1] = 1
    f[f < 0] = 0
    [x, y] = np.mgrid[-ne//2:ne//2, -ne//2:ne//2]
    circ = (2*x/ne)**2+(2*y/ne)**2 < 1
    fe = np.zeros([nz, ne, ne], dtype='float32')
    
    fe[:, ne//2-n//2:ne//2+n//2, ne//2-n//2:ne//2+n//2] = f
    if(noise):
        fe +=(ndimage.filters.gaussian_filter(np.random.random(fe.shape), 3, truncate=8).astype('float32')-0.5)*12

    f = (fe-np.min(fe))/(np.max(fe)-np.min(fe))*circ
 
    namepart = '_pprot'+str(pprot)+'_noise'+str(noise)
    dxchange.write_tiff(
        f, 'data/init_object'+namepart, overwrite=True)

    # generate angles
    theta = np.array(gen_ang(ntheta, pprot)).astype('float32')/360*np.pi*2

    # deform data
    points = [3, 3, 3]
    r = np.random.rand(3, *points)  # displacement in 3d

    # compute data without deformation
    with SolverTomo(theta, ntheta, nz, ne, 32, n/2+(ne-n)/2, 1) as tslv:
        data = tslv.fwd_tomo_batch(f)

    dxchange.write_tiff(
        data, 'data/data'+namepart, overwrite=True)

    # compute data with deformation
    data = deform_data_batch(f, theta, r*adef)[:, :, ne//2-n//2:ne//2+n//2]
    dxchange.write_tiff(
        data, 'data/deformed_data'+namepart, overwrite=True)

    # save angles, and displacement vectors
    np.save('data/theta'+namepart, theta)
    
    print('generated data have been written in data/')

def gen_data(f, theta):
    n = f.shape[2]
    ne=n
    with SolverTomo(theta, len(theta), 1, ne, 1, n/2+(ne-n)/2, 1) as tslv:
        data = tslv.fwd_tomo_batch(f)
    return data

