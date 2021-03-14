"""Module for tomography."""

import cupy as cp
import numpy as np
import threading
import concurrent.futures as cf
from .radonusfft import radonusfft
from .utils import chunk
from functools import partial
import matplotlib
matplotlib.use('Agg')

class SolverTomo(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of pair slice partitions to process together
        simultaneously (multiple of nz)
    ngpus : int
        Number of gpus        
    """

    def __init__(self, theta, ntheta, nz, n, pnz, center, ngpus):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        # print(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus)
        if(nz % pnz > 0):
            print('Error, pnz is not a multiple of nz')
            exit()
        super().__init__(ntheta, pnz, n, center, theta.ctypes.data, ngpus)
        self.nz = nz

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u, gpu):
        """Radon transform (R)"""
        data = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        u = cp.ascontiguousarray(u)
        data = cp.ascontiguousarray(data)        
        self.fwd(data.data.ptr, u.data.ptr, gpu)
        return data

    def adj_tomo(self, data, gpu, filter=False):
        """Adjoint Radon transform (R^*)"""
        u = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        u = cp.ascontiguousarray(u)
        data = cp.ascontiguousarray(data)
        # C++ wrapper, send pointers to GPU arrays
        self.adj(u.data.ptr, data.data.ptr, gpu, filter)
        return u

    def line_search(self, minf, gamma, Ru, Rd):
        """Line search for the step sizes gamma"""
        while(minf(Ru)-minf(Ru+gamma*Rd) < 0):
            gamma *= 0.5
        return gamma

    def line_search_ext(self, minf, gamma, Ru, Rd, gu, gd):
        """Line search for the step sizes gamma"""
        while(minf(Ru, gu)-minf(Ru+gamma*Rd, gu+gamma*gd) < 0):
            gamma *= 0.5
        return gamma

    def fwd_tomo_batch(self, u):
        """Batch of Tomography transform (R)"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for ids in chunk(range(self.nz), self.pnz):
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu, 0)
            # copy result to cpu
            res[:, ids] = res_gpu.get()
        return res

    def adj_tomo_batch(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = np.zeros([self.nz, self.n, self.n], dtype='complex64')
        for ids in chunk(range(self.nz), self.pnz):
            # copy data part to gpu
            data_gpu = cp.array(data[:, ids])

            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu, 0)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def cg_tomo(self, xi0, u, titer, gpu=0, dbg=False):
        """CG solver for 1 slice partition"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo(u, gpu)
            grad = self.adj_tomo(Ru-xi0, gpu) / \
                (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo(d, gpu)
            gamma = 0.5*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u

    def cg_tomo_multi_gpu(self, xi0, u, titer, lock, ids):
        """Pick GPU, copy data, run reconstruction"""
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        u_gpu = cp.array(u[ids])
        xi0_gpu = cp.array(xi0[:, ids])
        # reconstruct
        u_gpu = self.cg_tomo(xi0_gpu, u_gpu, titer, gpu)
        u[ids] = u_gpu.get()

        BUSYGPUS[gpu] = 0

        return u[ids]

    def cg_tomo_batch(self, xi0, init, titer, dbg=False):
        """CG solver for rho||Ru-xi0||_2 by z-slice partitions"""
        u = init.copy()
        ids_list = chunk(range(self.nz), self.pnz)
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.cg_tomo_multi_gpu, xi0, u, titer, lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]
        cp.cuda.Device(0).use()
        return u




    def cg_tomo_ext(self, xi0, K, u, titer, gpu=0, dbg=False):
        """CG solver for 1 slice partition"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = K*self.fwd_tomo(u, gpu)
            grad = self.adj_tomo(cp.conj(K)*(Ru-xi0), gpu) / \
                (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = K*self.fwd_tomo(d, gpu)
            gamma = 0.5*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (1):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u

    def cg_tomo_ext_multi_gpu(self, xi0, K, u, titer, lock, ids):
        """Pick GPU, copy data, run reconstruction"""
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        u_gpu = cp.array(u[ids])
        xi0_gpu = cp.array(xi0[:, ids])
        K_gpu = cp.array(K[:, ids])
        # reconstruct
        u_gpu = self.cg_tomo_ext(xi0_gpu, K_gpu, u_gpu, titer, gpu)
        u[ids] = u_gpu.get()

        BUSYGPUS[gpu] = 0

        return u[ids]

    def cg_tomo_ext_batch(self, xi0, K, init, titer, dbg=False):
        """CG solver for rho||Ru-xi0||_2 by z-slice partitions"""
        u = init.copy()
        ids_list = chunk(range(self.nz), self.pnz)
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.cg_tomo_ext_multi_gpu, xi0, K, u, titer, lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]
        cp.cuda.Device(0).use()
        return u

    
    def grad_tomo(self, xi0, K, init, titer, igpu):
        """G solver for rho||Ru-xi0||_2 by z-slice partitions"""
        # minimization functional
        def minf(KRu, gu):
            f = cp.linalg.norm(KRu-xi0)**2            
            return f
        u = init.copy()
        minf1 = 1e12
        for i in range(titer):           
            KRu = K*self.fwd_tomo(u, igpu)
            grad = self.adj_tomo(cp.conj(K)*(KRu-xi0), igpu)/(self.ntheta * self.n)
            # update step
            u = u + 0.5*(-grad)
            # if(i%4==0):              
            #    print('t',i,minf(KRu, -1))
            minf0 = minf(KRu, None)                
            if(minf1 < minf0):
                print('error in tomo', minf0, minf1)
            minf1 = minf0
        return u

    def grad_tomo_multi_gpu(self, xi0, K, u, titer, lock, ids):
        """Pick GPU, copy data, run reconstruction"""
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        u_gpu = cp.array(u[ids])
        xi0_gpu = cp.array(xi0[:, ids])
        K_gpu = cp.array(K[:, ids])
        # reconstruct
        u_gpu = self.grad_tomo(xi0_gpu, K_gpu, u_gpu, titer, gpu)
        u[ids] = u_gpu.get()

        BUSYGPUS[gpu] = 0

        return u[ids]

    def grad_tomo_batch(self, xi0, K, init, titer, dbg=False):
        """CG solver for rho||Ru-xi0||_2 by z-slice partitions"""
        u = init.copy()
        ids_list = chunk(range(self.nz), self.pnz)
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.grad_tomo_multi_gpu, xi0, K, u, titer, lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]
        cp.cuda.Device(0).use()
        return u