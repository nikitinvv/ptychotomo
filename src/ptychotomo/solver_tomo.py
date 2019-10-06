"""Module for tomography."""

import warnings

import cupy as cp
import numpy as np
import dxchange
import signal
import sys
from ptychotomo.radonusfft import radonusfft

warnings.filterwarnings("ignore")


class SolverTomo(object):
    def __init__(self, theta, ntheta, nz, n, pnz):

        self.ntheta = ntheta
        self.nz = nz
        self.n = n
        self.pnz = pnz
        # create class for the tomo transform
        self.cl_tomo = radonusfft(self.ntheta, self.pnz, self.n)
        self.cl_tomo.setobj(theta.ctypes.data)

        # GPU memory deallocation with ctrl+C, ctrl+Z signals
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Free gpu memory after SIGINT, SIGSTSTP (destructor)"""
        self = []
        sys.exit(0)

    def fwd_tomo(self, u):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.cl_tomo.fwd(res.data.ptr, u.data.ptr)
        return res

    def adj_tomo(self, data):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.cl_tomo.adj(res.data.ptr, data.data.ptr)
        return res

    def fwd_reg(self, u):
        """Forward operator for regularization (J)"""
        res = cp.zeros([3, *u.shape], dtype='complex64')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        return res

    def adj_reg(self, gr):
        """Adjoint operator for regularization (J^*)"""
        res = cp.zeros(gr.shape[1:], dtype='complex64')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res *= -1
        return res

    def line_search(self, minf, gamma, Ru, gu, Rd, gd):
        """Line search for the step sizes gamma"""
        while(minf(Ru, gu)-minf(Ru+gamma*Rd, gu+gamma*gd) < 0 and gamma > 1e-32):
            gamma *= 0.5
        if(gamma <= 1e-32):  # direction not found
            gamma = 0
            warnings.warn("Line search failed for conjugate gradient.")
        return gamma

    # Conjugate gradients tomography
    def cg_tomo(self, xi0, u, titer, rho, tau, K, xi1):
        """CG solver for rho||KRu-xi0||_2+tau||\nablau-xi1||_2"""
        # minimization functional
        def minf(KRu, gu):
            f = rho*cp.linalg.norm(KRu-xi0)**2
            if(tau is not 0):  # additional L2 term + tau||\nabla u - xi1||_2
                f += tau*cp.linalg.norm(gu-xi1)**2
            return f
        for i in range(titer):
            KRu = K*self.fwd_tomo(u)
            grad = rho * self.adj_tomo(cp.conj(K)*(KRu-xi0)) / \
                (self.ntheta * self.n/2)
            if(tau is not 0):  # gradient for the additional L2 term: tau||\nabla u - xi1||_2
                grad += tau*self.adj_reg(self.fwd_reg(u)-xi1)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            grad0 = grad
            # line search
            gamma = self.line_search(
                minf, 1, KRu, self.fwd_reg(u), K*self.fwd_tomo(d), self.fwd_reg(d))
            # update step
            u = u + gamma*d

            # check convergence
            if (np.mod(i, 4) == 0):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(KRu, self.fwd_reg(u))))
        return u

    def cg_tomo_batch(self, xi0, init, titer, rho=1, tau=0, K=None, xi1=None):
        """CG solver for rho||KRu-xi0||_2+tau||\nabla u-xi1||_2 by z-slice partitions"""
        if (K is None):  # ignore K
            K = np.ones([1, self.nz, 1], dtype='float32')
        if (xi1 is None):  # term tau||\nablau-xi1||_2
            xi1 = np.zeros(self.nz, dtype='float32')

        u = init.copy()
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            xi0_gpu = cp.array(xi0[:, ids])
            K_gpu = cp.array(K[:, ids])
            xi1_gpu = cp.array(xi1[ids])
            # reconstruction
            u_gpu = self.cg_tomo(
                xi0_gpu, u_gpu, titer, rho, tau, K_gpu, xi1_gpu)
            # copy result to cpu
            u[ids] = u_gpu.get()
        return u

    # For data generation and adjoint test
    def fwd_tomo_batch(self, u):
        """Batch of Tomography transform (R)"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu)
            # copy result to cpu
            res[:, ids] = res_gpu.get()
        return res

    def adj_tomo_batch(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = np.zeros([self.nz, self.n, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            data_gpu = cp.array(data[:, ids])
            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res
