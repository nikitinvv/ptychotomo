
"""Module for 3D ptychography."""

import radonusfft
import ptychofft
import numpy as np
import cupy as cp
import objects
import warnings
import dxchange

warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, prb, scan, theta, det, voxelsize, energy, tomoshape, ptheta):
        self.voxelsize = voxelsize
        self.energy = energy
        self.scan = scan
        self.prb = prb
        self.ptheta = ptheta
        # shapes
        self.tomoshape = tomoshape
        self.objshape = [tomoshape[1], tomoshape[2], tomoshape[2]]
        self.ptychoshape = [tomoshape[0], scan.shape[2], det[0], det[1]]
        # ptychography angle partitions
        self.ptychoshapep = [tomoshape[0] //
                             ptheta, scan.shape[2], det[0], det[1]]
        self.tomoshapep = [tomoshape[0]//ptheta, tomoshape[1], tomoshape[2]]
        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta.data.ptr)
        # create class for the ptycho transform
        self.cl_ptycho = ptychofft.ptychofft(
            *self.tomoshapep, *self.ptychoshapep, prb.shape[0])
        # normalization coefficients
        self.coeftomo = 1 / \
            np.sqrt(self.tomoshape[0] *
                    self.tomoshape[2]/2*2/3).astype('float32')
        self.coefptycho = 1/cp.abs(prb).max().get()
        self.coefdata = 1 / \
            (self.ptychoshapep[2]*self.ptychoshapep[3]
             * (cp.abs(prb)**2).max().get())

    def mlog(self, psi):
        res = psi.copy()
        res[cp.abs(res) < 1e-32] = 1e-32
        res = cp.log(res)
        return res

    # Wave number index
    def wavenumber(self):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    # Exp representation of projections, exp(i\nu\psi)
    def exptomo(self, psi):
        return cp.exp(1j*psi * self.voxelsize * self.wavenumber()/self.coeftomo)

    # Log representation of projections, -i/\nu log(psi)
    def logtomo(self, psi):
        return -1j / self.wavenumber() * self.mlog(psi) / self.voxelsize*self.coeftomo

    # Radon transform (R)
    def fwd_tomo(self, psi):
        res = cp.zeros(self.tomoshape, dtype='complex64', order='C')
        self.cl_tomo.fwd(res.data.ptr, psi.data.ptr)
        res *= self.coeftomo  # normalization
        return res

    # Adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res = cp.zeros(self.objshape, dtype='complex64', order='C')
        self.cl_tomo.adj(res.data.ptr, data.data.ptr)
        res *= self.coeftomo  # normalization
        return res

    # Ptychography transform (FQ)
    def fwd_ptycho(self, psi):
        res = cp.zeros(self.ptychoshapep, dtype='complex64', order='C')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr)
        res *= self.coefptycho  # normalization
        return res

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data):
        res = cp.zeros(self.tomoshapep, dtype='complex64', order='C')
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr)
        res *= self.coefptycho  # normalization
        return res

    # Forward operator for regularization (J)
    def fwd_reg(self, u):
        res = cp.zeros([3, *self.objshape], dtype='complex64', order='C')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        res *= 2/np.sqrt(3)  # normalization
        return res

    # Adjoint operator for regularization (J^*)
    def adj_reg(self, gr):
        res = cp.zeros(self.objshape, dtype='complex64', order='C')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res *= -2/np.sqrt(3)  # normalization
        return res

    # xi0,K, and K for linearization of the tomography problem
    def takexi(self, psi, phi, lamd, mu, rho, tau):

        r = self.prb.shape[0]/2
        m1 = cp.mean(
            cp.angle(psi[:, psi.shape[1]/2-r:psi.shape[1]/2+r, r:2*r]))
        m2 = cp.mean(cp.angle(
            psi[:, psi.shape[1]/2-r:psi.shape[1]/2+r, psi.shape[2]-2*r:psi.shape[2]-r]))
        pshift = (m1+m2)/2
        t = psi-lamd/rho
        t *= cp.exp(-1j*pshift)

        K = 1j*self.voxelsize * self.wavenumber()*t/self.coeftomo
        K = K/cp.amax(cp.abs(K))  # normalization
        xi0 = K*(-1j*(self.mlog(t)) /
                 (self.voxelsize * self.wavenumber()))*self.coeftomo
        xi1 = phi-mu/tau
        return xi0, xi1, K, pshift

    # Line search for the step sizes gamma
    def line_search(self, minf, gamma, u, fu, d, fd):
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-8):
            gamma *= 0.5
        if(gamma <= 1e-8):  # direction not found
            gamma = 0
        return gamma

    # Conjugate gradients tomography
    def cg_tomo(self, xi0, xi1, K, init, rho, tau, titer):
        # minimization functional
        def minf(KRu, gu): return rho*cp.linalg.norm(KRu-xi0)**2 + \
            tau*cp.linalg.norm(gu-xi1)**2
        u = init.copy()
        gamma = 8  # init gamma as a large value
        for i in range(titer):
            KRu = K*self.fwd_tomo(u)
            gu = self.fwd_reg(u)
            grad = rho*self.adj_tomo(cp.conj(K)*(KRu-xi0)) + \
                tau*self.adj_reg(gu-xi1)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    ((cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            gamma = self.line_search(
                minf, gamma, KRu, gu, K*self.fwd_tomo(d), self.fwd_reg(d))
            # update step
            u = u + gamma*d
        return u

    # Conjugate gradients for ptychography
    def cg_ptycho(self, data, init, h, lamd, rho, piter, model):
        # minimization functional
        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * self.mlog(cp.abs(fpsi)))
            f += rho*cp.linalg.norm(h-psi+lamd/rho)**2
            return f

        psi = init.copy()
        gamma = 8  # init gamma as a large value
        for i in range(piter):
            fpsi = self.fwd_ptycho(psi)
            if model == 'gaussian':
                grad = self.adj_ptycho(
                    fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)))
            elif model == 'poisson':
                grad = self.adj_ptycho(fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32))
            grad -= rho*(h - psi + lamd/rho)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    ((cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            fd = self.fwd_ptycho(d)
            gamma = self.line_search(minf, gamma, psi, fpsi, d, fd)
            psi = psi + gamma*d
            # print(i,minf(psi,fpsi))
        return psi

    # Solve ptycho by angles partitions
    def cg_ptycho_batch(self, data, init, h, lamd, rho, piter, model):
        psi = init.copy()
        for k in range(0, self.ptheta):
            ids = np.arange(k*self.tomoshapep[0], (k+1)*self.tomoshapep[0])
            self.cl_ptycho.setobj(
                self.scan[:, ids].data.ptr, self.prb.data.ptr)
            psi[ids] = self.cg_ptycho(
                cp.array(data[ids]), psi[ids], h[ids], lamd[ids], rho, piter, model)
        return psi

    # Regularizer problem
    def solve_reg(self, u, mu, tau, alpha):
        z = self.fwd_reg(u)+mu/tau
        # Soft-thresholding
        za = cp.sqrt(cp.real(cp.sum(z*cp.conj(z), 0)))
        z[:, za <= alpha/tau] = 0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/(za[za > alpha/tau])
        return z

    # Update rho, tau for a faster convergence
    def update_penalty(self, psi, h, h0, phi, e, e0, rho, tau):
        # rho
        r = cp.linalg.norm(psi - h)**2
        s = cp.linalg.norm(rho*(h-h0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        # tau
        r = cp.linalg.norm(phi - e)**2
        s = cp.linalg.norm(tau*(e-e0))**2
        if (r > 10*s):
            tau *= 2
        elif (s > 10*r):
            tau *= 0.5
        return rho, tau

    # Lagrangian terms for monitoring convergence
    def take_lagr(self, psi, phi, data, h, e, lamd, mu, alpha, rho, tau, model):
        lagr = cp.zeros(7, dtype="float32")
        # Lagrangian ptycho part by angles partitions
        for k in range(0, self.ptheta):
            ids = np.arange(k*self.tomoshapep[0], (k+1)*self.tomoshapep[0])
            self.cl_ptycho.setobj(
                self.scan[:, ids].data.ptr, self.prb.data.ptr)
            fpsi = self.fwd_ptycho(psi[ids])
            datap = cp.array(data[ids])
            if (model == 'poisson'):
                lagr[0] += cp.sum(cp.abs(fpsi)**2-2*datap *
                                  self.mlog(cp.abs(fpsi)))
            if (model == 'gaussian'):
                lagr[0] += cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(datap))**2
        lagr[1] = alpha*cp.sum(np.sqrt(cp.real(cp.sum(phi*cp.conj(phi), 0))))
        lagr[2] = 2*cp.sum(cp.real(cp.conj(lamd)*(h-psi)))
        lagr[3] = rho*cp.linalg.norm(h-psi)**2
        lagr[4] = 2*cp.sum(np.real(cp.conj(mu)*(e-phi)))
        lagr[5] = tau*cp.linalg.norm(e-phi)**2
        lagr[6] = cp.sum(lagr[0:5])
        return lagr

    # ADMM for ptycho-tomography problem
    def admm(self, data, h, e, psi, phi, lamd, mu, u, alpha, piter, titer, NITER, model):
        data = data.copy()*self.coefdata  # normalization
        # init penalties
        rho, tau = 1, 1
        # Lagrangian for each iter
        lagr = cp.zeros([NITER, 7], dtype="float32")
        lagr0 = self.take_lagr(psi, phi, data, h, e, lamd,
                               mu, tau, rho, alpha, model)
        for m in range(NITER):
            # keep previous iteration for penalty updates
            h0, e0 = h, e
            psi = self.cg_ptycho_batch(data, psi, h, lamd, rho, piter, model)
            # tomography problem
            xi0, xi1, K, pshift = self.takexi(psi, phi, lamd, mu, rho, tau)
            u = self.cg_tomo(xi0, xi1, K, u, rho, tau, titer)
            # regularizer problem
            phi = self.solve_reg(u, mu, tau, alpha)
            # h,e updates
            h = self.exptomo(self.fwd_tomo(u))*cp.exp(1j*pshift)
            e = self.fwd_reg(u)
            # lambda, mu updates
            lamd = lamd + rho * (h-psi)
            mu = mu + tau * (e-phi)
            # update rho, tau for a faster convergence
            rho, tau = self.update_penalty(
                psi, h, h0, phi, e, e0, rho, tau)
            # Lagrangians difference between two iterations
            if (np.mod(m, 10) == 0):
                lagr[m] = self.take_lagr(
                    psi, phi, data, h, e, lamd, mu, alpha, rho, tau, model)
                print("%d/%d) rho=%.2e, tau=%.2e, Lagr terms diff:  %.2e %.2e %.2e %.2e %.2e %.2e, Sum: %.2e" %
                      (m, NITER, rho, tau, *(lagr0-lagr[m])))
                lagr0 = lagr[m]
                name = 'reg'+str(model)+str(piter)+str(titer) + \
                    str(NITER)+str(np.amax(data))
                dxchange.write_tiff(
                    u[u.shape[0]//2].imag.get(),  'betap/beta'+name)
                dxchange.write_tiff(
                    u[u.shape[0]//2].real.get(),  'deltap/delta'+name)
                dxchange.write_tiff(
                    cp.abs(psi).get(),  'psip/psiamp'+name)
                dxchange.write_tiff(
                    cp.angle(psi).get(),  'psip/psiangle'+name)

        lagrr = self.take_lagr(psi, phi, data, h, e, lamd,
                               mu, tau, rho, alpha, model)
        print(lagrr)
        return u, psi, lagrr
