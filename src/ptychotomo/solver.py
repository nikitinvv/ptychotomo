"""Module for 3D ptychography."""

import warnings

import cupy as cp
import numpy as np
import dxchange
import signal
import sys
from ptychotomo.radonusfft import radonusfft
from ptychotomo.ptychofft import ptychofft


warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, nscan, nprb, theta, det, voxelsize, energy, ntheta, nz, n, ptheta, pnz):

        self.voxelsize = voxelsize
        self.energy = energy
        self.nscan = nscan
        self.nprb = nprb
        self.ntheta = ntheta
        self.nz = nz
        self.n = n
        self.nscan = nscan
        self.ndety = det[0]
        self.ndetx = det[1]
        self.nprb = nprb
        self.ptheta = ptheta
        self.pnz = pnz

        # create class for the tomo transform
        self.cl_tomo = radonusfft(self.ntheta, self.pnz, self.n)
        self.cl_tomo.setobj(theta.data.ptr)
        # create class for the ptycho transform
        self.cl_ptycho = ptychofft(
            self.ptheta, self.nz, self.n, self.nscan, self.ndetx, self.ndety, self.nprb)
        # normalization coefficients
        self.coeftomo = 1 / np.sqrt(self.ntheta * self.n/2).astype('float32')

        # GPU memory deallocation with ctrl+C, ctrl+Z sygnals
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Free gpu memory after SIGINT, SIGSTSTP (destructor)"""
        self = []
        sys.exit(0)

    def mlog(self, psi):
        res = psi.copy()
        res[cp.abs(psi) < 1e-32] = 1e-32
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
    def fwd_tomo(self, u):
        res = cp.zeros([self.ntheta, self.pnz, self.n],
                       dtype='complex64', order='C')
        self.cl_tomo.fwd(res.data.ptr, u.data.ptr)
        res *= self.coeftomo  # normalization
        return res

    # Adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res = cp.zeros([self.pnz, self.n, self.n],
                       dtype='complex64', order='C')
        self.cl_tomo.adj(res.data.ptr, data.data.ptr)
        res *= self.coeftomo  # normalization
        return res

    # Batch of Radon transform (R)
    def fwd_tomo_batch(self, u):
        res = cp.zeros([self.ntheta, self.nz, self.n],
                       dtype='complex64', order='C')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            res[:, ids] = self.fwd_tomo(u[ids])
        return res

    # Batch of adjoint Radon transform (R^*)
    def adj_tomo_batch(self, data):
        res = cp.zeros([self.nz, self.n, self.n], dtype='complex64', order='C')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            res[ids] = self.adj_tomo(data[:, ids])
        return res

    # Forward operator for regularization (J)
    def fwd_reg(self, u):
        res = cp.zeros([3, *u.shape], dtype='complex64', order='C')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        res *= 2/np.sqrt(3)  # normalization
        return res

    # Adjoint operator for regularization (J^*)
    def adj_reg(self, gr):
        res = cp.zeros(gr.shape[1:], dtype='complex64', order='C')
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
        # bg subtraction parameters
        r = self.nprb/2
        m1 = cp.mean(
            cp.angle(psi[:, :, r:2*r]))
        m2 = cp.mean(cp.angle(
            psi[:, :, psi.shape[2]-2*r:psi.shape[2]-r]))
        pshift = (m1+m2)/2

        t = psi-lamd/rho
        t *= cp.exp(-1j*pshift)
        logt = self.mlog(t)

        # K, xi0, xi1
        K = 1j*self.voxelsize * self.wavenumber()*t/self.coeftomo
        K = K/cp.amax(cp.abs(K))  # normalization
        xi0 = K*(-1j*(logt) /
                 (self.voxelsize * self.wavenumber()))*self.coeftomo
        xi1 = phi-mu/tau
        return xi0, xi1, K, pshift

    # Conjugate gradients tomography
    def cg_tomo(self, xi0, xi1, K, init, rho, tau, titer):
        # minimization functional
        def minf(KRu, gu):
            return rho*cp.linalg.norm(KRu-xi0)**2#+tau*cp.linalg.norm(gu-xi1)**2
        u = init.copy()
        gamma = 2  # init gamma as a large value
        for i in range(titer):
            KRu = K*self.fwd_tomo_batch(u)
            gu = self.fwd_reg(u)
            grad = rho*self.adj_tomo_batch(cp.conj(K)*(KRu-xi0)) #+ \
                #tau*self.adj_reg(gu-xi1)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    ((cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            gamma = self.line_search(
                minf, gamma, KRu, gu, K*self.fwd_tomo_batch(d), self.fwd_reg(d))
            # update step
            u = u + gamma*d

             # check convergence
            if (np.mod(i, 1) == 0):                
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(KRu, gu)))
        return u

    def fwd_ptycho(self, psi, scan, prb):
        """Ptychography transform (FQ)"""
        res = cp.zeros([self.ptheta, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr,
                           scan.data.ptr, prb.data.ptr)  # C++ wrapper, send pointers to GPU arrays
        return res

    def fwd_ptycho_batch(self, psi, scan, prb):
        """Batch of Ptychography transform (FQ)"""
        data = np.zeros([self.ntheta, self.nscan, self.ndety,
                         self.ndetx], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data0 = cp.abs(self.fwd_ptycho(
                psi[ids], scan[:, ids], prb[ids]))**2  # compute part on GPU
            data[ids] = data0.get()  # copy to CPU
        return data

    def adj_ptycho(self, data, scan, prb):
        """Adjoint ptychography transform (Q*F*)"""
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64')
        flg = 0  # compute adjoint operator with respect to object
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr,
                           scan.data.ptr, prb.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    def adj_ptycho_prb(self, data, scan, psi):
        """Adjoint ptychography probe transform (O*F*), object is fixed"""
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64')
        flg = 1  # compute adjoint operator with respect to probe
        self.cl_ptycho.adj(psi.data.ptr, data.data.ptr,
                           scan.data.ptr, res.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    def line_search(self, minf, gamma, u, fu, d, fd):
        """Line search for the step sizes gamma"""
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-32):
            gamma *= 0.5
        if(gamma <= 1e-32):  # direction not found
            gamma = 0
            warnings.warn("Line search failed for conjugate gradient.")
        return gamma

    def cg_ptycho(self, data, psi, scan, prb, h, lamd, rho, piter,  model, recover_prb):
        """Conjugate gradients for ptychography"""
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim
        # minimization functional

        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * cp.log(cp.abs(fpsi)+1e-32))
           # f += rho*cp.linalg.norm(h-psi+lamd/rho)**2
            return f

        # initial gradient step
        gammapsi = 1
        gammaprb = 1
        print("# congujate gradient parameters\n"
              "iteration, step size object, step size probe, function min")  # csv column headers

        for k in range(piter):
            # 1) object retrieval subproblem with fixed probe
            # forward operator
            fpsi = self.fwd_ptycho(psi, scan, prb)
            # take gradient
            if model == 'gaussian':
                gradpsi = self.adj_ptycho(
                    fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), scan, prb)/(cp.max(cp.abs(prb))**2)
            elif model == 'poisson':
                gradpsi = self.adj_ptycho(
                    fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32), scan, prb)/(cp.max(cp.abs(prb))**2)
            #gradpsi -= rho*(h - psi + lamd/rho)
            # Dai-Yuan direction
            if k == 0:
                dpsi = -gradpsi
            else:
                dpsi = -gradpsi+cp.linalg.norm(gradpsi)**2 / \
                    ((cp.sum(cp.conj(dpsi)*(gradpsi-gradpsi0))))*dpsi
            gradpsi0 = gradpsi
            # line search
            fdpsi = self.fwd_ptycho(dpsi, scan, prb)
            if(recover_prb):
                # reset gamma
                gammapsi = 1
            gammapsi = self.line_search(
                minf, gammapsi, psi, fpsi, dpsi, fdpsi)
            # update psi
            psi = psi + gammapsi*dpsi

            if(recover_prb):
                # 2) probe retrieval subproblem with fixed object
                # forward operator
                fprb = self.fwd_ptycho(psi, scan, prb)
                # take gradient
                if model == 'gaussian':
                    gradprb = self.adj_ptycho_prb(
                        fprb-cp.sqrt(data)*cp.exp(1j*cp.angle(fprb)), scan, psi)/cp.max(cp.abs(psi))**2/self.nscan
                elif model == 'poisson':
                    gradprb = self.adj_ptycho_prb(
                        fprb-data*fprb/(cp.abs(fprb)**2+1e-32), scan, psi)/cp.max(cp.abs(psi))**2/self.nscan
                # Dai-Yuan direction
                if (k == 0):
                    dprb = -gradprb
                else:
                    dprb = -gradprb+cp.linalg.norm(gradprb)**2 / \
                        ((cp.sum(cp.conj(dprb)*(gradprb-gradprb0))))*dprb
                gradprb0 = gradprb
                # line search
                fdprb = self.fwd_ptycho(psi, scan, dprb)
                gammaprb = self.line_search(
                    minf, 1, psi, fprb, psi, fdprb)  # start with gammaprb = 1 on each iteration
                # update prb
                prb = prb + gammaprb*dprb

            # check convergence
            if (np.mod(k, 1) == 0):
                fpsi = self.fwd_ptycho(psi, scan, prb)
                print("%4d, %.3e, %.3e, %.7e" %
                      (k, gammapsi, gammaprb, minf(psi, fpsi)))

        return psi, prb

    def cg_ptycho_batch(self, data, initpsi, scan, initprb, h, lamd, rho, piter, model, recover_prb):
        """Solve ptycho by angles partitions."""
        assert initprb.ndim == 3, "prb needs 3 dimensions, not %d" % initprb.ndim

        psi = initpsi.copy()
        prb = initprb.copy()

        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            psi[ids], prb[ids] = self.cg_ptycho(
                datap, psi[ids], scan[:, ids], prb[ids], h[ids], lamd[ids], rho, piter, model, recover_prb)
        return psi, prb

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
    def take_lagr(self, psi, scan, prb, phi, data, h, e, lamd, mu, alpha, rho, tau, model):
        lagr = cp.zeros(7, dtype="float32")
        # Lagrangian ptycho part by angles partitions
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            fpsi = self.fwd_ptycho(psi[ids], scan[:, ids], prb[ids])
            datap = cp.array(data[ids])
            if (model == 'poisson'):
                lagr[0] += cp.sum(cp.abs(fpsi)**2-2*datap *
                                  self.mlog(cp.abs(fpsi))-(datap-2*datap*self.mlog(cp.sqrt(datap))))
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
    def admm(self, data, h, e, psi, scan, prb, phi, lamd, mu, u, alpha, piter, titer, niter, model, retrieve_prb):
        rho = 0.5
        tau = 0.5
        for m in range(niter):
            # keep previous iteration for penalty updates
            h0, e0 = h, e
            psi, prb = self.cg_ptycho_batch(
                data, psi, scan, prb,  h, lamd, rho, piter, model, retrieve_prb)
            # # tomography problem
            xi0, xi1, K, pshift = self.takexi(psi, phi, lamd, mu, rho, tau)
            u = self.cg_tomo(xi0, xi1, K, u, rho, tau, titer)
            # # regularizer problem
            # phi = self.solve_reg(u, mu, tau, alpha)
            # # h,e updates
            # h = self.exptomo(self.fwd_tomo_batch(u))*cp.exp(1j*pshift)
            # e = self.fwd_reg(u)
            # # lambda, mu updates
            # lamd = lamd + rho * (h-psi)
            # mu = mu + tau * (e-phi)
            # # update rho, tau for a faster convergence
            # rho, tau = self.update_penalty(
            #     psi, h, h0, phi, e, e0, rho, tau)
            # # Lagrangians difference between two iterations
            # if (np.mod(m, 10) == 0):
            #     lagr = self.take_lagr(
            #         psi, scan, prb, phi, data, h, e, lamd, mu, alpha, rho, tau, model)
            #     print("%d/%d) rho=%.2e, tau=%.2e, Lagrangian terms:  %.2e %.2e %.2e %.2e %.2e %.2e, Sum: %.2e" %
            #           (m, niter, rho, tau, *lagr))
        return u, psi, prb
