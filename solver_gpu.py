# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import radonusfft
import ptychofft
import numpy as np
import objects
import warnings

warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, prb, scanax, scanay, theta, det, voxelsize, energy, tomoshape):
        self.prb = prb
        self.scanax = scanax
        self.scanay = scanay
        self.theta = theta
        self.det = det
        self.voxelsize = voxelsize
        self.energy = energy
        self.maxint = np.power(np.abs(prb.complex), 2).max().astype('float')
        self.tomoshape = tomoshape
        self.objshape = [tomoshape[1], tomoshape[2], tomoshape[2]]
        self.ptychoshape = [theta.size, scanax.shape[1], det.x, det.y]
        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta)
        # create class for the ptycho transform
        # number of angles for simultaneous processing by 1 gpu
        self.theta_gpu = tomoshape[0]//10
        self.cl_ptycho = ptychofft.ptychofft(self.theta_gpu, tomoshape[1], tomoshape[2],
                                             scanax.shape[1], det.x, det.y, prb.size)

    def wavenumber(self):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    # Exp representation of projections, exp(i\nu\psi)
    def exptomo(self, psi):
        return np.exp(1j * psi * self.voxelsize * self.wavenumber())

    # Log representation of projections, -i/\nu log(psi)
    def logtomo(self, psi):
        return -1j / self.wavenumber() * np.log(psi) / self.voxelsize

    # Radon transform (R)
    def fwd_tomo(self, psi):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        self.cl_tomo.fwd(res_gpu, psi)
        return res_gpu

    # adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res_gpu = np.zeros(self.objshape, dtype='complex64', order='C')
        self.cl_tomo.adj(res_gpu, data)
        return res_gpu

    # ptychography transform (FQ)
    def fwd_ptycho(self, psi):
        res_gpu = np.zeros(self.ptychoshape, dtype='complex64', order='C')
        for k in range(0, self.ptychoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.fwd(res_gpu[ast:aend], psi[ast:aend])
        return res_gpu

    # adjoint ptychography transfrorm (Q*F*)
    def adj_ptycho(self, data):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        for k in range(0, self.ptychoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.adj(res_gpu[ast:aend], data[ast:aend])
        return res_gpu

    # Amplitude update in Gradient descent ptychography, f = sqrt(data) exp(1j * angle(f))
    def update_amp(self, init, data):
        init_gpu = init.copy()
        for k in range(0, self.tomoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.update_amp(init_gpu[ast:aend], data[ast:aend])
        return init_gpu

    # forward operator for regularization (q)
    def fwd_reg(self, x):
        res = np.zeros([3, *self.objshape], dtype='complex64', order='C')
        res[0, :, :, :-1] = x[:, :, 1:]-x[:, :, :-1]
        res[1, :, :-1, :] = x[:, 1:, :]-x[:, :-1, :]
        res[2, :-1, :, :] = x[1:, :, :]-x[:-1, :, :]
        return res*2/np.sqrt(3)

    # adjoint operator for regularization (q^*)
    def adj_reg(self, gr):
        res = np.zeros(self.objshape, dtype='complex64', order='C')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res = -res
        return res*2/np.sqrt(3)

    # compute xi for the tomography problem
    def takexi(self, psi, phi, lamd, mu, rho, tau):
        xi0 = -1j*np.log(psi-lamd/rho)/(self.voxelsize * self.wavenumber())
        xi1 = phi-mu/tau
        xi2 = 1j*self.voxelsize * self.wavenumber()*(psi-lamd/rho)
        return xi0, xi1, xi2

    # @profile
    # Gradient descent tomography
    def grad_tomo(self, xi0, xi1, xi2, niter, init, rho, tau, eta):
        # normalization coefficient for KR
        r = 1/(xi0.shape[0]*xi0.shape[2]/2)/np.amax(np.abs(xi2)**2)
        res = init.complexform
        for i in range(niter):
            # R^*K^*K(Rx-xi_0)
            tmp0 = self.adj_tomo(np.conj(xi2)*xi2*(self.fwd_tomo(res)-xi0))
            tmp1 = self.adj_reg(self.fwd_reg(res)-xi1)
            res = res - eta*r*rho*tmp0 - eta*tau*tmp1
        return objects.Object(res.imag, res.real, self.voxelsize)

    # Gradient descent ptychography
    def grad_ptycho(self, data, init, niter, rho, gamma, hobj, lamd):
        # whole scheme on gpu
        # psi = init.copy()
        # for k in range(0, self.tomoshape[0]//self.theta_gpu):
        #     # process self.theta_gpu angles on 1gpu simultaneously
        #     ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
        #     self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
        #                           self.prb.complex)
        #     self.cl_ptycho.grad_ptycho(
        #         psi[ast:aend], data[ast:aend], hobj[ast:aend], lamd[ast:aend], rho, gamma, self.maxint, niter)
        psi = init.copy()
        for i in range(niter):
            tmp = self.fwd_ptycho(psi)
            tmp = self.adj_ptycho(tmp-self.update_amp(tmp, data))
            psi = psi - gamma*tmp/self.maxint + \
                gamma*rho/2*(hobj - psi + lamd/rho)
        return psi

    def ml_ptycho(self, data, init, niter, rho, gamma, hobj, lamd):
        psi = init.copy()
        for i in range(niter):
            tmp = self.fwd_ptycho(psi)
            tmp = self.adj_ptycho(tmp-data/np.conj(tmp))
            psi = psi - gamma*tmp/self.maxint + \
                gamma*rho/2*(hobj - psi + lamd/rho)
        return psi

    def solve_reg(self, x, mu, tau, alpha):
        z = self.fwd_reg(x)+mu/tau
        za = np.sqrt(np.sum(z*np.conj(z), 0))
        z[:, za <= alpha/tau] = 0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/za[za > alpha/tau]
        return z

    def update_penalty(self, rho, tau, psi, h, h0, phi, e, e0):
        # rho
        r = np.linalg.norm(psi - h)**2
        s = np.linalg.norm(rho*(h-h0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho /= 2

        # tau, alpha
        r = np.linalg.norm(phi - e)**2
        s = np.linalg.norm(tau*(e-e0))**2
        if (r > 10*s):
            tau *= 2
        elif (s > 10*r):
            tau /= 2
        return rho, tau

    # @profile
    # ADMM for ptycho-tomography problem
    def admm(self, data, h, e, psi, phi, lamd, mu, x, rho, tau, alpha, gamma, eta, piter, titer, NITER, type='grad'):
        res = np.zeros([NITER, 4], dtype="float32")
        for m in range(NITER):
            # keep previous iteration
            psi0, phi0, x0, h0, e0, lamd0, mu0 = psi, phi, x, h, e, lamd, mu
            # ptychography problem
            if type == 'ml':
                psi = self.ml_ptycho(data, psi, piter, rho, gamma, h, lamd)
            else:
                psi = self.grad_ptycho(data, psi, piter, rho, gamma, h, lamd)
            # tomography problem
            xi0, xi1, xi2 = self.takexi(psi, phi, lamd, mu, rho, tau)
            x = self.grad_tomo(xi0, xi1, xi2, titer, x, rho, tau, eta)
            # regularizer problem
            phi = self.solve_reg(x.complexform, mu, tau, alpha)
            # h update
            h = self.exptomo(self.fwd_tomo(x.complexform))
            # e update
            e = self.fwd_reg(x.complexform)
            # lambda update
            lamd = lamd + rho * (h-psi)
            # mu update
            mu = mu + tau * (e-phi)
            # update rho, tau, alpha for a faster convergence
            rho, tau = self.update_penalty(
                rho, tau, psi, h, h0, phi, e, e0)

            # if conventional approach
            if(rho < 1e-6):
                lamd = lamd*0
                psi = h
            if(tau < 1e-6):
                mu = mu*0
                phi = e

            # check convergence of the Lagrangian
            if (np.mod(m, 50) == 0):
                res = np.zeros(7, dtype='float32')
                res[0] = 0.5 * np.linalg.norm(
                    np.abs(self.fwd_ptycho(psi))-np.sqrt(data))**2
                res[1] = np.sum(np.conj(lamd)*(h-psi))
                res[2] = 0.5*rho*np.linalg.norm(h-psi)**2
                res[3] = alpha*np.sum(np.sqrt(np.sum(phi*np.conj(phi), 0)))
                res[4] = np.sum(
                    np.conj(mu)*(e-phi))
                res[5] = 0.5*tau * \
                    np.linalg.norm(e-phi)**2
                res[6] = np.sum(res[0:5])

                print("%d) rho=%.2e, tau=%.2e, Lagrangian terms:  %.2e %.2e %.2e %.2e %.2e %.2e %.2e" %
                      (m, rho, tau, res[0], res[1], res[2], res[3], res[4], res[5], res[6]))
        x.delta+=1.4e-5
        return x, psi, res

    def power_method(self, g):
        r = 1/(self.tomoshape[0]*self.tomoshape[2])
        x = self.adj_reg(g)*r
        for k in range(1, 10000):
            x = self.adj_tomo(self.fwd_tomo(x))*r
            s = np.linalg.norm(x)
            x = x/s
            print(np.sqrt(s))





    
    def grad_ptycho0(self, data, init, niter, gamma):        
        psi = init.copy()
        for i in range(niter):
            tmp = self.fwd_ptycho(psi)
            grad = self.adj_ptycho(tmp-self.update_amp(tmp, data))
            psi = psi - gamma*grad/self.maxint 
        return psi