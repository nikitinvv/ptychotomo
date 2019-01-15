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
        self.ptychoshape = [theta.size, scanax.shape[1]
                            * scanay.shape[1], det.x, det.y]
        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta)
        # create class for the ptycho transform
        # number of angles for simultaneous processing by 1 gpu
        self.theta_gpu = tomoshape[0]//10
        self.cl_ptycho = ptychofft.ptychofft(self.theta_gpu, tomoshape[1], tomoshape[2],
                                             scanax.shape[1], scanay.shape[1], det.x, det.y, prb.size)

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

    # multiply by probe and adj probe (Q^*Q)
    def adjfwd_prb(self, psi):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        for k in range(0, self.tomoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.adjfwd_prb(res_gpu[ast:aend], psi[ast:aend])
        return res_gpu

    # Amplitude update in Gradient descent ptychography, f = sqrt(data) exp(1j * angle(f))
    def update_amp(self, init, data):
        for k in range(0, self.tomoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.update_amp(init[ast:aend], data[ast:aend])
        return init

    # Gradient descent tomography
    def grad_tomo(self, data, niter, init, rho, eta):
        # normalization coefficient
        r = 1/np.sqrt(data.shape[0]*data.shape[2]/2)

        # whole scheme on gpu
        # res_gpu = init.complexform
        # self.cl_tomo.grad_tomo(res_gpu, data, r*r*eta, niter)

        xdiff = np.zeros(niter, dtype="float32")
        # Alternative computation
        res = init.complexform
        for i in range(niter):
            tmp0 = self.fwd_tomo(res)
            tmp = self.adj_tomo(tmp0-data)
            xdiff[i] = np.linalg.norm(2*eta*r*r*tmp)
            res = res - 2*eta*r*r*tmp
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))
        return objects.Object(res.imag, res.real, self.voxelsize), xdiff

    # Gradient descent ptychography
    def grad_ptycho(self, data, init, niter, rho, gamma, hobj, lamd):
        # whole scheme on gpu
        psi = init.copy()
        for k in range(0, self.tomoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.grad_ptycho(
                psi[ast:aend], data[ast:aend], hobj[ast:aend], lamd[ast:aend], rho, gamma, self.maxint, niter)

        # Alternative computation
        # psi2 = init.copy()
        # for i in range(niter):
        #     tmp = self.fwd_ptycho(psi2)
        #     tmp = self.update_amp(tmp, data)
        #     upd1 = self.adj_ptycho(tmp)
        #     upd2 = self.adjfwd_prb(psi2)
        #     psi2 = (1 - rho*gamma) * psi2 + rho*gamma * \
        #         (hobj - lamd/rho) + (gamma / 2) * (upd1-upd2) / self.maxint
        # Compare results
        # print(np.linalg.norm(psi2-psi)/np.linalg.norm(psi))
        return psi
    # @profile
    # ADMM for ptycho-tomography problem

    def admm(self, data, h, psi, lamd, x, rho, gamma, eta, piter, titer, NITER):
        xdiff = np.zeros(NITER*titer, dtype="float32")
        res = np.zeros([NITER, 4], dtype="float32")
        for m in range(NITER):
            # keep previous iteration
            psi0, x0, h0, lamd0 = psi, x, h, lamd
            # psi update
            psi = self.grad_ptycho(data, psi, piter, rho, gamma, h, lamd)
            # x update
            tmp = self.logtomo(psi+lamd/rho)
            x, xdiff[m*titer: (m+1)*titer] = self.grad_tomo(tmp, titer, x, rho, eta)
            # h update
            h = self.exptomo(self.fwd_tomo(x.complexform))
            # lambda update
            lamd = lamd + rho * (psi - h)
            # if(rho < 1e-6):  # standard cases
            #     lamd = lamd*0
            #     psi = h
            # else: # rho update
            #     r = np.linalg.norm(psi - h)**2
            #     s = np.linalg.norm(rho*(h-h0))**2
            #     if (r > 10*s):
            #         rho *= 2
            #     elif (s > 10*r):
            #         rho /= 2

            # residuals
            if (np.mod(m, 1) == 0):
                # psi^{k+1}-h^{k+1}
                res[m,0] = np.linalg.norm(psi-h)
                # h{k+1}-h^k
                res[m,1] = np.linalg.norm(h-h0)
                # 0.5*(Q^*Q psi^{k+1} - 0.5*Q^*F^*(FQ psi^{k+1}/ |FQ psi^{k+1}|)d ) + lamd
                res[m,2] = np.linalg.norm(0.5*self.adjfwd_prb(psi)
                    -0.5*self.adj_ptycho(self.update_amp(self.fwd_ptycho(psi), data))+lamd)
                # R^*(Rx+i/\nu log(psi^{k+1}+lamd^{k}/rho))
                res[m,3]=np.linalg.norm(self.adj_tomo(self.fwd_tomo(
                    x.complexform)-self.logtomo(psi-lamd0/rho)))

                print("%d %f %.2e %.2e %.2e %.2e " %
                      (m, rho, res[m, 0], res[m, 1], res[m, 2], res[m, 3]))
        return x, xdiff, res
