# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import dxchange
import tomopy
import radonusfft
import ptychofft
import numpy as np
import objects
import warnings

warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, prb, scan, scanax, scanay, theta, det, voxelsize, energy, tomoshape):
        self.prb = prb
        self.scan = scan
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
        print(self.objshape)
        print(self.ptychoshape)

    def wavenumber(self):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    # exp representation of R
    def exptomo(self, psi):
        return np.exp(1j * psi * self.voxelsize * self.wavenumber())

    # log represnetation of R
    def logtomo(self, psi):
        return -1j / self.wavenumber() * np.log(psi) / self.voxelsize

    # Radon transform (R )
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
        r = 1/np.sqrt(data.shape[0]*data.shape[1]/2)
        res = init.complexform
        for i in range(niter):
            tmp0 = self.fwd_tomo(res)
            tmp = self.adj_tomo(2*(tmp0-data))
            res = res - eta*r*r*tmp
        return objects.Object(res.imag, res.real, self.voxelsize)

    # Gradient descent ptychography

    def grad_ptycho(self, data, init, niter, rho, gamma, hobj, lamd):
        psi = init.copy()

        # psi2 = init.copy()
        # for i in range(niter):
        #     tmp = self.fwd_ptycho(psi2)
        #     tmp = self.update_amp(tmp, data)
        #     upd1 = self.adj_ptycho(tmp)
        #     upd2 = self.adjfwd_prb(psi2)
        #     psi2 = (1 - rho*gamma) * psi2 + rho*gamma * \
        #         (hobj - lamd/rho) + (gamma / 2) * (upd1-upd2) / self.maxint

        # whole scheme on gpu
        for k in range(0, self.tomoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.grad_ptycho(
                psi[ast:aend], data[ast:aend], hobj[ast:aend], lamd[ast:aend], rho, gamma, self.maxint, niter)

        # print(np.linalg.norm(psi2-psi))
        # print(np.linalg.norm(psi))
        return psi

    #@profile
    # ADMM for ptycho-tomography problem
    def admm(self, data, h, psi, lamd, x, rho, gamma, eta, piter, titer, NITER):
        for m in range(NITER):
            psi0, x0 = psi, x
            # psi update
            psi = self.grad_ptycho(data, psi, piter, rho, gamma, h, lamd)
            #dxchange.write_tiff(psi.real,"../rec_ptycho/psir15")
            #dxchange.write_tiff(psi.imag,"../rec_ptycho/psii15")
            # x update
            x = self.grad_tomo(self.logtomo(psi+lamd/rho), titer, x, rho, eta)
            # h update
            h = self.exptomo(self.fwd_tomo(x.complexform))
            # lambda update
            lamd = lamd + rho * (psi - h)

            # # check convergence of the Lagrangian
            # if (np.mod(m, 1) == 0):
            #     terms = np.zeros(4, dtype='float32')  # ignore imag part
            #     terms[0] = 0.5 * np.linalg.norm(
            #         np.abs(self.fwd_ptycho(psi))-np.sqrt(data))**2
            #     terms[1] = np.sum(np.conj(lamd)*(psi-h))
            #     terms[2] = 0.5*rho*np.linalg.norm(psi-h)**2
            #     terms[3] = np.sum(terms[0:3])

            #     print("%d) Lagrangian terms:  %.2e %.2e %.2e %.2e" %
            #           (m, terms[0], terms[1], terms[2], terms[3]))
            # check convergence of psi and x
            #print("%d) Conv psi, x:  %.2e %.2e" % (m, np.linalg.norm(psi-psi0),np.linalg.norm(x.complexform-x0.complexform)))

        return x
