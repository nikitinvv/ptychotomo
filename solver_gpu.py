# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import dxchange
import tomopy
import radonusfft
import ptychofft
import xraylib as xl
import numpy as np
import scipy as sp
import pyfftw
import shutil
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
        self.tomoshape = tomoshape
        self.objshape = [tomoshape[1], tomoshape[2], tomoshape[2]]
        self.ptychoshape = [theta.size, scanax.shape[1]
                            * scanay.shape[1], det.x, det.y]
        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta)
        # create class for the ptycho transform
        self.theta_gpu = tomoshape[0]//2 #number of angles for simultaneous processing by 1 gpu
        self.cl_ptycho = ptychofft.ptychofft(self.theta_gpu, tomoshape[1], tomoshape[2],
                                             scanax.shape[1], scanay.shape[1], det.x, det.y, prb.size)

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
        self.cl_tomo.fwd(res_gpu.view('float32'), psi.view('float32'))

        # pb = tomopy.project(psi.imag, self.theta, pad=False)
        # pd = tomopy.project(psi.real, self.theta, pad=False)
        # #r = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2)
        # res = (pd + 1j*pb)#*r
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))

        return res_gpu

    # adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res_gpu = np.zeros(self.objshape, dtype='complex64', order='C')
        self.cl_tomo.adj(res_gpu.view('float32'), data.view('float32'))

        # pb = tomopy.recon(np.imag(data), self.theta, algorithm='fbp')
        # pd = tomopy.recon(np.real(data), self.theta, algorithm='fbp')
        # # r = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2)
        # res = (pd + 1j*pb)#*r
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))

        return res_gpu

    @profile
    # ptychography transform (FQ)
    def fwd_ptycho(self, psi):
        res_gpu = np.zeros(self.ptychoshape, dtype='complex64', order='C')
        for k in range(0, self.ptychoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.fwd(res_gpu[ast:aend].view(
                'float32'), psi[ast:aend].view('float32'))

        # res = np.zeros([self.theta.size,
        #                 self.scanax.shape[1]*self.scanay.shape[1],
        #                 self.det.x,self.det.y],dtype='complex64')

        # npadx = (self.det.x - self.prb.size) // 2
        # npady = (self.det.y - self.prb.size) // 2

        # for k in range(self.theta.size):
        #     for m in range(self.scanax.shape[1]):
        #         for n in range(self.scanay.shape[1]):
        #             stx = self.scanax[k,m]
        #             sty = self.scanay[k,n]
        #             if (stx==-1 or sty==-1):
        #                 continue
        #             phi = np.multiply(self.prb.complex, psi[k][stx:stx+self.prb.size, sty:sty+self.prb.size])
        #             phi = np.pad(phi, ((npadx, npadx), (npady, npady)), mode='constant')
        #             res[k,n+m*self.scanay.shape[1]] = np.fft.fft2(phi)/np.sqrt(phi.shape[0]*phi.shape[1])

        #             # phi = np.pad(
        #             #     phi, ((npadx, npadx), (npady, npady)), mode='constant')
        #             # tmp[n+m*self.scanay.shape[1]] = np.fft.fft2(phi) / \
        #             #     np.sqrt(phi.shape[0]*phi.shape[1])

        # import matplotlib.pyplot as plt
        # plt.subplot(2,2,1)
        # plt.imshow(res[0,0].real)
        # plt.colorbar()
        # plt.subplot(2,2,2)
        # plt.imshow(res_gpu[0,0].real)
        # plt.colorbar()
        # plt.subplot(2,2,3)
        # plt.imshow(res[0,0].imag)
        # plt.colorbar()
        # plt.subplot(2,2,4)
        # plt.imshow(res_gpu[0,0].imag)
        # plt.colorbar()
        # plt.show()
        # print('fwd ptycho '+str(np.linalg.norm(res-res_gpu)/np.linalg.norm(res)))
        return res_gpu

    @profile
    # adjoint ptychography transfrorm (Q*F*)
    def adj_ptycho(self, data):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        for k in range(0, self.ptychoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.adj(res_gpu[ast:aend].view(
                'float32'), data[ast:aend].view('float32'))

        # res = np.zeros(self.tomoshape,dtype='complex64')scan
        # npadx = (self.det.x - self.prb.size) // 2
        # npady = (self.det.y - self.prb.size) // 2

        # for k in range(self.theta.size):
        #     for m in range(self.scanax.shape[1]):
        #         for n in range(self.scanay.shape[1]):
        #             tmp = data[k,n+m*self.scanay.shape[1]]
        #             iphi = np.fft.ifft2(tmp)*np.sqrt(tmp.shape[0]*tmp.shape[1])
        #             delphi = iphi[npadx:npadx+self.prb.size, npady:npady+self.prb.size]
        #             stx = self.scanax[k,m]
        #             sty = self.scanay[k,n]
        #             if(stx==-1 or sty==-1):
        #                 continue
        #             res[k,stx:stx+self.prb.size, sty:sty+self.prb.size] += np.multiply(np.conj(self.prb.complex), delphi)

        # import matplotlib.pyplot as plt
        # plt.subplot(2,2,1)
        # plt.imshow(res[0].real)
        # plt.colorbar()
        # plt.subplot(2,2,2)
        # plt.imshow(res_gpu[0].real)
        # plt.colorbar()
        # plt.subplot(2,2,3)
        # plt.imshow(res[0].imag)
        # plt.colorbar()
        # plt.subplot(2,2,4)
        # plt.imshow(res_gpu[0].imag)
        # plt.colorbar()
        # plt.show()
        # print('adj ptycho '+str(np.linalg.norm(res-res_gpu)/np.linalg.norm(res)))

        return res_gpu

    # multiply by probe and adj probe (Q^*Q)
    def adjfwd_prb(self, psi):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        for k in range(0, self.tomoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.adjfwd_prb(res_gpu[ast:aend].view(
                'float32'), psi[ast:aend].view('float32'))

        # res = np.zeros([len(self.theta),psi.shape[1],psi.shape[2]],dtype='complex')

        # for k in range(self.theta.size):
        #     for m in range(self.scanax.shape[1]):
        #         for n in range(self.scanay.shape[1]):
        #             stx = self.scanax[k,m]
        #             sty = self.scanay[k,n]
        #             res[k,stx:stx+self.prb.size, sty:sty+self.prb.size] += \
        #                 np.multiply(np.abs(self.prb.complex)**2, psi[k,stx:stx + self.prb.size, sty:sty + self.prb.size])

        # import matplotlib.pyplot as plt
        # plt.subplot(2,2,1)
        # plt.imshow(res[0].real)
        # plt.colorbar()
        # plt.subplot(2,2,2)
        # plt.imshow(res_gpu[0].real)
        # plt.colorbar()
        # plt.subplot(2,2,3)
        # plt.imshow(res[0].imag)
        # plt.colorbar()
        # plt.subplot(2,2,4)
        # plt.imshow(res_gpu[0].imag)
        # plt.colorbar()
        # plt.show()
        # print('adjfbp_prb ptycho '+str(np.linalg.norm(res-res_gpu)/np.linalg.norm(res)))
        return res_gpu

    @profile
    # amplitude update in Gradient descent ptychography
    def update_amp(self, init, data):
        for k in range(0, self.tomoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend = k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.update_amp(
                init[ast:aend].view('float32'), data[ast:aend].real)

        # res = init.copy()
        # for k in range(self.theta.size):
        #     res[k] = np.multiply(np.sqrt(data[k]), np.exp(1j * np.angle(res[k])))

        # import matplotlib.pyplot as plt
        # plt.subplot(2,2,1)
        # plt.imshow(res[0,0].real)
        # plt.colorbar()
        # plt.subplot(2,2,2)
        # plt.imshow(res_gpu[0,0].real)
        # plt.colorbar()
        # plt.subplot(2,2,3)
        # plt.imshow(res[0,0].imag)
        # plt.colorbar()
        # plt.subplot(2,2,4)
        # plt.imshow(res_gpu[0,0].imag)
        # plt.colorbar()
        # plt.show()
        # print('update ptycho '+str(np.linalg.norm(res-res_gpu)/np.linalg.norm(res)))
        return init

    # @profile
    # Gradient descent tomography
    def grad_tomo(self, data, niter, init, rho, eta):
        r = 1/np.sqrt(data.shape[0]*data.shape[1]/2)
        res = init.complexform/r
        for i in range(niter):
            tmp0 = self.fwd_tomo(res)*r
            tmp = self.adj_tomo(2*(tmp0-data))*r
            res = res - eta*tmp
        res *= r
        return objects.Object(res.imag, res.real, self.voxelsize)

    @profile
    # Gradient descent ptychography
    def grad_ptycho(self, data, init, niter, rho, gamma, hobj, lamd):
        psi = init.copy()
        for i in range(niter):
            tmp = self.fwd_ptycho(psi)
            tmp = self.update_amp(tmp, data)
            upd1 = self.adj_ptycho(tmp)
            upd2 = self.adjfwd_prb(psi)
            psi = (1 - rho*gamma) * psi + rho*gamma * \
                (hobj - lamd/rho) + (gamma / 2) * (upd1-upd2) / \
                np.power(np.abs(self.prb.complex), 2).max()
        return psi

    @profile
    # ADMM for ptycho-tomography problem
    def admm(self, data, h, psi, lamd, x, rho, gamma, eta, piter, titer, NITER):
        for m in range(NITER):
            # psi update
            psi = self.grad_ptycho(data, psi, piter, rho, gamma, h, lamd)
            # x update
            tmp0 = self.logtomo(psi+lamd/rho)
            _x = self.grad_tomo(tmp0, titer, x, rho, eta)

            # lambda update
            _h = self.fwd_tomo(_x.complexform)
            _h = self.exptomo(_h)
            _lamd = lamd + rho * (psi - _h)
            # convergence
            cp = np.sqrt(
                np.sum(np.power(np.abs(_x.complexform-x.complexform), 2)))
            print(m, cp)

            lamd = _lamd
            x = _x
            h = _h

        dxchange.write_tiff(x.beta,  'beta2/beta')
        dxchange.write_tiff(x.delta,  'delta2/delta')
