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
    def __init__(self, prb, scan, scanax, scanay, theta, det, voxelsize, energy, tomoshape):
        self.prb = prb
        self.scan = scan
        self.scanax = scanax
        self.scanay = scanay
        self.theta = theta
        self.det = det
        self.voxelsize = voxelsize
        self.energy = energy
        self.tomoshape = tomoshape
        self.objshape = [self.tomoshape[1],
                         self.tomoshape[2], self.tomoshape[2]]
        self.ptychoshape = [scanax.shape[1],
                            scanay.shape[1], det.x, det.y, prb.size]

        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta)
        # create class for the ptycho transform
        self.cl_ptycho = ptychofft.ptychofft(1,
            *self.tomoshape[1:], *self.ptychoshape)
        self.cl_ptycho.setobj(theta[0:1], scanax, scanay,
                              prb.complex.view('float32'))

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
        psi_gpu = np.array(psi, dtype='complex64', order='C')
        self.cl_tomo.fwd(res_gpu.view('float32'), psi_gpu.view('float32'))

        # pb = tomopy.project(psi.imag, self.theta, pad=False)
        # pd = tomopy.project(psi.real, self.theta, pad=False)
        # r = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2)
        # res = (pd + 1j*pb)#*r
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))

        return res_gpu

    # adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res_gpu = np.zeros(self.objshape, dtype='complex64', order='C')
        data_gpu = np.array(data, dtype='complex64', order='C')
        self.cl_tomo.adj(res_gpu.view('float32'), data_gpu.view('float32'))

        # pb = tomopy.recon(np.imag(data), self.theta, algorithm='fbp')
        # pd = tomopy.recon(np.real(data), self.theta, algorithm='fbp')
        #r = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2)
        # res = (pd + 1j*pb)#*r
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))

        return res_gpu

    # ptychography transform (FQ)
    def fwd_ptycho(self, psi):
        res_gpu = np.zeros([self.theta.size,
                            self.scanax.shape[1]*self.scanay.shape[1],
                            self.det.x, self.det.y], dtype='complex64', order='C')
        psi_gpu = np.array(psi.astype('complex64'), order='C')
        for k in range(len(self.theta)):
            self.cl_ptycho.fwd(res_gpu[k:k+1].view('float32'), psi_gpu[k:k+1].view('float32'))

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
        #             phi = np.multiply(self.prb.c.astype('float32')lex, psi[k][stx:stx+self.prb.size, sty:sty+self.prb.size])
        #             phi = np.pad(phi, ((npadx, n.astype('float32')x), (npady, npady)), mode='constant')
        #             res[k,n+m*self.scanay.shape[.astype('float32') = np.fft.fft2(phi)/np.sqrt(phi.shape[0]*phi.shape[1])

        # import matplotlib.pyplot as plt
        # plt.subplot(2,2,1)                      .astype('float32')
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
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))
        return res_gpu

    # adjoint ptychography transfrorm (Q*F*)
    def adj_ptycho(self, data):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        data_gpu = np.array(data.astype('complex64'), order='C')
        for k in range(len(self.theta)):
            self.cl_ptycho.adj(res_gpu[k:k+1].view('float32'), data_gpu[k:k+1].view('float32'))

        # res = np.zeros(self.tomoshape,dtype='complex')
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
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))

        return res_gpu

    # multiply by probe and adj probe (Q^*Q)
    def adjfwd_prb(self, psi):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        psi_gpu = np.array(psi.astype('complex64'), order='C')

        for k in range(len(self.theta)):
            self.cl_ptycho.adjfwd_prb(res_gpu[k:k+1].view(
                'float32'), psi_gpu[k:k+1].view('float32'))

        # res = np.zeros([len(self.theta),psi.shape[1],psi.shape[2]],dtype='complex')
        # for k in range(self.theta.size):
        #     for m in range(len(self.scan[k].x)):
        #         for n in range(len(self.scan[k].y)):
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
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))
        return res_gpu

    # amplitude update in Gradient descent ptychography
    def update_amp(self, init, data):
        res_gpu = np.array(init.copy().astype('complex64'), order='C')
        data_gpu = np.array(data.astype('float32'), order='C')
        for k in range(len(self.theta)):
            self.cl_ptycho.update_amp(res_gpu[k:k+1].view('float32'), data_gpu[k:k+1])

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
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))

        return res_gpu

    @profile
    # Gradient descent tomography
    def grad_tomo(self, data, niter, init, eta):
        r = 1/np.sqrt(data.shape[0]*data.shape[1]/2)
        res = init.complexform/r
        for i in range(niter):
            tmp = self.fwd_tomo(res)*r
            tmp = self.adj_tomo(2*(tmp-data))*r
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
                (hobj - lamd/rho) + (gamma / 2) * (upd1-upd2)
        return psi

    @profile
    # ADMM for ptycho-tomography problem
    def admm(self, data, hobj, psi, lamd, recobj, rho, gamma, eta, piter, titer):
        for m in range(10):
            # Ptychography
            psi = self.grad_ptycho(data, psi, piter, rho, gamma, hobj, lamd)
            # Tomography
            tmp = self.logtomo(psi+lamd/rho)
            _recobj = self.grad_tomo(tmp, titer, recobj, eta)
            # Lambda update
            _hobj = self.fwd_tomo(_recobj.complexform)
            _hobj = self.exptomo(_hobj)
            _lamd = lamd + 1 * rho * (psi - _hobj)

            # convergence
            cp = np.sqrt(np.sum(np.power(np.abs(hobj-psi), 2)))
            co = np.sqrt(
                np.sum(np.power(np.abs(recobj.complexform - _recobj.complexform), 2)))
            cl = np.sqrt(np.sum(np.power(np.abs(lamd-_lamd), 2)))
            print(m, cp, co, cl)
            dxchange.write_tiff(
                recobj.beta[:, recobj.beta.shape[0] // 2],  'beta/beta')
            dxchange.write_tiff(
                recobj.delta[:, recobj.delta.shape[0] // 2],  'delta/delta')
            ##############

            # update to next iter
            lamd = _lamd
            recobj = _recobj
            hobj = _hobj
