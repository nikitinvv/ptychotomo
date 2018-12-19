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
        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta)
        # create class for the ptycho transform
        self.projgpu = tomoshape[0]//2
        self.cl_ptycho = ptychofft.ptychofft(self.projgpu, tomoshape[1], tomoshape[2],
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
        psi_gpu = np.array(psi, dtype='complex64', order='C')
        self.cl_tomo.fwd(res_gpu.view('float32'), psi_gpu.view('float32'))

        # pb = tomopy.project(psi.imag, self.theta, pad=False)
        # pd = tomopy.project(psi.real, self.theta, pad=False)
        # #r = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2)
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
        # # r = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2)
        # res = (pd + 1j*pb)#*r
        # print(np.linalg.norm(res-res_gpu)/np.linalg.norm(res))

        return res_gpu

    # forward operator for regularization (q)
    def fwd_reg(self,x):
        res = np.zeros([3,*self.objshape], dtype='complex64', order='C')
        res[0,:,:,:-1] = x[:,:,1:]-x[:,:,:-1]
        res[1,:,:-1,:] = x[:,1:,:]-x[:,:-1,:]
        res[2,:-1,:,:] = x[1:,:,:]-x[:-1,:,:]
        return res

    # adjoint operator for regularization (q^*)
    def adj_reg(self,gr):
        res = np.zeros(self.objshape, dtype='complex64', order='C')
        res[:,:,1:] = gr[0,:,:,1:]-gr[0,:,:,:-1]
        res[:,:,0] = gr[0,:,:,0]
        res[:,1:,:] += gr[1,:,1:,:]-gr[1,:,:-1,:]
        res[:,0,:] += gr[1,:,0,:] 
        res[1:,:,:] += gr[2,1:,:,:]-gr[2,:-1,:,:]
        res[0,:,:] += gr[2,0,:,:] 
        return -res



    # ptychography transform (FQ)
    def fwd_ptycho(self, psi):
        res_gpu = np.zeros([self.theta.size,
                            self.scanax.shape[1]*self.scanay.shape[1],
                            self.det.x, self.det.y], dtype='complex64', order='C')
        psi_gpu = np.array(psi.astype('complex64'), order='C')
        for k in range(0, self.tomoshape[0]//self.projgpu):
            ast, aend = k*self.projgpu, (k+1)*self.projgpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.fwd(res_gpu[ast:aend].view(
                'float32'), psi_gpu[ast:aend].view('float32'))

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
        #             # tmp[n+m*len(self.scan[k].y)] = np.fft.fft2(phi) / \
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

    # adjoint ptychography transfrorm (Q*F*)
    def adj_ptycho(self, data):
        res_gpu = np.zeros(self.tomoshape, dtype='complex64', order='C')
        data_gpu = np.array(data.astype('complex64'), order='C')
        for k in range(0, self.tomoshape[0]//self.projgpu):
            ast, aend = k*self.projgpu, (k+1)*self.projgpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.adj(res_gpu[ast:aend].view(
                'float32'), data_gpu[ast:aend].view('float32'))

        # res = np.zeros(self.tomoshape,dtype='complex64')
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
        psi_gpu = np.array(psi.astype('complex64'), order='C')
        for k in range(0, self.tomoshape[0]//self.projgpu):
            ast, aend = k*self.projgpu, (k+1)*self.projgpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.adjfwd_prb(res_gpu[ast:aend].view(
                'float32'), psi_gpu[ast:aend].view('float32'))

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
        # print('adjfbp_prb ptycho '+str(np.linalg.norm(res-res_gpu)/np.linalg.norm(res)))
        return res_gpu

    # amplitude update in Gradient descent ptychography
    def update_amp(self, init, data):
        res_gpu = np.array(init.copy().astype('complex64'), order='C')
        data_gpu = np.array(data.astype('float32'), order='C')
        for k in range(0, self.tomoshape[0]//self.projgpu):
            ast, aend = k*self.projgpu, (k+1)*self.projgpu
            self.cl_ptycho.setobj(self.scanax[ast:aend], self.scanay[ast:aend],
                                  self.prb.complex.view('float32'))
            self.cl_ptycho.update_amp(
                res_gpu[ast:aend].view('float32'), data_gpu[ast:aend])
        #self.cl_ptycho.update_amp(res_gpu.view('float32'), data_gpu)

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

        return res_gpu

    # @profile
    # Gradient descent tomography
    def grad_tomo(self, data, data_reg, niter, init, rho,tau, eta):
        r = 1/np.sqrt(data.shape[0]*data.shape[1]/2)
        res = init.complexform/r
        for i in range(niter):
            tmp0 = self.fwd_tomo(res)*r
            tmp1 = self.fwd_reg(res)
            tmp = self.adj_tomo(2*(tmp0-data))*r
            tmp += self.adj_reg(2*(tmp1-data_reg))*tau/rho
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
                (hobj - lamd/rho) + (gamma / 2) * (upd1-upd2)/np.power(np.abs(self.prb.complex), 2).max()
        return psi

    @profile
    # ADMM for ptycho-tomography problem
    def admm(self, data, h, psi, y, lamd, x, rho, mu, tau, gamma, eta, piter, titer,reg_term):
        for m in range(128):
            # psi update
            psi = self.grad_ptycho(data, psi, piter, rho, gamma, h, lamd)
            # x update
            tmp0 = self.logtomo(psi+lamd/rho)
            tmp1 = y + mu/tau
            _x = self.grad_tomo(tmp0, tmp1, titer, x, rho,tau, eta)
            
            gr_tmp = self.fwd_reg(_x.complexform)
            # y update
            if reg_term==0:
                z = np.sqrt(gr_tmp[0]**2+gr_tmp[1]**2+gr_tmp[2]**2)
                y = gr_tmp  
                y[:,z<=1/tau] = 0
                y[:,z>1/tau] = z[z>1/tau]-1/tau*gr_tmp[:,z>1/tau]/z[z>1/tau]
            else:
                y = (mu+tau*self.fwd_reg(_x.complexform))/(2+tau)
            # lambda update
            _h = self.fwd_tomo(_x.complexform)
            _h = self.exptomo(_h)
            _lamd = lamd + rho * (psi - _h)
            # mu update
            q = self.fwd_reg(_x.complexform)
            _mu = mu + tau * (y - q)
            # # convergence
            cp = np.sqrt(np.sum(np.power(np.abs(h-psi), 2)))
            # cy = np.sqrt(np.sum(np.power(np.abs(q-y), 2)))
            # co = np.sqrt(
            #     np.sum(np.power(np.abs(x.complexform - _x.complexform), 2)))
            # cl = np.sqrt(np.sum(np.power(np.abs(lamd-_lamd), 2)))
            # cm = np.sqrt(np.sum(np.power(np.abs(mu-_mu), 2)))
            # print(m, cp, cy, co, cl,cm)
            print(m,cp)

            ##############
            # htmp = self.fwd_tomo(_x.complexform)
            # htmp = self.exptomo(_h)

            # data_cmp = fwd_ptycho(htmp)
            # data_cmp = np.abs(data_cmp)**2

            # gr_tmp = fwd_reg(_x.complexform)

            # cp = 0.5*np.sum((htmp-psi)**2)
            # cd = 0.5*np.sum((data-data_cmp)**2)
            # cgr = np.sum(gr_tmp[0]**2+gr_tmp[1]**2)
            # print(cp,cd,cgr,cd+)
            # update to next iter
            lamd = _lamd
            x = _x
            h = _h
            mu = _mu
        dxchange.write_tiff(
            x.beta,  'beta2/beta')
        dxchange.write_tiff(
            x.delta,  'delta2/delta')
