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
    def __init__(self, prb, scan, theta, det, voxelsize, energy, tomoshape):
        self.prb = prb
        self.scan = scan
        self.theta = theta
        self.det = det
        self.voxelsize = voxelsize
        self.energy = energy
        self.tomoshape = tomoshape
        self.objshape = [tomoshape[1], tomoshape[2], tomoshape[2]]
        self.ptychoshape = [theta.size, scan.shape[2], det.x, det.y]
        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta)
        # normalization coefficients
        self.coeftomo = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2)
        maxint = np.power(np.abs(prb.complex), 2).max().astype('float')
        self.coefptycho = 1/np.sqrt(maxint)
        self.coefdata = 1/(det.x*det.y*maxint)
        # create class for the ptycho transform
        # number of angles for simultaneous processing by 1 gpu
        self.theta_gpu = tomoshape[0]//10
        self.cl_ptycho = ptychofft.ptychofft(self.theta_gpu, tomoshape[1], tomoshape[2],
                                             scan.shape[2], det.x, det.y, prb.size)

    def wavenumber(self):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    # Exp representation of projections, exp(i\nu\psi)
    def exptomo(self, psi):
        return np.exp(1j * psi * self.voxelsize * self.wavenumber()/self.coeftomo)

    # Log representation of projections, -i/\nu log(psi)
    def logtomo(self, psi):
        return -1j / self.wavenumber() * np.log(psi) / self.voxelsize*self.coeftomo

    # Radon transform (R)
    def fwd_tomo(self, psi):
        res=np.zeros(self.tomoshape, dtype='complex64', order='C')
        self.cl_tomo.fwd(res, psi)
        # normalization
        res *= self.coeftomo
        return res

    # Adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res=np.zeros(self.objshape, dtype='complex64', order='C')
        self.cl_tomo.adj(res, data)
        # normalization
        res *= self.coeftomo
        return res

    # Ptychography transform (FQ)
    def fwd_ptycho(self, psi):
        res=np.zeros(self.ptychoshape, dtype='complex64', order='C')
        #self.cl_ptycho.fwd(res[ast:aend], psi[ast:aend])
        for k in range(0, self.ptychoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend=k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scan[0, ast:aend], self.scan[1, ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.fwd(res[ast:aend], psi[ast:aend])
        # normalization
        res *= self.coefptycho
        return res

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data):
        res=np.zeros(self.tomoshape, dtype='complex64', order='C')
        for k in range(0, self.ptychoshape[0]//self.theta_gpu):
            # process self.theta_gpu angles on 1gpu simultaneously
            ast, aend=k*self.theta_gpu, (k+1)*self.theta_gpu
            self.cl_ptycho.setobj(self.scan[0, ast:aend], self.scan[1, ast:aend],
                                  self.prb.complex)
            self.cl_ptycho.adj(res[ast:aend], data[ast:aend])
        # normalization
        res *= self.coefptycho
        return res

    # Forward operator for regularization (J)
    def fwd_reg(self, x):
        res=np.zeros([3, *self.objshape], dtype='complex64', order='C')
        res[0, :, :, :-1]=x[:, :, 1:]-x[:, :, :-1]
        res[1, :, :-1, :]=x[:, 1:, :]-x[:, :-1, :]
        res[2, :-1, :, :]=x[1:, :, :]-x[:-1, :, :]
        # normalization
        res *= 2/np.sqrt(3)
        return res

    # Adjoint operator for regularization (J^*)
    def adj_reg(self, gr):
        res=np.zeros(self.objshape, dtype='complex64', order='C')
        res[:, :, 1:]=gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0]=gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res=-res
        # normalization
        res *= 2/np.sqrt(3)
        return res

    # xi0,K, and K for linearization of the tomography problem
    def takexi(self, psi, phi, lamd, mu, rho, tau):
        K=1j*self.voxelsize * self.wavenumber()*(psi-lamd/rho)
        # normalization
        K=K/np.amax(np.abs(K))
        xi0=K*(-1j*np.log(psi-lamd/rho) / \
               (self.voxelsize * self.wavenumber()))*self.coeftomo
        xi1=phi-mu/tau
        return xi0, xi1, K

    # Line search for the step sizes gamma
    def line_search(self,minf,gamma,x,fx,d,fd):
        while(minf(x, fx)-minf(x+gamma*d, fx+gamma*fd) < 0 and gamma > 1e-4):
            gamma *= 0.5            
        return gamma
            
    # Conjugate gradients tomography
    def cg_tomo(self, xi0, xi1, K, niter, init, rho, tau):
        def minf(KRx, gx): return rho*np.linalg.norm(KRx-xi0)**2+tau*np.linalg.norm(gx-xi1)**2             
        x = init.complexform
        gamma = 8
        for i in range(niter):
            KRx = K*self.fwd_tomo(x)
            gx = self.fwd_reg(x)
            grad = rho*self.adj_tomo(np.conj(K)*(KRx-xi0))+tau*self.adj_reg(gx-xi1)
            if i == 0:
                d=-grad
            else:                
              d=-grad+np.linalg.norm(grad)**2/((np.sum(d*np.conj(grad-grad0))))*d
            grad0=grad

            KRd=K*self.fwd_tomo(d)
            gd=self.fwd_reg(d)
            gamma = self.line_search(minf,gamma,KRx,gx,KRd,gd)                   
            #print(i, gamma, minf(KRx, gx), minf(
                    #KRx+gamma*KRd, gx+gamma*gd))
            x = x + gamma*d
        return objects.Object(x.imag, x.real, self.voxelsize)        

    # Conjugate gradients for the Gaussian ptychography model
    def cg_gaussian_ptycho(self, data, init, niter, rho, h, lamd):
        def minf(psi, fpsi): return np.linalg.norm(
            np.abs(fpsi)-np.sqrt(data))**2+rho*np.linalg.norm(h-psi+lamd/rho)**2
        psi=init.copy()
        gamma=8  # init gamma as a large value
        eps=1e-5
        for i in range(niter):
            fpsi=self.fwd_ptycho(psi)
            grad=self.adj_ptycho(
                fpsi-fpsi*eps/(np.abs(fpsi)*eps+eps*eps)*np.sqrt(data))-rho*(h - psi + lamd/rho)
            if i == 0:
                d=-grad
            else:                
                d=-grad+np.linalg.norm(grad)**2/((np.sum(d*np.conj(grad-grad0))))*d
            grad0=grad
            fd=self.fwd_ptycho(d)
            gamma =  self.line_search(minf,gamma,psi,fpsi,d,fd)                   
            #print(i, gamma, minf(psi, fpsi), minf(
                    #psi+gamma*d, fpsi+gamma*fd))
            psi=psi + gamma*d
        return psi

    # Conjugate gradients for the Poisson ptychography model
    def cg_poisson_ptycho(self, data, init, niter, rho, h, lamd):
        def minf(psi, fpsi): return np.sum(np.abs(fpsi)**2-2*data*np.log(np.abs(fpsi)+np.float32(data<1e-7))) +rho*np.linalg.norm(h-psi+lamd/rho)**2
        psi=init.copy()
        gamma=8  # init gamma as a large value
        eps=1e-5
        for i in range(niter):
            fpsi=self.fwd_ptycho(psi)
            grad=self.adj_ptycho(
                fpsi-data*eps /(np.conj(fpsi)*eps+eps*eps))-rho*(h - psi + lamd/rho)            
            if i == 0:
                d=-grad
            else:
                d=-grad+np.linalg.norm(grad)**2/((np.sum(d*np.conj(grad-grad0))))*d
            grad0=grad
            fd=self.fwd_ptycho(d)
            gamma =  self.line_search(minf,gamma,psi,fpsi,d,fd)                   
            #print(i, gamma, minf(psi, fpsi), minf(
            #       psi+gamma*d, fpsi+gamma*fd))
            psi=psi + gamma*d
        return psi        

    # Regularizer problem
    def solve_reg(self, x, mu, tau, alpha):
        z=self.fwd_reg(x)+mu/tau
        za=np.sqrt(np.sum(z*np.conj(z), 0))
        z[:, za <= alpha/tau]=0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/za[za > alpha/tau]
        return z

    # Update rho, tau for a faster convergence
    def update_penalty(self, rho, tau, psi, h, h0, phi, e, e0):
        # rho
        r=np.linalg.norm(psi - h)**2
        s=np.linalg.norm(rho*(h-h0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        # tau
        r=np.linalg.norm(phi - e)**2
        s=np.linalg.norm(tau*(e-e0))**2
        if (r > 10*s):
            tau *= 2
        elif (s > 10*r):
            tau *= 0.5
        return rho, tau

    # Lagrangian terms
    def take_lagr(self,psi,phi,data,h,e,lamd,mu,tau,rho,alpha,model):
        lagr = np.zeros(7, dtype="float32")
        fpsi = self.fwd_ptycho(psi)
        if (model == 'poisson'):                    
            lagr[0]=np.sum(np.abs(fpsi)**2-2*data*np.log(np.abs(fpsi)+np.float32(data<1e-7)))
        if (model == 'gaussian'):
            lagr[0]=np.linalg.norm(np.abs(fpsi)-np.sqrt(data))
        lagr[1]=alpha*np.sum(np.sqrt(np.sum(phi*np.conj(phi), 0)))
        lagr[2]=2*np.sum(np.real(np.conj(lamd)*(h-psi)))
        lagr[3]=rho*np.linalg.norm(h-psi)**2
        lagr[4]=2*np.sum(np.real(np.conj(mu)*(e-phi)))
        lagr[5]=tau*np.linalg.norm(e-phi)**2
        lagr[6]=np.sum(lagr[0:5])
        return lagr        

    # ADMM for ptycho-tomography problem
    def admm(self, data, h, e, psi, phi, lamd, mu, x, alpha, piter, titer, NITER, model):
        # normalization
        data=data.copy()*self.coefdata
        # init penalties
        rho,tau = 1,1           
        #Lagrangian for each iter
        lagr=np.zeros([NITER, 7], dtype="float32")
        lagr0 = self.take_lagr(psi,phi,data,h,e,lamd,mu,tau,rho,alpha,model)     
        for m in range(NITER):
            # keep previous iteration
            psi0, phi0, x0, h0, e0, lamd0, mu0=psi, phi, x, h, e, lamd, mu
            # ptychography problem
            if (model == 'gaussian'):
                psi=self.cg_gaussian_ptycho(data, psi, piter, rho, h, lamd)
            elif (model == 'poisson'):
                psi=self.cg_poisson_ptycho(data, psi, piter, rho, h, lamd)
            # tomography problem
            xi0, xi1, K=self.takexi(psi, phi, lamd, mu, rho, tau)
            x=self.cg_tomo(xi0, xi1, K, titer, x, rho, tau)
            # regularizer problem
            phi=self.solve_reg(x.complexform, mu, tau, alpha)
            # h,e updates
            h=self.exptomo(self.fwd_tomo(x.complexform))            
            e=self.fwd_reg(x.complexform)
            # lambda, mu updates
            lamd=lamd + rho * (h-psi)            
            mu=mu + tau * (e-phi)
            # update rho, tau for a faster convergence
            rho, tau = self.update_penalty(
               rho, tau, psi, h, h0, phi, e, e0)

            if (np.mod(m, 1) == 0):
                lagr[m] = self.take_lagr(psi,phi,data,h,e,lamd,mu,tau,rho,alpha,model)
                print("%d) rho=%.2e, tau=%.2e, Terms:  %.2e %.2e %.2e %.2e %.2e %.2e, Sum: %.2e" %
                   (m, rho, tau, *(lagr[m]-lagr0)))
                lagr0=lagr[m]                
        return x, psi, lagr

