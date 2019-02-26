# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import radonusfft
import ptychofft
import cupy as np
import objects
import warnings

warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, prb, scan, theta, det, voxelsize, energy, tomoshape):
        self.voxelsize = voxelsize
        self.energy = energy
        # shapes
        self.tomoshape = tomoshape
        self.objshape = [tomoshape[1], tomoshape[2], tomoshape[2]]
        self.ptychoshape = [theta.size, scan.shape[2], det[0], det[1]]
        # create class for the tomo transform
        self.cl_tomo = radonusfft.radonusfft(*self.tomoshape)
        self.cl_tomo.setobj(theta.data.ptr)
        # create class for the ptycho transform
        self.cl_ptycho = ptychofft.ptychofft(*self.tomoshape,*self.ptychoshape[1:],prb.shape[0])
        self.cl_ptycho.setobj(scan.data.ptr, prb.data.ptr)                                             

        
        # normalization coefficients
        self.coeftomo = 1/np.sqrt(self.tomoshape[0]*self.tomoshape[2]/2).astype('float32')
        maxint = np.power(np.abs(prb), 2).max().astype('float32')
        self.coefptycho = 1/np.sqrt(maxint)
        self.coefdata = 1/(self.ptychoshape[2]*self.ptychoshape[3]*maxint)
        
    def wavenumber(self):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    # Exp representation of projections, exp(i\nu\psi)
    def exptomo(self, psi):
        return np.exp(1j*psi * self.voxelsize * self.wavenumber()/self.coeftomo)

    # Log representation of projections, -i/\nu log(psi)
    def logtomo(self, psi):
        return -1j / self.wavenumber() * np.log(psi) / self.voxelsize*self.coeftomo

    # Radon transform (R)
    @profile
    def fwd_tomo(self, psi):
        res=np.zeros(self.tomoshape, dtype='complex64', order='C')
        self.cl_tomo.fwd(res.data.ptr, psi.data.ptr)
        res *= self.coeftomo #normalization
        return res

    # Adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res=np.zeros(self.objshape, dtype='complex64', order='C')
        self.cl_tomo.adj(res.data.ptr, data.data.ptr)
        res *= self.coeftomo #normalization
        return res

    # Ptychography transform (FQ)
    @profile
    def fwd_ptycho(self, psi):
        res=np.zeros(self.ptychoshape, dtype='complex64', order='C')        
        #psi = np.array(psi, dtype='complex64',order='C')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr)     
        res *= self.coefptycho #normalization
        return res

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data):
        res=np.zeros(self.tomoshape, dtype='complex64', order='C')
        self.cl_ptycho.adj(res.data.ptr,data.data.ptr)        
        res *= self.coefptycho # normalization
        return res

    # Forward operator for regularization (J)
    @profile
    def fwd_reg(self, u):
        res=np.zeros([3, *self.objshape], dtype='complex64', order='C')
        res[0, :, :, :-1]=u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :]=u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :]=u[1:, :, :]-u[:-1, :, :]        
        res *= 2/np.sqrt(3)# normalization
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
        res *= -2/np.sqrt(3)# normalization
        return res

    # xi0,K, and K for linearization of the tomography problem
    @profile
    def takexi(self, psi, phi, lamd, mu, rho, tau):
        K=1j*self.voxelsize * self.wavenumber()*(psi-lamd/rho)        
        K=K/np.amax(np.abs(K))# normalization
        xi0=K*(-1j*np.log(psi-lamd/rho) / \
               (self.voxelsize * self.wavenumber()))*self.coeftomo
        xi1=phi-mu/tau
        return xi0, xi1, K

    # Line search for the step sizes gamma
    @profile
    def line_search(self,minf,gamma,u,fu,d,fd):
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-5):
            gamma *= 0.5            
        return gamma
            
    # Conjugate gradients tomography
    @profile
    def cg_tomo(self, xi0, xi1, K, niter, init, rho, tau):
        # minimization functional
        def minf(KRu, gu): return rho*np.linalg.norm(KRu-xi0)**2+tau*np.linalg.norm(gu-xi1)**2             
        u = init
        gamma = 8 # init gamma as a large value
        for i in range(niter):
            KRu = K*self.fwd_tomo(u)
            gu = self.fwd_reg(u)
            grad = rho*self.adj_tomo(np.conj(K)*(KRu-xi0))+tau*self.adj_reg(gu-xi1)
            #Dai-Yuan direction         
            if i == 0:
                d=-grad
            else:                
                d=-grad+np.linalg.norm(grad)**2/((np.sum(d*np.conj(grad-grad0))))*d
            grad0=grad
            #line search
            KRd=K*self.fwd_tomo(d)
            gd=self.fwd_reg(d)
            gamma = self.line_search(minf,gamma,KRu,gu,KRd,gd)                   
            #print(i, gamma, minf(KRu, gu), minf(
                    #KRu+gamma*KRd, gu+gamma*gd))
            u = u + gamma*d
        return u

    # Conjugate gradients for the Gaussian ptychography model
    @profile
    def cg_gaussian_ptycho(self, data, init, niter, rho, h, lamd):
        # minimization functional
        def minf(psi, fpsi): 
            return np.linalg.norm(np.abs(fpsi)-np.sqrt(data))**2+rho*np.linalg.norm(h-psi+lamd/rho)**2
        psi=init.copy()
        gamma=8  # init gamma as a large value
        for i in range(niter):
            fpsi=self.fwd_ptycho(psi)
            grad=self.adj_ptycho(
                fpsi-fpsi*1e-5/(np.abs(fpsi)*1e-5+1e-10)*np.sqrt(data))-rho*(h - psi + lamd/rho)
            #Dai-Yuan direction                
            if i == 0:
                d=-grad
            else:                
                d=-grad+np.linalg.norm(grad)**2/((np.sum(d*np.conj(grad-grad0))))*d
            grad0=grad
            #line search
            fd=self.fwd_ptycho(d)
            gamma =  self.line_search(minf,gamma,psi,fpsi,d,fd)                   
            #print(i, gamma, minf(psi, fpsi), minf(
                    #psi+gamma*d, fpsi+gamma*fd))
            psi=psi + gamma*d
        return psi

    # Conjugate gradients for the Poisson ptychography model
    @profile
    def cg_poisson_ptycho(self, data, init, niter, rho, h, lamd):
        def minf(psi, fpsi): return np.sum(np.abs(fpsi)**2-2*data*np.log(np.abs(fpsi)+np.float32(data<1e-5))) +rho*np.linalg.norm(h-psi+lamd/rho)**2
        psi=init.copy()
        gamma=8  # init gamma as a large value
        for i in range(niter):
            fpsi=self.fwd_ptycho(psi)
            grad=self.adj_ptycho(
                fpsi-data*1e-5/(np.conj(fpsi)*1e-5+1e-10))-rho*(h - psi + lamd/rho)            
            #Dai-Yuan
            if i == 0:
                d=-gradfloat32
            else:
                d=-grad+np.linalg.norm(grad)**2/((np.sum(d*np.conj(grad-grad0))))*d            
            grad0=grad            
            #line search
            fd=self.fwd_ptycho(d)
            gamma =  self.line_search(minf,gamma,psi,fpsi,d,fd)                   
            #print(i, gamma, minf(psi, fpsi), minf(
            #       psi+gamma*d, fpsi+gamma*fd))
            psi=psi + gamma*d
        return psi        

    # Regularizer problem
    @profile
    def solve_reg(self, u, mu, tau, alpha):
        z=self.fwd_reg(u)+mu/tau
        #Soft-thresholding
        za=np.sqrt(np.real(np.sum(z*np.conj(z), 0)))
        z[:, za <= alpha/tau]=0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/za[za > alpha/tau]
        return z

    # Update rho, tau for a faster convergence
    @profile
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
    @profile
    def take_lagr(self,psi,phi,data,h,e,lamd,mu,tau,rho,alpha,model):
        lagr = np.zeros(7, dtype="float32")
        fpsi = self.fwd_ptycho(psi)
        if (model == 'poisson'):                    
            lagr[0]=np.sum(np.abs(fpsi)**2-2*data*np.log(np.abs(fpsi)+np.float32(data<1e-5)))
        if (model == 'gaussian'):
            lagr[0]=np.linalg.norm(np.abs(fpsi)-np.sqrt(data))
        lagr[1]=alpha*np.sum(np.sqrt(np.real(np.sum(phi*np.conj(phi), 0))))
        lagr[2]=2*np.sum(np.real(np.conj(lamd)*(h-psi)))
        lagr[3]=rho*np.linalg.norm(h-psi)**2
        lagr[4]=2*np.sum(np.real(np.conj(mu)*(e-phi)))
        lagr[5]=tau*np.linalg.norm(e-phi)**2
        lagr[6]=np.sum(lagr[0:5])
        return lagr        

    # ADMM for ptycho-tomography problem
    @profile
    def admm(self, data, h, e, psi, phi, lamd, mu, u, alpha, piter, titer, NITER, model):        
        data=data.copy()*self.coefdata # normalization
        # init penalties
        rho,tau = 1,1           
        # Lagrangian for each iter
        lagr=np.zeros([NITER, 7], dtype="float32")
        lagr0 = self.take_lagr(psi,phi,data,h,e,lamd,mu,tau,rho,alpha,model)     
        for m in range(NITER):
            # keep previous iteration
            psi0, phi0, u0, h0, e0, lamd0, mu0=psi, phi, u, h, e, lamd, mu
            # ptychography problem
            if (model == 'gaussian'):
                psi=self.cg_gaussian_ptycho(data, psi, piter, rho, h, lamd)
            elif (model == 'poisson'):
                psi=self.cg_poisson_ptycho(data, psi, piter, rho, h, lamd)
            # tomography problem
            xi0, xi1, K=self.takexi(psi, phi, lamd, mu, rho, tau)
            u=self.cg_tomo(xi0, xi1, K, titer, u, rho, tau)
            # regularizer problem
            phi=self.solve_reg(u, mu, tau, alpha)
            # h,e updates
            h=self.exptomo(self.fwd_tomo(u))            
            e=self.fwd_reg(u)
            # lambda, mu updates
            lamd=lamd + rho * (h-psi)            
            mu=mu + tau * (e-phi)
            # update rho, tau for a faster convergence
            rho, tau = self.update_penalty(
               rho, tau, psi, h, h0, phi, e, e0)

            # Lagrangians difference between two iterations
            if (np.mod(m, 4) == 0):
                lagr[m] = self.take_lagr(psi,phi,data,h,e,lamd,mu,tau,rho,alpha,model)
                print("%d) rho=%.2e, tau=%.2e, Terms:  %.2e %.2e %.2e %.2e %.2e %.2e, Sum: %.2e" %
                   (m, rho, tau, *(lagr0-lagr[m])))
                lagr0=lagr[m]                
        return u, psi, lagr

