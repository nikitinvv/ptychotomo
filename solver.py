# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import dxchange
import tomopy
import xraylib as xl
import numpy as np
import scipy as sp
import pyfftw
import shutil
import objects
import warnings
import copy
warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, prb, scan, theta, det, voxelsize, energy):
        self.prb = prb
        self.scan = scan
        self.theta = theta
        self.det = det
        self.voxelsize = voxelsize
        self.energy = energy
        
    def wavenumber(self,energy):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy)

    # exp representation of R
    def exptomo(self,psi):
        return np.exp(1j * psi * self.voxelsize* self.wavenumber(self.energy))

    # log represnetation of R
    def logtomo(self,psi):
        return 1 / self.wavenumber(self.energy) * np.log(psi) / self.voxelsize
       
    # Radon transform (R )
    def fwd_tomo(self,psi):
        pb = tomopy.project(psi.imag, self.theta, pad=False) 
        pd = tomopy.project(psi.real, self.theta, pad=False) 
        r = 1/np.sqrt(pb.shape[0]*pb.shape[1]/2)
        res = (pd + 1j*pb)#*r
        return res

    # adjoint Radon transform (R^*)
    def adj_tomo(self,data):
        pb = tomopy.recon(np.imag(data), self.theta, algorithm='fbp') 
        pd = tomopy.recon(np.real(data), self.theta, algorithm='fbp') 
        r = 1/np.sqrt(data.shape[0]*data.shape[1]/2)
        res = (pd + 1j*pb)#*r
        return res

    # ptychography transform (FQ)
    def fwd_ptycho(self,psi):
        res = []
        npadx = (self.det.x - self.prb.size) // 2
        npady = (self.det.y - self.prb.size) // 2

        for k in range(self.theta.size):
            tmp = np.zeros([len(self.scan[k].x)*len(self.scan[k].y),self.det.x,self.det.y],dtype=complex)
            for m in range(len(self.scan[k].x)):
                for n in range(len(self.scan[k].y)):
                    stx = self.scan[k].x[m]
                    sty = self.scan[k].y[n]
                    phi = np.multiply(self.prb.complex, psi[k][stx:stx+self.prb.size, sty:sty+self.prb.size])
                    phi = np.pad(phi, ((npadx, npadx), (npady, npady)), mode='constant')
                    tmp[n+m*len(self.scan[k].y)] = np.fft.fft2(phi)/np.sqrt(phi.shape[0]*phi.shape[1])
            res.append(tmp)
        return res

    # adjoint ptychography transfrorm (Q*F*)
    def adj_ptycho(self,data,psi):
        res = np.zeros([len(self.theta),psi.shape[1],psi.shape[2]],dtype='complex')
        npadx = (self.det.x - self.prb.size) // 2
        npady = (self.det.y - self.prb.size) // 2
    
        for k in range(self.theta.size):
           for m in range(len(self.scan[k].x)):
                for n in range(len(self.scan[k].y)):
                    tmp = data[k][n+m*len(self.scan[k].y)]
                    iphi = np.fft.ifft2(tmp)*np.sqrt(tmp.shape[0]*tmp.shape[1])
                    delphi = iphi[npadx:npadx+self.prb.size, npady:npady+self.prb.size]
                    stx = self.scan[k].x[m]
                    sty = self.scan[k].y[n]
                    res[k,stx:stx+self.prb.size, sty:sty+self.prb.size] += np.multiply(np.conj(self.prb.complex), delphi)
        return res                

    # multiply by probe and adj probe (Q^*Q)
    def adjfwd_prb(self,psi):
        res = np.zeros([len(self.theta),psi.shape[1],psi.shape[2]],dtype='complex')
        for k in range(self.theta.size):
            for m in range(len(self.scan[k].x)):
                for n in range(len(self.scan[k].y)):
                    stx = self.scan[k].x[m]
                    sty = self.scan[k].y[n]
                    res[k,stx:stx+self.prb.size, sty:sty+self.prb.size] += np.multiply(np.abs(self.prb.complex)**2, psi[k][stx:stx + self.prb.size, sty:sty + self.prb.size])
        return res                 

    # amplitude update in Gradient descent ptychography
    def update_amp(self,init,data):
        res = init.copy()
        for k in range(self.theta.size):
            res[k] = np.multiply(np.sqrt(data[k]), np.exp(1j * np.angle(res[k])))
        return res  

    # Gradient descent tomography
    def grad_tomo(self,data, niter, init, eta):
        r = 1/np.sqrt(data.shape[0]*data.shape[1]/2)
        res = init.complexform/r
        for i in range(niter):
            tmp = self.fwd_tomo(res)*r
            tmp = self.adj_tomo(2*(tmp-data))*r
            res = res - eta*tmp
        res*=r
        return objects.Object(res.imag,res.real,self.voxelsize)

    # Gradient descent ptychography
    def grad_ptycho(self,data, init, niter, rho, gamma, hobj, lamd):
        psi = init.copy()
        for i in range(niter):
            tmp = self.fwd_ptycho(psi)
            tmp = self.update_amp(tmp,data)
            upd1 = self.adj_ptycho(tmp,psi)
            upd2 = self.adjfwd_prb(psi)
            psi = (1 - rho*gamma) * psi + rho*gamma * (hobj - lamd/rho) + (gamma / 2) * (upd1-upd2)
        return psi

    # ADMM for ptycho-tomography problem 
    def admm(self,data,hobj,psi,lamd,recobj,rho,gamma,eta,piter,titer):
        for m in range(3):
            # Ptychography
            psi = self.grad_ptycho(data, psi, piter, rho, gamma, hobj, lamd)
            # Tomography
            tmp = -1j*self.logtomo(psi+lamd/rho)
            _recobj = self.grad_tomo(tmp, titer, recobj, eta)
            # Lambda update
            _hobj = self.fwd_tomo(_recobj.complexform)
            _hobj = self.exptomo(_hobj)
            _lamd = lamd + 1 * rho * (psi - _hobj)

            #convergence
            cp = np.sqrt(np.sum(np.power(np.abs(hobj-psi), 2)))
            co = np.sqrt(np.sum(np.power(np.abs(recobj.complexform- _recobj.complexform), 2)))
            cl = np.sqrt(np.sum(np.power(np.abs(lamd-_lamd), 2)))
            print (m, cp, co, cl)
            
            # update to next iter
            lamd = _lamd
            recobj = _recobj
            hobj = _hobj

