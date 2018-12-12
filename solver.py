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


def wavelength(energy):
    """Calculates the wavelength [cm] given energy [keV].
    
    Parameters
    ----------
    energy : scalar

    Returns
    -------
    scalar
    """
    return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def wavenumber(energy):
    """Calculates the wavenumber [1/cm] given energy [keV].
    
    Parameters
    ----------
    energy : scalar

    Returns
    -------
    scalar
    """
    return 2 * np.pi / wavelength(energy)

def fwd_tomo(psi, theta):
    pb = tomopy.project(psi.imag, theta, pad=False) 
    pd = tomopy.project(psi.real, theta, pad=False) 
    return pd + 1j*pb

def adj_tomo(data, theta):
    pb = tomopy.recon(np.imag(data), theta, algorithm='fbp') 
    pd = tomopy.recon(np.real(data), theta, algorithm='fbp') 
    return pd + 1j*pb

def grad_tomo(data, theta, voxelsize, energy, niter, init, eta):
    r = 1/np.sqrt(data.shape[0]*data.shape[1]/2)
    res = init.complexform/r
    for i in range(niter):
        tmp = fwd_tomo(res,theta)*r
        tmp = adj_tomo(2*(tmp-data),theta)*r
        res = res - eta*tmp
    res*=r
    return objects.Object(res.imag,res.real,voxelsize)

def exitwave(prb, psi, scan):
    return np.array([prb.complex * psi[i:i + prb.size, j:j + prb.size] for i in scan.x for j in scan.y], dtype='complex')


def propagate2(phi, det):
    npadx = (det.x - phi.shape[1]) // 2
    npady = (det.y - phi.shape[2]) // 2
    _phi = np.pad(phi, ((0, 0), (npadx, npadx), (npady, npady)), mode='constant')
    intensity = np.abs(np.fft.fft2(_phi)/np.sqrt(_phi.shape[1]*_phi.shape[2])) ** 2
    return intensity.astype('float32')


def propagate3(prb, psi, scan, theta, det):
    data = [] 
    for m in range(theta.size):
        phi = exitwave(prb, np.squeeze(psi[m]), scan[m])
        dat = propagate2(phi, det)
        data.append(dat)
    return data


def grad_ptycho(data, prb, scan, init, theta, niter, rho, gamma, hobj, lamd):
    psi = init.copy()

    for i in range(niter):
        upd1 = psi*0
        upd2 = psi*0

# Fwd ptycho FQ
        fwd_ptycho_tmp = []
        for k in range(theta.size):
            npadx = (data[k].shape[1] - prb.size) // 2
            npady = (data[k].shape[2] - prb.size) // 2
            tmp2 = np.zeros([len(scan[k].x)*len(scan[k].y),data[k].shape[1],data[k].shape[2]],dtype=complex)
            for m in range(len(scan[k].x)):
                for n in range(len(scan[k].y)):
                    stx = scan[k].x[m]
                    sty = scan[k].y[n]
                    phi = np.multiply(prb.complex, psi[k][stx:stx+prb.size, sty:sty+prb.size])
                    phi = np.pad(phi, ((npadx, npadx), (npady, npady)), mode='constant')
                    tmp2[n+m*len(scan[k].y)] = np.fft.fft2(phi)/np.sqrt(phi.shape[0]*phi.shape[1])
            fwd_ptycho_tmp.append(tmp2)

# Update amp
        for k in range(theta.size):
            fwd_ptycho_tmp[k] = np.multiply(np.sqrt(data[k]), np.exp(1j * np.angle(fwd_ptycho_tmp[k])))

# Adj ptycho Q*F*
        for k in range(theta.size):
            tmp2 = fwd_ptycho_tmp[k]
            for m in range(len(scan[k].x)):
                for n in range(len(scan[k].y)):
                    #tmp = np.multiply(np.sqrt(data[k][n+m*len(scan[k].y)]), np.exp(1j * np.angle(tmp2[n+m*len(scan[k].y)])))
                    tmp = tmp2[n+m*len(scan[k].y)]
                    iphi = np.fft.ifft2(tmp)*np.sqrt(tmp.shape[0]*tmp.shape[1])
                    delphi = iphi[npadx:npadx+prb.size, npady:npady+prb.size]
                    stx = scan[k].x[m]
                    sty = scan[k].y[n]
                    upd1[k,stx:stx+prb.size, sty:sty+prb.size] += np.multiply(np.conj(prb.complex), delphi)

# Q^*Q
        for k in range(theta.size):
            for m in range(len(scan[k].x)):
                for n in range(len(scan[k].y)):
                    stx = scan[k].x[m]
                    sty = scan[k].y[n]
                    upd2[k,stx:stx+prb.size, sty:sty+prb.size] += np.multiply(np.abs(prb.complex)**2, psi[k][stx:stx + prb.size, sty:sty + prb.size])

        psi = (1 - rho*gamma) * psi + rho*gamma * (hobj - lamd/rho) + (gamma / 2) * (upd1-upd2)
    return psi


def admm(data,prb,scan,hobj,psi,lamd,recobj,theta,voxelsize,energy,rho,gamma,eta,piter,titer):
   for m in range(3):
        # Ptychography.
        psi = grad_ptycho(data, prb, scan, psi, theta, piter, rho, gamma, hobj, lamd)
        cp = np.sqrt(np.sum(np.power(np.abs(hobj-psi), 2)))

        # Tomography.
        tmp = -1j / wavenumber(energy) * np.log(psi+lamd/rho) / voxelsize
        _recobj = grad_tomo(tmp, theta, voxelsize, energy, titer, recobj, eta)

        co = np.sqrt(np.sum(np.power(np.abs(recobj.complexform- _recobj.complexform), 2)))
        recobj = _recobj
        
        # Lambda update.
        hobj = fwd_tomo(recobj.complexform, theta)*voxelsize
        hobj = np.exp(1j * wavenumber(energy) * hobj)
       
        _lamd = lamd + 1 * rho * (psi - hobj)
        cl = np.sqrt(np.sum(np.power(np.abs(lamd-_lamd), 2)))
        lamd = _lamd.copy()
       
        print (m, cp, co, cl)