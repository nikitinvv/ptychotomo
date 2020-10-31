"""Module for 3D ptychography."""
import signal
import sys
import cupy as cp
import numpy as np
import dxchange
import time
import concurrent.futures as cf
import threading
from itertools import repeat
from functools import partial
import cv2
from ptychotomo.radonusfft import radonusfft
from ptychotomo.ptychofft import ptychofft
from ptychotomo.deform import deform
from ptychotomo.util import tic, toc, find_min_max, find_mass_center_shifts
from ptychotomo.flowvis import flow_to_color
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, scan, theta, det, voxelsize, energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes):

        self.voxelsize = voxelsize
        self.energy = energy
        self.scan = scan
        self.ntheta = ntheta
        self.nz = nz
        self.n = n
        self.nscan = scan.shape[2]
        self.ndety = det[0]
        self.ndetx = det[1]
        self.nprb = nprb
        self.ptheta = ptheta
        self.pnz = pnz
        self.nmodes = nmodes
        # create class for the ptycho transform
        self.cl_ptycho = ptychofft(
            self.ptheta, self.nz, self.n, self.ptheta, self.nscan, self.ndety, self.ndetx, self.nprb)

        # create class for the tomo transform
        self.cl_tomo = radonusfft(self.ntheta, self.pnz, self.n)
        self.cl_tomo.setobj(cp.ascontiguousarray(theta).data.ptr)

        self.cl_deform = deform(self.nz, self.n, self.ptheta)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def signal_handler(self, sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        self.cl_tomo = []
        self.cl_ptycho = []
        sys.exit(0)

    def mlog(self, psi3):
        res = psi3.copy()
        res[cp.abs(psi3) < 1e-32] = 1e-32
        res = cp.log(res)
        return res

    # Wave number index
    def wavenumber(self):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    # Exp representation of projections, exp(i\nu\psi3)
    def exptomo(self, psi3):
        return cp.exp(1j*psi3 * self.voxelsize * self.wavenumber())

    # Log representation of projections, -i/\nu log(psi3)
    def logtomo(self, psi3):
        return -1j / self.wavenumber() * self.mlog(psi3) / self.voxelsize

    # Radon transform (R)
    def fwd_tomo(self, u):
        res = cp.zeros([self.ntheta, self.pnz, self.n],
                       dtype='complex64')
        self.cl_tomo.fwd(cp.ascontiguousarray(res).data.ptr,
                         cp.ascontiguousarray(u).data.ptr)
        return res

    # Adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res = cp.zeros([self.pnz, self.n, self.n],
                       dtype='complex64')
        self.cl_tomo.adj(cp.ascontiguousarray(res).data.ptr,
                         cp.ascontiguousarray(data).data.ptr)
        return res

    # Batch of Radon transform (R)
    def fwd_tomo_batch(self, u):
        res = cp.zeros([self.ntheta, self.nz, self.n],
                       dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            res[:, ids] = self.fwd_tomo(u[ids])
        return res

    # Batch of adjoint Radon transform (R^*)
    def adj_tomo_batch(self, data):
        res = cp.zeros([self.nz, self.n, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            res[ids] = self.adj_tomo(data[:, ids])
        return res

    # Ptychography transform (FQ)
    def fwd_ptycho(self, psi3, prb, scan):
        res = cp.zeros([self.ptheta, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64')
        self.cl_ptycho.fwd(cp.ascontiguousarray(res).data.ptr, cp.ascontiguousarray(psi3).data.ptr,
                           cp.ascontiguousarray(prb).data.ptr, cp.ascontiguousarray(scan).data.ptr)
        return res

    # Batch of Ptychography transform (FQ)
    def fwd_ptycho_batch(self, psi3, prb, scan):
        data = np.zeros([self.ntheta, self.nscan, self.ndety,
                         self.ndetx], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data0 = cp.zeros(
                [len(ids), self.nscan, self.ndety, self.ndetx], dtype='float32')
            for k in range(self.nmodes):
                tmp = self.fwd_ptycho(psi3[ids], prb[ids, k], scan[:, ids])
                data0 += cp.abs(tmp)**2
            data[ids] = data0.get()
        data *= (self.ndetx*self.ndety)  # FFT compensation
        return data

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data, prb, scan):
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64')
        self.cl_ptycho.adj(cp.ascontiguousarray(res).data.ptr, cp.ascontiguousarray(data).data.ptr,
                           cp.ascontiguousarray(prb).data.ptr, cp.ascontiguousarray(scan).data.ptr)
        return res

    def adj_ptycho_prb(self, data, psi3, scan):
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64')
        self.cl_ptycho.adjprb(cp.ascontiguousarray(res).data.ptr, cp.ascontiguousarray(data).data.ptr,
                              cp.ascontiguousarray(psi3).data.ptr, cp.ascontiguousarray(scan).data.ptr)
        return res

    # Forward operator for regularization (J)
    def fwd_reg(self, u):
        res = cp.zeros([3, *u.shape], dtype='complex64')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        return res

    # Adjoint operator for regularization (J^*)
    def adj_reg(self, gr):
        res = cp.zeros(gr.shape[1:], dtype='complex64')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res *= -1
        return res

    # xi0,K, and K for linearization of the tomography problem
    def takexi(self, psi3, psi2, lamd3, lamd2, rho3, rho2):
        # bg subtraction parameters
        r = self.nprb/2
        m1 = cp.mean(
            cp.angle(psi3[:, :, r:2*r]))
        m2 = cp.mean(cp.angle(
            psi3[:, :, psi3.shape[2]-2*r:psi3.shape[2]-r]))
        pshift = (m1+m2)/2

        
        t = psi3-lamd3/rho3
        t *= cp.exp(-1j*pshift)
        logt = self.mlog(t)

        # K, xi0, xi1
        K = 1j*self.voxelsize * self.wavenumber()*t
        K = K/cp.amax(cp.abs(K))  # normalization
        xi0 = K*(-1j*(logt) /
                 (self.voxelsize * self.wavenumber()))
        xi1 = psi2-lamd2/rho2
        return xi0, xi1, K, pshift

    # Conjugate gradients tomography
    def cg_tomo(self, xi0, xi1, K, init, rho3, rho2, titer):
        # minimization functional
        def minf(KRu, gu):
            return rho3*cp.linalg.norm(KRu-xi0)**2+rho2*cp.linalg.norm(gu-xi1)**2
        u = init.copy()
        # minf1 = 1e9
        for i in range(titer):
            KRu = K*self.fwd_tomo_batch(u)
            gu = self.fwd_reg(u)
            grad = rho3*self.adj_tomo_batch(cp.conj(K)*(KRu-xi0))/(self.ntheta * self.n/2) + \
                rho2*self.adj_reg(gu-xi1)
            r = min(1/rho3, 1/rho2)/2
            grad *= r
            # update step
            u = u + 0.5*(-grad)

            # minf0 = minf(KRu, gu)
            # if(minf1 < minf0):
            #     print('error in tomo', minf0, minf1)
            # minf1 = minf0
        return u

    def cg_ptycho(self, data, psi1, prb, scan, h1, lamd1, rho1, piter, model, recover_prb):
        # &\psi_1^{k+1}, (q^{k+1}) =  \argmin_{\psi_1, q} \sum_{j = 1}^{n}
        # \left\{ |\Fop\Qop_q\psi_1|_j^2-2d_j\log |\Fop\Qop_p\psi_1|_j \right\} +
        # \rho_1\|h1 -\psi_1 +\lambda_1^k /\rho_1\| _2^2,
        # h1 == \Top_{t^k} \psi_3^k
        # minimization functional
        def minf(fpsi, psi1):
            f = cp.linalg.norm(cp.sqrt(cp.abs(fpsi)) - cp.sqrt(data))**2
            if(rho1 is not None):
                f += rho1*cp.linalg.norm(h1-psi1+lamd1/rho1)**2
            return f

        # minf1 = 1e12
        for i in range(piter):

            # 1) object retrieval subproblem with fixed prbs
            # sum of forward operators associated with each prb
            # sum of abs value of forward operators
            absfpsi = data*0
            for k in range(self.nmodes):
                tmp = self.fwd_ptycho(psi1, prb[:, k], scan)
                absfpsi += cp.abs(tmp)**2

            a = cp.sum(cp.sqrt(absfpsi*data))
            b = cp.sum(absfpsi)
            prb *= (a/b)
            absfpsi *= (a/b)**2

            gradpsi = cp.zeros(
                [self.ptheta, self.nz, self.n], dtype='complex64')
            for k in range(self.nmodes):
                fpsi = self.fwd_ptycho(psi1, prb[:, k], scan)
                afpsi = self.adj_ptycho(fpsi, prb[:, k], scan)
                if(k == 0):
                    r = cp.real(cp.sum(psi1*cp.conj(afpsi)) /
                                cp.sum(afpsi*cp.conj(afpsi)))
                gradpsi += self.adj_ptycho(
                    fpsi - cp.sqrt(data) * fpsi/(cp.sqrt(absfpsi)+1e-32), prb[:, k], scan)
            if(rho1 is not None):                                
                gradpsi -= rho1*(h1 - psi1 + lamd1/rho1)
                gradpsi *= min(1/rho1, r)/2
            else:
                gradpsi *= r/2
            # update psi1
            psi1 = psi1 + 0.5 * (-gradpsi)

            if (recover_prb):
                if(i == 0):
                    gradprb = prb*0
                for m in range(0, self.nmodes):
                    # 2) prb retrieval subproblem with fixed object
                    # sum of forward operators associated with each prb

                    absfprb = data*0
                    for k in range(self.nmodes):
                        tmp = self.fwd_ptycho(psi1, prb[:, k], scan)
                        absfprb += np.abs(tmp)**2

                    fprb = self.fwd_ptycho(psi1, prb[:, m], scan)
                    afprb = self.adj_ptycho_prb(fprb, psi1, scan)
                    r = cp.real(
                        cp.sum(prb[:, m]*cp.conj(afprb))/cp.sum(afprb*cp.conj(afprb)))
                    # take gradient
                    gradprb[:, m] = self.adj_ptycho_prb(
                        fprb - cp.sqrt(data) * fprb/(cp.sqrt(absfprb)+1e-32), psi1, scan)
                    gradprb[:, m] *= r/2
                    prb[:, m] = prb[:, m] + 0.5 * (-gradprb[:, m])

            # # check convergence
            # minf0 = minf(absfpsi, psi1)
            # if(minf0 > minf1):
            #     print('error inptycho', minf0, minf1)
            # minf1 = minf0

        return psi1, prb

    # Solve ptycho by angles partitions
    def cg_ptycho_batch(self, data, psiinit, prbinit, scan, h1, lamd1, rho1, piter, model, recover_prb):
        psi1 = psiinit.copy()
        prb = prbinit.copy()
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            psi1[ids], prb[ids] = self.cg_ptycho(cp.array(
                data[ids]), psi1[ids], prb[ids], scan[:, ids], h1[ids], lamd1[ids], rho1, piter, model, recover_prb)
        return psi1, prb

    def registration_flow(self, psi, g, mmin, mmax, flow, pars, id):
        """Find optical flow for one projection"""
        tmp1 = ((psi[id]-mmin[id]) /
                (mmax[id]-mmin[id])*255)
        tmp1[tmp1 > 255] = 255
        tmp1[tmp1 < 0] = 0
        tmp2 = ((g[id]-mmin[id]) /
                (mmax[id]-mmin[id])*255)
        tmp2[tmp2 > 255] = 255
        tmp2[tmp2 < 0] = 0
        # print(np.max(tmp2))
        # print(np.min(tmp2))
        cv2.calcOpticalFlowFarneback(
            tmp1, tmp2, flow[id], *pars)  # updates flow

    def registration_flow_batch(self, psi3, psilamd1, mmin, mmax, flow=None, pars=[0.5, 3, 20, 16, 5, 1.1, 4]):
        """Find optical flow for all projections in parallel"""
        if (flow is None):
            flow = np.zeros([self.ntheta, self.nz, self.n, 2], dtype='float32')

        flow0 = flow.copy()
        with cf.ThreadPoolExecutor() as e:
            # update flow in place
            e.map(partial(self.registration_flow, np.angle(psi3), np.angle(psilamd1), mmin,
                          mmax, flow, pars), range(0, psi3.shape[0]))

        # # control Farneback's (may diverge for small window sizes)
        err = np.linalg.norm(psilamd1-self.apply_flow_gpu_batch(psi3, flow0).get(),axis=(1,2))
        err1 = np.linalg.norm(psilamd1-self.apply_flow_gpu_batch(psi3, flow).get(),axis=(1,2))
        idsbad = np.where(err1>err)[0]
        
        print('bad alignment for:',len(idsbad))
        flow[idsbad] = flow0[idsbad]


        return flow

    def apply_flow_gpu(self, f, flow, value):
        h, w = flow.shape[1:3]
        flow = -flow.copy()
        flow[:, :, :, 0] += cp.arange(w)
        flow[:, :, :, 1] += cp.arange(h)[:, cp.newaxis]

        flowx = cp.array(flow[:, :, :, 0])
        flowy = cp.array(flow[:, :, :, 1])
        g = f.copy()#*0+value  # keep values that were not affected
        # g = cp.zeros([self.ptheta,self.nz,self.n],dtype='float32')

        self.cl_deform.remap(cp.ascontiguousarray(g).data.ptr, cp.ascontiguousarray(f).data.ptr,
                             cp.ascontiguousarray(flowx).data.ptr, cp.ascontiguousarray(flowy).data.ptr)
        return g

    def apply_flow_gpu_batch(self, f, flow):
        res = cp.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            f_gpu = cp.array(f[ids])
            flow_gpu = cp.array(flow[ids])
            res_gpu = self.apply_flow_gpu(
                f_gpu.real, flow_gpu,1)+1j*self.apply_flow_gpu(f_gpu.imag, flow_gpu,0)
            # copy result to cpu
            res[ids] = res_gpu
        return res

    def cg_deform_gpu(self, psilamd1, psi3, flow, diter, h3lamd3, rho1=0, rho3=0):
        """CG solver for deformation"""
        # minimization functional
        def minf(psi3, Tpsi3):
            f = rho1*cp.linalg.norm(Tpsi3-psilamd1)**2 + \
                rho3*cp.linalg.norm(psi3-h3lamd3)**2
            return f

        # minf1 = 1e9
        for i in range(diter):
            Tpsi3 = self.apply_flow_gpu(
                psi3.real, flow, 1)+1j*self.apply_flow_gpu(psi3.imag, flow,0)
            grad = rho1*(self.apply_flow_gpu((Tpsi3-psilamd1).real, -flow,1)+1j *
                         self.apply_flow_gpu((Tpsi3-psilamd1).imag, -flow,0)) + rho3*(psi3-h3lamd3)
            r = min(1/rho1, 1/rho3)/2
            grad *= r
            # update step
            psi3 = psi3 + 0.5*(-grad)
            # check convergence
            Tpsi3 = self.apply_flow_gpu(psi3.real, flow,1)+1j*self.apply_flow_gpu(psi3.imag, flow,0)
            # minf0 = minf(psi3, Tpsi3)
            # if(minf0 > minf1):
            #     print('error in deform', minf0, minf1)
            #     minf1 = minf0
            # print(i,minf0)
        return psi3

    def cg_deform_gpu_batch(self, psilamd1, psi3, flow, diter, h3lamd3, rho1=0, rho3=0):
        res = cp.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            psilamd1_gpu = cp.array(psilamd1[ids])
            psi3_gpu = cp.array(psi3[ids])
            h3lamd3_gpu = cp.array(h3lamd3[ids])
            flow_gpu = cp.array(flow[ids])
            # Radon transform
            res_gpu = self.cg_deform_gpu(
                psilamd1_gpu, psi3_gpu, flow_gpu, diter, h3lamd3_gpu, rho1, rho3)
            # copy result to cpu
            res[ids] = res_gpu
        return res

    # Regularizer problem
    def solve_reg(self, u, lamd2, rho2, alpha):
        z = self.fwd_reg(u)+lamd2/rho2
        # Soft-thresholding
        za = cp.sqrt(cp.real(cp.sum(z*cp.conj(z), 0)))
        z[:, za <= alpha/rho2] = 0
        z[:, za > alpha/rho2] -= alpha/rho2 * \
            z[:, za > alpha/rho2]/(za[za > alpha/rho2])
        return z

    # Update rho3, rho2 for a faster convergence
    def update_penalty(self, psi3, h3, h30, psi2, h2, h20, psi1, h1, h10, rho3, rho2, rho1):
        # rho3
        r = cp.linalg.norm(psi3 - h3)**2
        s = cp.linalg.norm(rho3*(h3-h30))**2
        if (r > 10*s):
            rho3 *= 2
        elif (s > 10*r):
            rho3 *= 0.5
        # rho2
        r = cp.linalg.norm(psi2 - h2)**2
        s = cp.linalg.norm(rho2*(h2-h20))**2
        if (r > 10*s):
            rho2 *= 2
        elif (s > 10*r):
            rho2 *= 0.5
        # rho1
        r = cp.linalg.norm(psi1 - h1)**2
        s = cp.linalg.norm(rho1*(h1-h10))**2
        if (r > 10*s):
            rho1 *= 2
        elif (s > 10*r):
            rho1 *= 0.5
        return rho3, rho2, rho1

    # Lagrangian terms for monitoring convergence
    def take_lagr(self, psi3, psi2, psi1, data, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, alpha, rho3, rho2, rho1, model):
        # \mathcal{L}_{\bar{\rho}} (u,\bar{\psi}, \bar{\lambda})=&\sum_{j=1}^{n} \left\{|\Fop\Qop_p\psi_1|_j^2-2d_j\log|\Fop\Qop_p\psi_1|_j \right\} +  \alpha q(\psi_2)\\&
        # + 2\text{Re}\{\lambda_1^H(\Top_t \psi_3-\psi_1)\}+\rho_1\|\Top_t \psi_3-\psi_1\|_2^2\\&
        # + 2\text{Re}\{\lambda_2^H(\Jop u-\psi_2)\}+ \rho_2\|\Jop u-\psi_2\|_2^2\\&
        # + 2\text{Re}\{\lambda_3^H(\Hop u-\psi_3)\}+\rho_3\|\Hop u-\psi_3\|_2^2
        lagr = cp.zeros(9, dtype="float32")
        # Lagrangian ptycho part by angles partitions
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            datap = cp.array(data[ids])
            # normalized data
            absfprb = datap*0
            for k in range(self.nmodes):
                tmp = self.fwd_ptycho(psi1[ids], prb[ids, k], scan[:, ids])
                absfprb += np.abs(tmp)**2
            lagr[0] += cp.linalg.norm(cp.sqrt(absfprb)-cp.sqrt(datap))**2
        lagr[1] = alpha*cp.sum(np.sqrt(cp.real(cp.sum(psi2*cp.conj(psi2), 0))))
        lagr[2] = 2*cp.sum(cp.real(cp.conj(lamd1)*(h1-psi1)))
        lagr[3] = rho1*cp.linalg.norm(h1-psi1)**2
        lagr[4] = 2*cp.sum(np.real(cp.conj(lamd2)*(h2-psi2)))
        lagr[5] = rho2*cp.linalg.norm(h2-psi2)**2
        lagr[6] = 2*cp.sum(np.real(cp.conj(lamd3)*(h3-psi3)))
        lagr[7] = rho2*cp.linalg.norm(h3-psi3)**2
        lagr[8] = cp.sum(lagr[0:8])
        return lagr

    # ADMM for ptycho-tomography problem
    def admm(self, data, psi3, psi2, psi1, flow, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, u, alpha, piter, titer, diter, niter, model, recover_prb, align, name):

        data /= (self.ndetx*self.ndety)  # FFT compensation

        pars = [0.5, 4, self.n+16, 4, 5, 1.1, 4]
        rho3 = 0.5
        rho2 = 0.5
        rho1 = 0.5
        
        for m in range(niter):
            # keep previous iteration for penalty updates
            h30, h20, h10 = h3, h2, h1
            # &\psi_1^{k+1}, (q^{k+1}) =  \argmin_{\psi_1, q} \sum_{j = 1}^{n}
            # \left\{ |\Fop\Qop_q\psi_1|_j^2-2d_j\log |\Fop\Qop_p\psi_1|_j \right\} +
            # \rho_1\|h1 -\psi_1 +\lambda_1^k /\rho_1\| _2^2,
            # h1 == \Top_{t^k} \psi_3^k
            psi1, prb = self.cg_ptycho_batch(
                data, psi1, prb, scan, h1, lamd1, rho1, piter, model, recover_prb)
            psi1lamds = psi1-lamd1/rho1
                                                                                           
            # &\psi_3^{k+1}, (t^{k+1}) = \argmin_{\psi_3,t} \rho_1\|\Top_t \psi_3-\psi_1^{k+1}+
            # \lambda_1^k/\rho_1\|_2^2+\rho_3\|\Hop u^k-\psi_3+\lambda_3^k/\rho_3\|_2^2
            mmin,mmax = find_min_max(cp.angle(psi1lamds).get())            
            flow = self.registration_flow_batch(
                psi3.get(), (psi1lamds).get(), mmin, mmax, flow, pars)*(align == True)
            
            psi3 = self.cg_deform_gpu_batch(
                psi1lamds, psi3, flow, diter, (h3+lamd3/rho3), rho1, rho3)
            # tomography problem
            # u^{k+1} = \argmin_{u,t} \rho_2\|\Jop u-\psi_2^{k+1}+\lambda_2^k/\rho_2\|_2^2
            # +\rho_3\|\Hop u-\psi_3^{k+1}+\lambda_3^k/\rho_3\|_2^2,\quad \text{//tomography}\\
            xi0, xi1, K, pshift = self.takexi(
                psi3, psi2, lamd3, lamd2, rho3, rho2)
            u = self.cg_tomo(xi0, xi1, K, u, rho3, rho2, titer)
            # regularizer problem
            psi2 = self.solve_reg(u, lamd2, rho2, alpha)
            # h3,h2 updates
            h3 = self.exptomo(self.fwd_tomo_batch(u))*cp.exp(1j*pshift)
            h2 = self.fwd_reg(u)
            h1 = self.apply_flow_gpu_batch(psi3, flow)
            # for k in range(self.ntheta):
                # h1[k] = cp.roll(h1[k],(int(cm[k,0]-self.nz//2),int(cm[k,1]-self.n//2)),axis=(0,1))

            # lamd updates
            lamd3 = lamd3 + rho3 * (h3-psi3)
            lamd2 = lamd2 + rho2 * (h2-psi2)
            lamd1 = lamd1 + rho1 * (h1-psi1)
            # lamd1*=0
            # update rho for a faster convergence
            rho3, rho2, rho1 = self.update_penalty(
                psi3, h3, h30, psi2, h2, h20, psi1, h1, h10, rho3, rho2, rho1)
            
            pars[2]-=1
            
            # Lagrangians difference between two iterations
            if (np.mod(m, 16) == 0):
                lagr = self.take_lagr(
                    psi3, psi2, psi1, data, prb, scan, h3, h2, h1, lamd3, lamd2, lamd1, alpha, rho3, rho2, rho1, model)
                print("%d/%d) flow=%.2e,  rho3=%.2e, rho2=%.2e, rho1=%.2e, Lagrangian terms:  %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e , Sum: %.2e" %
                      (m, niter, cp.linalg.norm(flow), rho3, rho2, rho1, *lagr))
                #plt.imshow(flow_to_color(flow[45]))
               # plt.savefig('flow/'+str(m)+'.png')   
            

                dxchange.write_tiff_stack(cp.angle(psi3).get(),
                                          'psi3iter'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)
                dxchange.write_tiff_stack(cp.abs(psi3).get(),
                                          'psi3iterabs'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)
                dxchange.write_tiff_stack(cp.angle(psi1).get(),
                                          'psi1iter'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)
                dxchange.write_tiff_stack(cp.abs(psi1).get(),
                                          'psi1iterabs'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)                
                dxchange.write_tiff_stack(cp.angle(prb[0]).get(),
                                          'prbiter'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)
                dxchange.write_tiff_stack(cp.abs(prb[0]).get(),
                                          'prbiterabs'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)
                dxchange.write_tiff_stack(cp.real(u).get(),
                                          'ure'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)
                dxchange.write_tiff_stack(cp.imag(u).get(),
                                          'uim'+str(self.n)+'/'+name+'/'+str(m), overwrite=True)

        return u, psi3, psi2, psi1, flow, prb
