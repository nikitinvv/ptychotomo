"""Module for 3D ptychography."""

import warnings

import cupy as cp
import numpy as np
import signal
import sys
from ptychotomo.ptychofft import ptychofft

warnings.filterwarnings("ignore")


class SolverPtycho(object):
    def __init__(self, nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta):
        self.nscan = nscan
        self.nprb = nprb
        self.ntheta = ntheta
        self.nz = nz
        self.n = n
        self.nscan = nscan
        self.ndetx = ndetx
        self.ndety = ndety
        self.nprb = nprb
        self.ptheta = ptheta

        # create class for the ptycho transform
        self.cl_ptycho = ptychofft(
            self.ptheta, self.nz, self.n, self.nscan, self.ndetx, self.ndety, self.nprb)

        # GPU memory deallocation with ctrl+C, ctrl+Z signals
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Free gpu memory after SIGINT, SIGSTSTP (destructor)"""
        self = []
        sys.exit(0)

    def fwd_ptycho(self, psi, scan, prb):
        """Ptychography transform (FQ)"""
        res = cp.zeros([self.ptheta, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr,
                           scan.data.ptr, prb.data.ptr)  # C++ wrapper, send pointers to GPU arrays
        return res

    def adj_ptycho(self, data, scan, prb):
        """Adjoint ptychography transform (Q*F*)"""
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64')
        flg = 0  # compute adjoint operator with respect to object
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr,
                           scan.data.ptr, prb.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    def adj_ptycho_prb(self, data, scan, psi):
        """Adjoint ptychography probe transform (O*F*), object is fixed"""
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64')
        flg = 1  # compute adjoint operator with respect to probe
        self.cl_ptycho.adj(psi.data.ptr, data.data.ptr,
                           scan.data.ptr, res.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    def line_search(self, minf, gamma, u, fu, d, fd):
        """Line search for the step sizes gamma"""
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-32):
            gamma *= 0.5
        if(gamma <= 1e-32):  # direction not found
            gamma = 0
            warnings.warn("Line search failed for conjugate gradient.")
        return gamma

    def cg_ptycho(self, data, psi, scan, prb, piter,  model, recover_prb, h, lamd, rho):
        """Conjugate gradients for ptychography"""
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim
        # minimization functional

        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * cp.log(cp.abs(fpsi)+1e-32))
            if(rho is not 0):
                f += rho*cp.linalg.norm(h-psi+lamd/rho)**2
            return f

        print("# congujate gradient parameters\n"
              "iteration, step size object, step size probe, function min")  # csv column headers
        for k in range(piter):
            # 1) object retrieval subproblem with fixed probe
            # forward operator
            fpsi = self.fwd_ptycho(psi, scan, prb)
            # take gradient
            if model == 'gaussian':
                gradpsi = self.adj_ptycho(
                    fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), scan, prb)/(cp.max(cp.abs(prb))**2)
            elif model == 'poisson':
                gradpsi = self.adj_ptycho(
                    fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32), scan, prb)/(cp.max(cp.abs(prb))**2)
            if(rho is not 0):
                gradpsi -= rho*(h - psi + lamd/rho)
            # Dai-Yuan direction
            if k == 0:
                dpsi = -gradpsi
            else:
                dpsi = -gradpsi+cp.linalg.norm(gradpsi)**2 / \
                    ((cp.sum(cp.conj(dpsi)*(gradpsi-gradpsi0)))+1e-32)*dpsi
            gradpsi0 = gradpsi
            # line search
            fdpsi = self.fwd_ptycho(dpsi, scan, prb)
            gammapsi = self.line_search(
                minf, 1, psi, fpsi, dpsi, fdpsi)  # start with gammapsi = 1 on each iteration
            # update psi
            psi = psi + gammapsi*dpsi

            if(recover_prb):
                # 2) probe retrieval subproblem with fixed object
                # forward operator
                fprb = self.fwd_ptycho(psi, scan, prb)
                # take gradient
                if model == 'gaussian':
                    gradprb = self.adj_ptycho_prb(
                        fprb-cp.sqrt(data)*cp.exp(1j*cp.angle(fprb)), scan, psi)/cp.max(cp.abs(psi))**2/self.nscan
                elif model == 'poisson':
                    gradprb = self.adj_ptycho_prb(
                        fprb-data*fprb/(cp.abs(fprb)**2+1e-32), scan, psi)/cp.max(cp.abs(psi))**2/self.nscan
                # Dai-Yuan direction
                if (k == 0):
                    dprb = -gradprb
                else:
                    dprb = -gradprb+cp.linalg.norm(gradprb)**2 / \
                        ((cp.sum(cp.conj(dprb)*(gradprb-gradprb0))))*dprb
                gradprb0 = gradprb
                # line search
                fdprb = self.fwd_ptycho(psi, scan, dprb)
                gammaprb = self.line_search(
                    minf, 1, psi, fprb, psi, fdprb)  # start with gammaprb = 1 on each iteration
                # update prb
                prb = prb + gammaprb*dprb

            # check convergence
            if (np.mod(k, 1) == 0):
                fpsi = self.fwd_ptycho(psi, scan, prb)
                if(recover_prb):
                    print("%4d, %.3e, %.3e, %.7e" %
                      (k, gammapsi, gammaprb, minf(psi, fpsi)))
                else:
                    print("%4d, %.3e,  %.7e" %
                      (k, gammapsi, minf(psi, fpsi)))

        return psi, prb

    def cg_ptycho_batch(self, data, initpsi, scan, initprb, piter, model, recover_prb, rho=0, h=np.array([None]), lamd=np.array([None])):
        """Solve ptycho by angles partitions."""
        assert initprb.ndim == 3, "prb needs 3 dimensions, not %d" % initprb.ndim

        psi = initpsi.copy()
        prb = initprb.copy()

        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            psi[ids], prb[ids] = self.cg_ptycho(
                datap, psi[ids], scan[:, ids], prb[ids], piter, model, recover_prb, h[ids], lamd[ids], rho)
        return psi, prb

    # For data generation and adjoint test
    def fwd_ptycho_batch(self, psi, scan, prb):
        """Batch of Ptychography transform (FQ)"""
        data = np.zeros([self.ntheta, self.nscan, self.ndety,
                         self.ndetx], dtype='complex64')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data_gpu = self.fwd_ptycho(
                psi[ids], scan[:, ids], prb[ids])  # compute part on GPU
            data[ids] = data_gpu.get()  # copy to CPU
        return data
    
    def adj_ptycho_batch(self, data, scan, prb):
        """batch Adjoint ptychography transform (Q*F*)"""
        res = np.zeros([self.ntheta, self.nz, self.n],dtype='complex64')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data_gpu = cp.array(data[ids])
            res_gpu = self.adj_ptycho(
                data_gpu, scan[:, ids], prb[ids])  # compute part on GPU            
            res[ids] = res_gpu.get()
        return res
