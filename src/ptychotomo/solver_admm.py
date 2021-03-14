
import numpy as np
from .solver_tomo import SolverTomo
from .solver_deform import SolverDeform
from .solver_ptycho import SolverPtycho
from .flowvis import flow_to_color
from .utils import *
import signal
import dxchange
import sys
import os
import matplotlib.pyplot as plt


class SolverAdmm(object):
    def __init__(self, nscan, theta, center, ndet, voxelsize, energy, ntheta, nz, n, nprb, ptheta, pnz, nmodes, ngpus):

        self.ntheta = ntheta
        self.nz = nz
        self.n = n
        self.nscan = nscan
        self.ndet = ndet
        self.nprb = nprb
        self.ptheta = ptheta
        self.pnz = pnz
        self.nmodes = nmodes
        self.ngpus = ngpus

        self.tslv = SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus)
        self.pslv = SolverPtycho(
            ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus)
        self.dslv = SolverDeform(ntheta, nz, n, ptheta, ngpus)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def signal_handler(self, sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        self.tslv = []
        self.pslv = []
        self.dslv = []
        sys.exit(0)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        return

    # Update rho3, for a faster convergence
    def update_penalty(self, psi3, h3, h30, psi1, h1, h10, rho3, rho1):
        # rho3
        r = np.linalg.norm(psi3 - h3)**2
        s = np.linalg.norm(rho3*(h3-h30))**2
        if (r > 10*s):
            rho3 *= 2
        elif (s > 10*r):
            rho3 *= 0.5
        # rho1
        r = np.linalg.norm(psi1 - h1)**2
        s = np.linalg.norm(rho1*(h1-h10))**2
        if (r > 10*s):
            rho1 *= 2
        elif (s > 10*r):
            rho1 *= 0.5
        return rho3, rho1

    # Lagrangian terms for monitoring convergence
    def take_lagr(self, psi3, psi1, data, prb, scan, h3, h1, lamd3, lamd1, rho3, rho1):
        # \mathcal{L}_{\bar{\rho}} (u,\bar{\psi}, \bar{\lambda})=&\sum_{j=1}^{n} \left\{|\Fop\Qop_p\psi_1|_j^2-2d_j\log|\Fop\Qop_p\psi_1|_j \right\} +  \alpha q(\psi_2)\\&
        # + 2\text{Re}\{\lambda_1^H(\Top_t \psi_3-\psi_1)\}+\rho_1\|\Top_t \psi_3-\psi_1\|_2^2\\&
        # + 2\text{Re}\{\lambda_2^H(\Jop u-\psi_2)\}+ \rho_2\|\Jop u-\psi_2\|_2^2\\&
        # + 2\text{Re}\{\lambda_3^H(\Hop u-\psi_3)\}+\rho_3\|\Hop u-\psi_3\|_2^2
        lagr = np.zeros(6, dtype="float32")
        #  Lagrangian ptycho part by angles partitions
        lagr[0] = self.pslv.take_error(data, psi1, prb, scan)
        lagr[1] = 2*np.sum(np.real(np.conj(lamd1)*(h1-psi1)))
        lagr[2] = rho1*np.linalg.norm(h1-psi1)**2
        lagr[3] = 2*np.sum(np.real(np.conj(lamd3)*(h3-psi3)))
        lagr[4] = rho3*np.linalg.norm(h3-psi3)**2
        lagr[5] = np.sum(lagr[0:5])

        return lagr

    # ADMM for ptycho-tomography problem
    def admm(self, data, psi3, psi1, flow, prb, scan,
             h3, h1, lamd3, lamd1,
             u, piter, titer, diter, niter, recover_prb, name='admm_rec'):

        # data /= (self.ndetx*self.ndety)  # FFT compensation

        pars = [0.5, 1, self.nz, 4, 5, 1.1, 4]
        rho3 = 0.5
        rho1 = 0.5

        for i in range(niter):
            # &\psi_1^{k+1}, (q^{k+1}) =  \argmin_{\psi_1, q} \sum_{j = 1}^{n}
            # \left\{ |\Fop\Qop_q\psi_1|_j^2-2d_j\log |\Fop\Qop_p\psi_1|_j \right\} +
            # \rho_1\|h1 -\psi_1 +\lambda_1^k /\rho_1\| _2^2,
            # h1 == \Top_{t^k} \psi_3^k
            # if(i==0):
            h30,  h10 = h3,  h1
            psi1, prb = self.pslv.grad_ptycho_batch(
                data, psi1, prb, scan, h1+lamd1/rho1, rho1, piter, recover_prb)
            # # keep previous iteration for penalty updates
            # # &\psi_3^{k+1}, (t^{k+1}) = \argmin_{\psi_3,t} \rho_1\|\Top_t \psi_3-\psi_1^{k+1}+
            # # \lambda_1^k/\rho_1\|_2^2+\rho_3\|\Hop u^k-\psi_3+\lambda_3^k/\rho_3\|_2^2

            mmin, mmax = find_min_max(np.angle(psi1-lamd1/rho1))
            flow = self.dslv.registration_flow_batch(
                np.angle(psi3), np.angle(psi1-lamd1/rho1), mmin, mmax, flow, pars)

            psi3 = self.dslv.grad_deform_gpu_batch(
                psi1-lamd1/rho1, psi3, flow, diter, h3+lamd3/rho3, rho3/rho1)
            # # tomography problem
            # # u^{k+1} = \argmin_{u,t} \rho_2\|\Jop u-\psi_2^{k+1}+\lambda_2^k/\rho_2\|_2^2
            # # +\rho_3\|\Hop u-\psi_3^{k+1}+\lambda_3^k/\rho_3\|_2^2,\quad \text{//tomography}\\
            xi0, K, pshift = self.pslv.takexi(psi3, lamd3, rho3)
            u = self.tslv.grad_tomo_batch(xi0, K, u, titer)
            h3 = self.pslv.exptomo(self.tslv.fwd_tomo_batch(u))*np.exp(1j*pshift)
            h1 = self.dslv.apply_flow_gpu_batch(
                psi3.real, flow)+1j*self.dslv.apply_flow_gpu_batch(psi3.imag, flow)
            # lamd updates
            lamd3 = lamd3 + rho3 * (h3-psi3)
            lamd1 = lamd1 + rho1 * (h1-psi1)
            # update rho for a faster convergence
            rho3, rho1 = self.update_penalty(
                psi3, h3, h30, psi1, h1, h10, rho3, rho1)

            pars[2] -= 1

            # Lagrangians difference between two iterations
            if (np.mod(i, 1) == 0):
                lagr = self.take_lagr(
                    psi3, psi1, data, prb, scan, h3, h1, lamd3, lamd1, rho3, rho1)
                #print("%d/%d) flow=%.2e, winsize=%d, rho3=%.2e, rho1=%.2e, Lagrangian terms:  %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e , Sum: %.2e" %
                      #(i, niter, np.linalg.norm(flow), pars[2], rho3, rho1, *lagr))
                print(f"{i}/{niter}) {np.linalg.norm(flow)=}, {pars[2]=}, {rho3=:.2e}, {rho1=:.2e}", 
                      "Lagrangian terms: [", *(f"{x:.1e}" for x in lagr), "]")
                if not os.path.exists(name+'flow/'):
                    os.makedirs(name+'flow/')
                plt.clf()
                plt.subplot(2, 2, 1)
                plt.imshow(flow_to_color(flow[0]))
                plt.subplot(2, 2, 2)
                plt.imshow(flow_to_color(flow[self.ntheta//4]))
                plt.subplot(2, 2, 3)
                plt.imshow(flow_to_color(flow[3*self.ntheta//4]))
                plt.subplot(2, 2, 4)
                plt.imshow(flow_to_color(flow[self.ntheta-1]))
                plt.savefig(name+'flow/'+str(i)+'.png')

                dxchange.write_tiff_stack(np.angle(psi3),
                                          name+'psi3iter'+str(self.n)+'/'+str(i), overwrite=True)
                dxchange.write_tiff_stack(np.abs(psi3),
                                          name+'psi3iterabs'+str(self.n)+'/'+str(i), overwrite=True)
                dxchange.write_tiff_stack(np.angle(psi1),
                                          name+'psi1iter'+str(self.n)+'/'+str(i), overwrite=True)
                dxchange.write_tiff_stack(np.abs(psi1),
                                          name+'psi1iterabs'+str(self.n)+'/'+str(i), overwrite=True)
                dxchange.write_tiff_stack(np.real(u),
                                          name+'ure'+str(self.n)+'/'+str(i), overwrite=True)
                dxchange.write_tiff_stack(np.imag(u),
                                          name+'uim'+str(self.n)+'/'+str(i), overwrite=True)

        return u, psi3, psi1, flow, prb
