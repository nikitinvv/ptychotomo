"""Module for 3D ptychography."""
import signal
import sys
import cupy as cp
import numpy as np
import dxchange
from ptychotomo.radonusfft import radonusfft
from ptychotomo.ptychofft import ptychofft
import time


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
        self.cl_tomo.setobj(theta.data.ptr)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def signal_handler(self, sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        self.cl_tomo = []
        self.cl_ptycho = []
        sys.exit(0)

    def mlog(self, psi):
        res = psi.copy()
        res[cp.abs(psi) < 1e-32] = 1e-32
        res = cp.log(res)
        return res

    # Wave number index
    def wavenumber(self):
        return 2 * np.pi / (2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy)

    # Exp representation of projections, exp(i\nu\psi)
    def exptomo(self, psi):
        return cp.exp(1j*psi * self.voxelsize * self.wavenumber())

    # Log representation of projections, -i/\nu log(psi)
    def logtomo(self, psi):
        return -1j / self.wavenumber() * self.mlog(psi) / self.voxelsize

    # Radon transform (R)
    def fwd_tomo(self, u):
        res = cp.zeros([self.ntheta, self.pnz, self.n],
                       dtype='complex64')
        self.cl_tomo.fwd(res.data.ptr, u.data.ptr)
        return res

    # Adjoint Radon transform (R^*)
    def adj_tomo(self, data):
        res = cp.zeros([self.pnz, self.n, self.n],
                       dtype='complex64')
        self.cl_tomo.adj(res.data.ptr, data.data.ptr)
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
    def fwd_ptycho(self, psi, prb, scan):
        res = cp.zeros([self.ptheta, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr,
                           prb.data.ptr, scan.data.ptr)
        return res

    # Batch of Ptychography transform (FQ)
    def fwd_ptycho_batch(self, psi, prb, scan):
        data = np.zeros([self.ntheta, self.nscan, self.ndety,
                         self.ndetx], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data0 = cp.abs(self.fwd_ptycho(
                psi[ids], prb[ids], scan[:, ids]))**2
            data[ids] = data0.get()
        data *= (self.ndetx*self.ndety)  # FFT compensation
        return data

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data, prb, scan):
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64')
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr,
                           prb.data.ptr, scan.data.ptr)
        return res

    def adj_ptycho_prb(self, data, psi, scan):
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64')
        self.cl_ptycho.adjprb(res.data.ptr, data.data.ptr,
                              psi.data.ptr, scan.data.ptr)
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
    def takexi(self, psi, phi, lamd, mu, rho, tau):
        # bg subtraction parameters
        r = self.nprb/2
        m1 = cp.mean(
            cp.angle(psi[:, :, r:2*r]))
        m2 = cp.mean(cp.angle(
            psi[:, :, psi.shape[2]-2*r:psi.shape[2]-r]))
        pshift = (m1+m2)/2

        t = psi-lamd/rho
        t *= cp.exp(-1j*pshift)
        logt = self.mlog(t)

        # K, xi0, xi1
        K = 1j*self.voxelsize * self.wavenumber()*t
        K = K/cp.amax(cp.abs(K))  # normalization
        xi0 = K*(-1j*(logt) /
                 (self.voxelsize * self.wavenumber()))
        xi1 = phi-mu/tau
        return xi0, xi1, K, pshift

    # Line search for the step sizes gamma
    def line_search(self, minf, gamma, u, fu, d, fd, coef):
        
        while(minf(u, fu, coef)-minf(u+gamma*d, fu+gamma*fd, coef) < 0 and gamma > 1e-7):
            # print('c:',minf(u, fu, coef), minf(u+gamma*d, fu+gamma*fd, coef))
            gamma *= 0.5
        if(gamma <= 1e-7):  # direction not found
            #print('no direction')
            gamma = 0
        return gamma

    # Conjugate gradients tomography
    def cg_tomo(self, xi0, xi1, K, init, rho, tau, titer):
        # minimization functional
        def minf(KRu, gu, coef):
            return rho*cp.linalg.norm(KRu-xi0)**2*coef+tau*cp.linalg.norm(gu-xi1)**2
        u = init.copy()
        coef = 1/(self.ntheta * self.n/2)
        for i in range(titer):
            KRu = K*self.fwd_tomo_batch(u)
            gu = self.fwd_reg(u)
            grad = rho*self.adj_tomo_batch(cp.conj(K)*(KRu-xi0))*coef + \
                tau*self.adj_reg(gu-xi1)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    ((cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            gamma = self.line_search(
                minf, 1, KRu, gu, K*self.fwd_tomo_batch(d), self.fwd_reg(d), coef)
            # print(gamma,minf(KRu, gu))
            # update step
            u = u + gamma*d
        return u

    def line_search_sqr(self, f, p1, p2, p3, step_length=1, step_shrink=0.5):
        """Optimized line search for square functions
            Example of otimized computation for the Gaussian model:
            sum_j|G_j(psi+gamma dpsi)|^2 = sum_j|G_j(psi)|^2+
                                           gamma^2*sum_j|G_j(dpsi)|^2+
                                           gamma*sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            p1,p2,p3 are temp variables to avoid computing the fwd operator during the line serch
            p1 = sum_j|G_j(psi)|^2
            p2 = sum_j|G_j(dpsi)|^2
            p3 = sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)

            Parameters	
            ----------	
            f : function(x)	
                The function being optimized.	
            p1,p2,p3 : vectors	
                Temporarily vectors to avoid computing forward operators        
        """
        assert step_shrink > 0 and step_shrink < 1
        fp1 = f(p1) # optimize computation
        # Decrease the step length while the step increases the cost function        
        while f(p1+step_length**2 * p2+step_length*p3) > fp1:          
            if step_length < 1e-7:
                #warnings.warn("Line search failed for conjugate gradient.")
                return 0
            step_length *= step_shrink            
        return step_length

    
    def cg_ptycho(self, data, psi, prb, scan, h, lamd, rho, piter, model, recover_prb):
        # minimization functional

        def minf(fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.sqrt(cp.abs(fpsi)) - cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(
                    cp.abs(fpsi) - data * cp.log(cp.abs(fpsi) + 1e-32))
            # f *= coef
            # f += rho*cp.linalg.norm(h-psi+lamd/rho)**2
            return f
        print("# congujate gradient parameters\n"
              "iteration, step size object, step size prb, function min"
              )  # csv column headers
        gammaprb = 0

        for i in range(piter):

            # 1) object retrieval subproblem with fixed prbs
            # sum of forward operators associated with each prb
            # sum of abs value of forward operators
            absfpsi = data*0
            for k in range(self.nmodes):
                tmp = self.fwd_ptycho(psi, prb[:, k], scan)
                absfpsi += np.abs(tmp)**2

            a = cp.sum(cp.sqrt(absfpsi*data))
            b = cp.sum(absfpsi)                        
            prb *= (a/b)
            absfpsi *= (a/b)**2
            
            gradpsi = cp.zeros(
                [self.ptheta, self.nz, self.n], dtype='complex64')
            if model == 'gaussian':
                for k in range(self.nmodes):
                    fpsi = self.fwd_ptycho(psi, prb[:, k], scan) 
                    gradpsi += self.adj_ptycho(
                        fpsi - cp.sqrt(data) * fpsi/(cp.sqrt(absfpsi)+1e-32), prb[:, k], scan)# / (cp.max(cp.abs(prb[:,k]))**2)
            elif model == 'poisson':
                for k in range(self.nmodes):
                    gradpsi += self.adj_ptycho(
                        fpsi - data * fpsi / (absfpsi + 1e-32), prb[:, k], scan)# / (cp.max(cp.abs(prb[:,k]))**2)
                        
            # Use optimized line search for square functions, note:
            # sum_j|G_j(psi+gamma dpsi)|^2 = sum_j|G_j(psi)|^2+
            #                               gamma^2*sum_j|G_j(dpsi)|^2+
            #                               gamma*sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            # temp variables to avoid computing the fwd operator during the line serch
            # p1 = sum_j|G_j(psi)|^2
            # p2 = sum_j|G_j(dpsi)|^2
            # p3 = sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            p1 = data*0
            p2 = data*0
            p3 = data*0
            for k in range(self.nmodes):
                tmp1 = self.fwd_ptycho(psi, prb[:, k], scan)
                tmp2 = self.fwd_ptycho(-gradpsi, prb[:, k], scan)
                p1 += cp.abs(tmp1)**2
                p2 += cp.abs(tmp2)**2
                p3 += 2*(tmp1.real*tmp2.real+tmp1.imag*tmp2.imag)
            # line search
            gammapsi = self.line_search_sqr(minf, p1, p2, p3)
            
            # update psi
            psi = psi + gammapsi * (-gradpsi)
            if (recover_prb):
                if(i == 0):
                    gradprb = prb*0                    
                for m in range(0, self.nmodes):
                    # 2) prb retrieval subproblem with fixed object
                    # sum of forward operators associated with each prb
                    fprb = self.fwd_ptycho(psi, prb[:, m], scan)
                    # sum of abs value of forward operators
                    absfprb = data*0
                    for k in range(self.nmodes):
                        tmp = self.fwd_ptycho(psi, prb[:, k], scan)
                        absfprb += np.abs(tmp)**2
                    # take gradient
                    if model == 'gaussian':
                        gradprb[:, m] = self.adj_ptycho_prb(
                            fprb - cp.sqrt(data) * fprb/(cp.sqrt(absfprb)+1e-32), psi, scan)/ self.nscan* self.nmodes #/ cp.max(cp.abs(psi))**2   # ?
                    elif model == 'poisson':
                        gradprb[:, m] = self.adj_ptycho_prb(
                            fprb - data * fprb / (absfprb + 1e-32), scan, psi) / self.nscan# / cp.max(cp.abs(psi))**2 
                    # temp variables to avoid computing the fwd operator during the line serch
                    p1 = data*0
                    p2 = data*0
                    p3 = data*0
                    for k in range(self.nmodes):
                        tmp1 = self.fwd_ptycho(psi, prb[:, k], scan)
                        p1 += cp.abs(tmp1)**2
                    tmp1 = self.fwd_ptycho(psi, prb[:, m], scan)
                    tmp2 = self.fwd_ptycho(psi, -gradprb[:,m], scan)
                    p2 = cp.abs(tmp2)**2
                    p3 = 2*(tmp1.real*tmp2.real+tmp1.imag*tmp2.imag)
                    # line search
                    gammaprb = self.line_search_sqr(minf, p1, p2, p3)
                    # update prb
                    prb[:, m] = prb[:, m] + gammaprb * (-gradprb[:,m])                    
            # check convergence
            if (np.mod(i, 4) == 0):                
                print("%4d, %.3e, %.3e, %.7e, " %
                      (i, gammapsi, gammaprb, minf(absfpsi)))                                    
                dxchange.write_tiff(cp.angle(psi).get(),
                                    'psiiter/'+str(i), overwrite=True)
                dxchange.write_tiff(cp.abs(psi).get(),
                                    'psiiterabs/'+str(i), overwrite=True)
        return psi, prb

    # Solve ptycho by angles partitions
    def cg_ptycho_batch(self, data, psiinit, prbinit, scan, h, lamd, rho, piter, model, recover_prb):
        psi = psiinit.copy()
        prb = prbinit.copy()
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            psi[ids], prb[ids] = self.cg_ptycho(cp.array(
                data[ids]), psi[ids], prb[ids], scan[:, ids], h[ids], lamd[ids], rho, piter, model, recover_prb)
        return psi, prb

    # Regularizer problem
    def solve_reg(self, u, mu, tau, alpha):
        z = self.fwd_reg(u)+mu/tau
        # Soft-thresholding
        za = cp.sqrt(cp.real(cp.sum(z*cp.conj(z), 0)))
        z[:, za <= alpha/tau] = 0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/(za[za > alpha/tau])
        return z

    # Update rho, tau for a faster convergence
    def update_penalty(self, psi, h, h0, phi, e, e0, rho, tau):
        # rho
        r = cp.linalg.norm(psi - h)**2
        s = cp.linalg.norm(rho*(h-h0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        # tau
        r = cp.linalg.norm(phi - e)**2
        s = cp.linalg.norm(tau*(e-e0))**2
        if (r > 10*s):
            tau *= 2
        elif (s > 10*r):
            tau *= 0.5
        return rho, tau

    # Lagrangian terms for monitoring convergence
    def take_lagr(self, psi, phi, data, prb, scan, h, e, lamd, mu, alpha, rho, tau, model):
        lagr = cp.zeros(7, dtype="float32")
        # Lagrangian ptycho part by angles partitions
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            fpsi = self.fwd_ptycho(psi[ids], prb[ids], scan[:, ids])
            datap = cp.array(data[ids])
            # normalized data
            if (model == 'poisson'):
                lagr[0] += cp.sum(cp.abs(fpsi)**2-2*datap *
                                  self.mlog(cp.abs(fpsi))-(datap-2*datap*self.mlog(cp.sqrt(datap))))
            if (model == 'gaussian'):
                lagr[0] += cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(datap))**2
        lagr[1] = alpha*cp.sum(np.sqrt(cp.real(cp.sum(phi*cp.conj(phi), 0))))
        lagr[2] = 2*cp.sum(cp.real(cp.conj(lamd)*(h-psi)))
        lagr[3] = rho*cp.linalg.norm(h-psi)**2
        lagr[4] = 2*cp.sum(np.real(cp.conj(mu)*(e-phi)))
        lagr[5] = tau*cp.linalg.norm(e-phi)**2
        lagr[6] = cp.sum(lagr[0:5])
        return lagr

    # ADMM for ptycho-tomography problem
    def admm(self, data, psi, phi, prb, scan, h, e, lamd, mu, u, alpha, piter, titer, niter, model, recover_prb):

        data /= (self.ndetx*self.ndety)  # FFT compensation

        rho = 0.5
        tau = 0.5
        for m in range(niter):
            # keep previous iteration for penalty updates
            h0, e0 = h, e
            psi, prb = self.cg_ptycho_batch(
                data, psi, prb, scan, h, lamd, rho, piter, model, recover_prb)
            # tomography problem
            xi0, xi1, K, pshift = self.takexi(psi, phi, lamd, mu, rho, tau)
            u = self.cg_tomo(xi0, xi1, K, u, rho, tau, titer)
            # regularizer problem
            phi = self.solve_reg(u, mu, tau, alpha)
            # h,e updates
            h = self.exptomo(self.fwd_tomo_batch(u))*cp.exp(1j*pshift)
            e = self.fwd_reg(u)
            # lambda, mu updates
            lamd = lamd + rho * (h-psi)
            mu = mu + tau * (e-phi)
            # update rho, tau for a faster convergence
            rho, tau = self.update_penalty(
                psi, h, h0, phi, e, e0, rho, tau)
            # Lagrangians difference between two iterations
            if (np.mod(m, 4) == 0):
                lagr = self.take_lagr(
                    psi, phi, data, prb, scan, h, e, lamd, mu, alpha, rho, tau, model)
                print("%d/%d) rho=%.2e, tau=%.2e, Lagrangian terms:  %.2e %.2e %.2e %.2e %.2e %.2e, Sum: %.2e" %
                      (m, niter, rho, tau, *lagr))                
        return u, psi, prb
