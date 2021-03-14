import numpy as np
import concurrent.futures as cf
import threading
import cv2
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
from .deform import deform
from .flowvis import flow_to_color
from .utils import chunk
from functools import partial


class SolverDeform(deform):
    """Base class for deformation solvers.
    This class is a context manager which provides the basic operators required
    to implement an alignment solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attributes
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.   
    """

    def __init__(self, ntheta, nz, n, ptheta, ngpus):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        if(ntheta % ptheta > 0):
            print('Error, ptheta is not a multiple of ntheta')
            exit()
        super().__init__(ntheta, nz, n, ptheta, ngpus)
        #self.err = np.ones(ntheta,dtype='float32')*1e8

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def registration_flow(self, psi, g, mmin, mmax, flow, pars, id):
        """Find optical flow for one projection by using opencv library on CPU"""
        tmp1 = ((psi[id]-mmin[id]) /
                (mmax[id]-mmin[id])*255)
        tmp1[tmp1 > 255] = 255
        tmp1[tmp1 < 0] = 0
        tmp2 = ((g[id]-mmin[id]) /
                (mmax[id]-mmin[id])*255)
        tmp2[tmp2 > 255] = 255
        tmp2[tmp2 < 0] = 0

        flow[id] = cv2.calcOpticalFlowFarneback(
            tmp1.astype('uint8'), tmp2.astype('uint8'), flow[id], *pars)  # updates flow

    def registration_flow_batch(self, psi, g, mmin, mmax, flow=None, pars=[0.5, 3, 20, 16, 5, 1.1, 4]):
        """Find optical flow for all projections in parallel on CPU"""
        if (flow is None):
            flow = np.zeros([*psi.shape, 2], dtype='float32')
        total = 0
        for ids in chunk(range(self.ntheta), self.ptheta):
            flownew = flow[ids]
            with cf.ThreadPoolExecutor(16) as e:
                # update flow in place
                e.map(partial(self.registration_flow, psi[ids], g[ids], mmin[ids],
                              mmax[ids], flownew, pars), range(len(ids)))

            # control Farneback's (may diverge for small window sizes)
            # err = np.linalg.norm(
            #    g[ids]-self.apply_flow_gpu_batch(psi[ids], flownew), axis=(1, 2))

            # err1 = np.linalg.norm(
            #    g[ids]-self.apply_flow_gpu_batch(psi[ids], flow[ids]), axis=(1, 2))

            #idsgood = np.where(err1 >= err)[0]

            g_gpu = cp.array(g[ids])
            psi_gpu = cp.array(psi[ids])
            flownew_gpu = cp.array(flownew)
            flow_gpu = cp.array(flow[ids])
            err = cp.linalg.norm(
                g_gpu-self.apply_flow_gpu(psi_gpu, flownew_gpu, 0), axis=(1, 2))
            err1 = cp.linalg.norm(
                g_gpu-self.apply_flow_gpu(psi_gpu, flow_gpu, 0), axis=(1, 2))

            idsgood = cp.where(err1 >= err)[0].get()
            total += len(idsgood)
            flow[ids[idsgood]] = flownew[idsgood]
        print('bad alignment for:', self.ntheta-total)
        return flow

    def line_search(self, minf, gamma, psi, Dpsi, d, Td):
        """Line search for the step sizes gamma"""
        while(minf(psi, Dpsi)-minf(psi+gamma*d, Dpsi+gamma*Td) < 0):
            gamma *= 0.5
        if(gamma < 0.125):
            gamma = 0
        return gamma

    def apply_flow_gpu(self, f, flow, gpu):

        h, w = flow.shape[1:3]
        flow = -flow.copy()
        flow[:, :, :, 0] += cp.arange(w)
        flow[:, :, :, 1] += cp.arange(h)[:, cp.newaxis]

        
        g = f.copy()  # keep values that were not affected
        # g = cp.zeros([self.ptheta,self.nz,self.n],dtype='float32')
        
        g = cp.ascontiguousarray(g)
        f = cp.ascontiguousarray(f)
        flowx = cp.ascontiguousarray(flow[:, :, :, 0])
        flowy = cp.ascontiguousarray(flow[:, :, :, 1])
        self.remap(g.data.ptr, f.data.ptr, 
            flowx.data.ptr, flowy.data.ptr, gpu)


        #     flowx = cp.asarray(flow[:, :, :, 0],order='C')
        # flowy = cp.asarray(flow[:, :, :, 1],order='C')
        # g = f.copy()  # keep values that were not affected

        # # g = cp.zeros([self.ptheta,self.nz,self.n],dtype='float32')

        # self.remap(g.data.ptr, f.data.ptr, 
        #     flowx.data.ptr, flowy.data.ptr, gpu)
        return g

    def apply_flow_gpu_batch(self, f, flow):
        res = np.zeros([f.shape[0], self.nz, self.n], dtype='float32')
        f_gpu = cp.zeros([self.ptheta, self.nz, self.n], dtype='float32')
        flow_gpu = cp.zeros([self.ptheta, self.nz, self.n, 2], dtype='float32')
        for ids in chunk(range(f.shape[0]), self.ptheta):
            # copy data part to gpu
            f_gpu = cp.array(f[ids])
            flow_gpu = cp.array(flow[ids])
            # Radon transform
            res_gpu = self.apply_flow_gpu(f_gpu, flow_gpu, 0)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def cg_deform(self, data, psi, flow, diter, xi1=0, rho=0, gpu=0):
        """CG solver for deformation"""
        # minimization functional
        def minf(psi, Dpsi):
            f = cp.linalg.norm(Dpsi-data)**2+rho*cp.linalg.norm(psi-xi1)**2
            return f

        for i in range(diter):
            Dpsi = self.apply_flow_gpu(psi, flow, gpu)
            grad = (self.apply_flow_gpu(Dpsi-data, -flow, gpu) +
                    rho*(psi-xi1))/max(rho, 1)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Td = self.apply_flow_gpu(d, flow, gpu)
            gamma = 0.5*self.line_search(minf, 1, psi, Dpsi, d, Td)
            if(gamma == 0):
                break
            grad0 = grad
            # update step
            psi = psi + gamma*d
            # check convergence
            # if (0):
            #     print("%4d, %.3e, %.7e" %
            #           (i, gamma, minf(psi, Dpsi+gamma*Td)))
        return psi

    def cg_deform_multi_gpu(self, data, psi, flow,  diter, xi1, rho, lock, ids):
        """Pick GPU, copy data, run reconstruction"""
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()
        cp.cuda.Device(gpu).use()

        data_gpu = cp.array(data[ids])
        psi_gpu = cp.array(psi[ids])
        xi1_gpu = cp.array(xi1[ids])
        flow_gpu = cp.array(flow[ids])
        # Radon transform
        psi_gpu = self.cg_deform(
            data_gpu, psi_gpu, flow_gpu, diter, xi1_gpu, rho, gpu)
        # copy result to cpu
        psi[ids] = psi_gpu.get()

        BUSYGPUS[gpu] = 0

        return psi[ids]

    def cg_deform_gpu_batch(self, data, init, flow, diter, xi1=0, rho=0, dbg=False):

        psi = init.copy()
        ids_list = chunk(range(self.ntheta), self.ptheta)
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for psii in e.map(partial(self.cg_deform_multi_gpu, data, psi, flow, diter, xi1, rho, lock), ids_list):
                psi[np.arange(0, psii.shape[0])+shift] = psii
                shift += psii.shape[0]
        cp.cuda.Device(0).use()

        return psi

    def grad_deform_gpu(self, data, psi, flow, diter, xi1=0, rho=0, gpu=0):
        """G solver for deformation"""
        # minimization functional
        def minf(psi, Dpsi):
            f = cp.linalg.norm(Dpsi-data)**2 + rho*cp.linalg.norm(psi-xi1)**2
            return f

        minf1 = 1e15
        for i in range(diter):
            Dpsi = self.apply_flow_gpu(
                psi.real, flow, gpu)+1j*self.apply_flow_gpu(psi.imag, flow, gpu)
            grad = self.apply_flow_gpu((Dpsi-data).real, -flow, gpu)+1j * \
                self.apply_flow_gpu((Dpsi-data).imag, -flow, gpu) + rho*(psi-xi1)
            r = min(1, 1/rho)/2.0
            grad *= r
            # update step
            psi = psi + 0.5*(-grad)
            
            # check convergence
            Dpsi = self.apply_flow_gpu(
                psi.real, flow, gpu)+1j*self.apply_flow_gpu(psi.imag, flow, gpu)
            minf0 = minf(psi, Dpsi)
            if(minf0 > minf1):
                print('error in deform', minf0, minf1)
            minf1 = minf0
            #print('d',i,minf0)
        return psi

    def grad_deform_multi_gpu(self, data, psi, flow,  diter, xi1, rho, lock, ids):
        """Pick GPU, copy data, run reconstruction"""
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()
        cp.cuda.Device(gpu).use()

        data_gpu = cp.array(data[ids])
        psi_gpu = cp.array(psi[ids])
        xi1_gpu = cp.array(xi1[ids])
        flow_gpu = cp.array(flow[ids])
        # Radon transform
        psi_gpu = self.grad_deform_gpu(
            data_gpu, psi_gpu, flow_gpu, diter, xi1_gpu, rho, gpu)
        # copy result to cpu
        psi[ids] = psi_gpu.get()

        BUSYGPUS[gpu] = 0

        return psi[ids]

    def grad_deform_gpu_batch(self, data, init, flow, diter, xi1=0, rho=0, dbg=False):

        psi = init.copy()
        ids_list = chunk(range(self.ntheta), self.ptheta)
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for psii in e.map(partial(self.grad_deform_multi_gpu, data, psi, flow, diter, xi1, rho, lock), ids_list):
                psi[np.arange(0, psii.shape[0])+shift] = psii
                shift += psii.shape[0]
        cp.cuda.Device(0).use()

        return psi

    def flowplot(self, u, psi, flow, fname):
        """Visualize 4 aligned projections, corrsponding optical flows, 
        and slices through reconsruction, save figure as a png file"""
        matplotlib.use('Agg')
        [ntheta, nz, n] = psi.shape

        plt.figure(figsize=(10, 7))
        plt.subplot(3, 4, 1)
        plt.imshow(psi[ntheta//4], cmap='gray')

        plt.subplot(3, 4, 2)
        plt.imshow(psi[ntheta//2], cmap='gray')
        plt.subplot(3, 4, 3)
        plt.imshow(psi[3*ntheta//4], cmap='gray')

        plt.subplot(3, 4, 4)
        plt.imshow(psi[-1].real, cmap='gray')

        plt.subplot(3, 4, 5)
        plt.imshow(flow_to_color(flow[ntheta//4]), cmap='gray')

        plt.subplot(3, 4, 6)
        plt.imshow(flow_to_color(flow[ntheta//2]), cmap='gray')

        plt.subplot(3, 4, 7)
        plt.imshow(flow_to_color(flow[3*ntheta//4]), cmap='gray')
        plt.subplot(3, 4, 8)
        plt.imshow(flow_to_color(flow[-1]), cmap='gray')

        plt.subplot(3, 4, 9)
        plt.imshow(u[nz//2], cmap='gray')
        plt.subplot(3, 4, 10)
        plt.imshow(u[nz//2+nz//8], cmap='gray')

        plt.subplot(3, 4, 11)
        plt.imshow(u[:, n//2-35], cmap='gray')

        plt.subplot(3, 4, 12)
        plt.imshow(u[:, :, n//2-35], cmap='gray')
        plt.savefig(fname)
        plt.close()
