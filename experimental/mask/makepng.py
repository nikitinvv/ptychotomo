import dxchange
import numpy as np
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
# from timing import tic,toc
import gc
import scipy.ndimage as ndimage
import cv2
# from flowvis import *

plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

# u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_admm_reg1e-05/u/r_00000.tiff', ind=np.arange(0,256))
# u=  u[:,u.shape[1]//2-612//4:u.shape[1]//2+612//4,u.shape[2]//2-612//4:u.shape[2]//2+612//4]
# print(u.shape)
# dxchange.write_tiff_stack(u,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/volume/r')
# exit()

# fname = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024'
# binning = 2
# ndsets = 7
# nth = 100
# data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
# theta = np.zeros(ndsets*nth,dtype='float32')
# for k in range(ndsets):
#     data[k*nth:(k+1)*nth] = np.load(fname+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
#     theta[k*nth:(k+1)*nth] = np.load(fname+'_theta'+str(k)+'.npy').astype('float32')
# data[np.isnan(data)]=0           
# data-=np.mean(data)
# for k in range(7):
#     dxchange.write_tiff(data[k*100]-data[0], '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/proj/d/d_0000'+str(k),overwrite=True)
#     dxchange.write_tiff(data[k*100], '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/proj/r/r_0000'+str(k),overwrite=True)
# exit()    

# u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_admm/u/r_00000.tiff',ind=np.arange(0,1024))
# ucg = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_cg/u/r_00000.tiff',ind=np.arange(0,1024))
# ucgn = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/revision_cg_nop_1/results/u/r_00000.tiff',ind=np.arange(0,1024))
# un = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_admm_reg3e-06/u/r_00000.tiff',ind=np.arange(0,1024))
upsi3 = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/c2_4_64_80_60_4_FBP_full/tomo_delta__ram-lak_freqscl_1.00_0001.tif',ind=np.arange(1,1025))
#upsi7 = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/c7_4_64_80_60_4_SART_full/tomo_delta__ram-lak_freqscl_1.00_0001.tif',ind=np.arange(1,1025))

# u=  u[:,u.shape[1]//2-612:u.shape[1]//2+612,u.shape[2]//2-612:u.shape[2]//2+612]
# un=un[:,un.shape[1]//2-612:un.shape[1]//2+612,un.shape[2]//2-612:un.shape[2]//2+612]
upsi3=upsi3[:,upsi3.shape[1]//2-612:upsi3.shape[1]//2+612,upsi3.shape[2]//2-612:upsi3.shape[2]//2+612]

# vmin=-0.0018
# vmax=0.0018
# a=u[u.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/uz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=u[:,u.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/uy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=u[:,:,u.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/ux.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# # u=[]

# a=ucg[ucg.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/ucgz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucg[:,ucg.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/ucgy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucg[:,:,ucg.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/ucgx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# # ucg=[]
# a=ucgn[ucg.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/ucgnz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucgn[:,ucg.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/ucgny.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucgn[:,:,ucg.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/ucgnx.png',a,vmin=vmin,vmax=vmax,cmap='gray')


# a=un[un.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/unz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=un[:,un.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/uny.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=un[:,:,un.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/unx.png',a,vmin=vmin,vmax=vmax,cmap='gray')


vmin=22000
vmax=32768*2-26000
a=upsi3[upsi3.shape[0]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/upsi3z.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=upsi3[:,upsi3.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/upsi3y.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=upsi3[:,:,upsi3.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/upsi3x.png',a,vmin=vmin,vmax=vmax,cmap='gray')

# a=upsi7[upsi7.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/upsi7z.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=upsi7[:,upsi7.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/upsi7y.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=upsi7[:,:,upsi7.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/upsi7x.png',a,vmin=vmin,vmax=vmax,cmap='gray')

# un=[]
exit()

flow = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_admm/flow.npy')
for j in range(2,11,1):
    for k in range(0,700,100):
        plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/flow'+str(k)+str(j)+'.png',flow_to_color(flow[k]/j))

flown = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_admm/flow.npy')/4
for k in range(0,700,100):
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/flown'+str(k)+'.png',flow_to_color(flown[k]))

#print('std:',np.std(u[128:-128,128:-128,128:-128]),np.std(ucg[128:-128,128:-128,128:-128]),np.std(un[128:-128,128:-128,128:-128]))
#print('snr:',np.mean(u[128:-128,128:-128,128:-128])/np.std(u[128:-128,128:-128,128:-128]),np.mean(ucg[128:-128,128:-128,128:-128])/np.std(ucg[128:-128,128:-128,128:-128]),np.mean(un[128:-128,128:-128,128:-128])/np.std(un[128:-128,128:-128,128:-128]))
#print('snr2:',np.mean(u[128:-128,128:-128,128:-128])**2/np.std(u[128:-128,128:-128,128:-128])**2,np.mean(ucg[128:-128,128:-128,128:-128])**2/np.std(ucg[128:-128,128:-128,128:-128])**2,np.mean(un[128:-128,128:-128,128:-128])**2/np.std(un[128:-128,128:-128,128:-128])**2)

plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[-1.8e-3,1.8e-3]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[-1.8e-3, 0, 1.8e-3])
cb.ax.tick_params(labelsize=22)
cb.ax.set_xticklabels([r'\textbf{-1.8e-3}', r'\textbf{0.0}', r'\textbf{1.8e-3}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/hcolorbarobj.png",bbox_inches = 'tight')
plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[-0.3,0.3]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[-0.3, 0.0, 0.3])
cb.ax.tick_params(labelsize=22)
cb.ax.set_xticklabels([r'\textbf{-0.3}', r'\textbf{0.0}', r'\textbf{0.3}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/figs/hcolorbarproj.png",bbox_inches = 'tight')