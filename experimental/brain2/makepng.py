import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from timing import tic,toc
import gc
import scipy.ndimage as ndimage
import cv2
import deformcg as dc
plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

# u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/fw__2880/rect20/r_00000.tiff',ind=np.arange(0,1024))
# ucg = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/cg__2880/rect3/r_00000.tiff',ind=np.arange(0,1024))
# ucg = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/cga__2880_median1x1x1/r0000.tif',ind=np.arange(0,1024))
#ucgn = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/revision_cg_nop_1/results/u/r_00000.tiff',ind=np.arange(0,1024))

#un = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/shiftfw__2880/rect4/r_00000.tiff',ind=np.arange(0,1024))
# upsi3 = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/4_4_20_80_60_4_FBP_full/tomo_delta__ram-lak_freqscl_1.00_0001.tif',ind=np.arange(1,1025))
#upsi7 = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/c7_4_64_80_60_4_SART_full/tomo_delta__ram-lak_freqscl_1.00_0001.tif',ind=np.arange(1,1025))

# u=  u[:,u.shape[1]//2-612:u.shape[1]//2+612,u.shape[2]//2-612:u.shape[2]//2+612]
# un=un[:,un.shape[1]//2-612:un.shape[1]//2+612,un.shape[2]//2-612:un.shape[2]//2+612]


# ucg = ndimage.filters.median_filter(ucg,3)
#ucgn = ndimage.filters.median_filter(ucgn,2)
# upsi3 = ndimage.filters.median_filter(upsi3,2)
# np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/tmp2',upsi3)
upsi3 = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/tmp.npy')
#vmin=0
#vmax=0.0016
# a=u[u.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/uz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=u[:,u.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/uy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=u[:,:,u.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/ux.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# # u=[]

# a=ucg[ucg.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/ucgz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucg[:,ucg.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/ucgy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucg[:,:,ucg.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/ucgx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# # ucg=[]

# a=un[un.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/unz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=un[:,un.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/uny.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=un[:,:,un.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/unx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# un=[]
# a=ucgn[ucgn.shape[0]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/ucgnz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucgn[:,ucgn.shape[1]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/ucgny.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# a=ucgn[:,:,ucgn.shape[2]//2];a[0]=vmin;a[1]=vmax
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/ucgnx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# # un=[]
vmin=40900
vmax=32768*2-21600
a=upsi3[upsi3.shape[0]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/upsi3z.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=upsi3[:,upsi3.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/upsi3y.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=upsi3[:,:,upsi3.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/upsi3x.png',a,vmin=vmin,vmax=vmax,cmap='gray')


exit()
flow = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/fw__2880/flownpy/20.npy')/3
for k in range(0,2880,180):
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/flow'+str(k)+'.png',dc.flowvis.flow_to_color(flow[k]))

flown = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/shiftfw__2880/flownpy/0.npy')/3
for k in range(0,2880,180):
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/flown'+str(k)+'.png',dc.flowvis.flow_to_color(flown[k]))

print('std:',np.std(u[128:-128,128:-128,128:-128]),np.std(ucg[128:-128,128:-128,128:-128]),np.std(un[128:-128,128:-128,128:-128]))
print('snr:',np.mean(u[128:-128,128:-128,128:-128])/np.std(u[128:-128,128:-128,128:-128]),np.mean(ucg[128:-128,128:-128,128:-128])/np.std(ucg[128:-128,128:-128,128:-128]),np.mean(un[128:-128,128:-128,128:-128])/np.std(un[128:-128,128:-128,128:-128]))
print('snr2:',np.mean(u[128:-128,128:-128,128:-128])**2/np.std(u[128:-128,128:-128,128:-128])**2,np.mean(ucg[128:-128,128:-128,128:-128])**2/np.std(ucg[128:-128,128:-128,128:-128])**2,np.mean(un[128:-128,128:-128,128:-128])**2/np.std(un[128:-128,128:-128,128:-128])**2)

plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[0,0.0012]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[0, 0.0006, 0.0012])
cb.ax.tick_params(labelsize=22)
cb.ax.set_xticklabels([r'\textbf{0}', r'\textbf{6.0e-4}', r'\textbf{1.2e-3}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/hcolorbarobj.png",bbox_inches = 'tight')
plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[-0.2,0.2]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[-0.2, 0.0, 0.2])
cb.ax.tick_params(labelsize=22)
cb.ax.set_xticklabels([r'\textbf{-0.2}', r'\textbf{0.0}', r'\textbf{0.2}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/figs/hcolorbarproj.png",bbox_inches = 'tight')