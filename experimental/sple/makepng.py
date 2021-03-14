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
centers = {
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple1_Phase_1201prj_interlaced_1s_010': 1227,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple2_Phase_1201prj_interlaced_1s_011': 1237,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple3_Phase_1201prj_1s_009': 1217,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple4_Phase_1201prj_interlaced_1s_012': 1183,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple5_Phase_1201prj_interlaced_1s_013': 1202,
}
plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
fname = sys.argv[1]   
print(fname+'dense_1'+str(centers[fname]/2)+'/results_admm/u/r_00000.tiff')
u = dxchange.read_tiff_stack(fname+'/dense_1'+str(centers[fname]/2)+'/results_admm/u/r_00000.tiff',ind=np.arange(0,1024))
ucg = dxchange.read_tiff_stack(fname+'/results_cg/u/r_00000.tiff',ind=np.arange(0,1024))

u=  u[:,u.shape[1]//2-612:u.shape[1]//2+612,u.shape[2]//2-612:u.shape[2]//2+612]

vmin=-0.0018
vmax=0.0018
a=u[u.shape[0]//2];a[0]=vmin;a[1]=vmax
plt.imsave(fname+'/uz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=u[:,u.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave(fname+'/uy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=u[:,:,u.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave(fname+'/ux.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# u=[]

a=ucg[ucg.shape[0]//2];a[0]=vmin;a[1]=vmax
plt.imsave(fname+'/ucgz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=ucg[:,ucg.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave(fname+'/ucgy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=ucg[:,:,ucg.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave(fname+'/ucgx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
exit()

flow = np.load(fname+'results_admm/flow.npy')
for j in range(2,11,1):
    for k in range(0,700,100):
        plt.imsave(fname+'/flow'+str(k)+str(j)+'.png',flow_to_color(flow[k]/j))

flown = np.load(fname+'results_admm/flow.npy')/4
for k in range(0,700,100):
    plt.imsave(fname+'/flown'+str(k)+'.png',flow_to_color(flown[k]))

#print('std:',np.std(u[128:-128,128:-128,128:-128]),np.std(ucg[128:-128,128:-128,128:-128]),np.std(un[128:-128,128:-128,128:-128]))
#print('snr:',np.mean(u[128:-128,128:-128,128:-128])/np.std(u[128:-128,128:-128,128:-128]),np.mean(ucg[128:-128,128:-128,128:-128])/np.std(ucg[128:-128,128:-128,128:-128]),np.mean(un[128:-128,128:-128,128:-128])/np.std(un[128:-128,128:-128,128:-128]))
#print('snr2:',np.mean(u[128:-128,128:-128,128:-128])**2/np.std(u[128:-128,128:-128,128:-128])**2,np.mean(ucg[128:-128,128:-128,128:-128])**2/np.std(ucg[128:-128,128:-128,128:-128])**2,np.mean(un[128:-128,128:-128,128:-128])**2/np.std(un[128:-128,128:-128,128:-128])**2)

plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[-1.8e-3,1.8e-3]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[-1.8e-3, 0, 1.8e-3])
cb.ax.tick_params(labelsize=22)
cb.ax.set_xticklabels([r'\textbf{-1.8e-3}', r'\textbf{0.0}', r'\textbf{1.8e-3}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024//hcolorbarobj.png",bbox_inches = 'tight')
plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[-0.3,0.3]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[-0.3, 0.0, 0.3])
cb.ax.tick_params(labelsize=22)
cb.ax.set_xticklabels([r'\textbf{-0.3}', r'\textbf{0.0}', r'\textbf{0.3}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024//hcolorbarproj.png",bbox_inches = 'tight')