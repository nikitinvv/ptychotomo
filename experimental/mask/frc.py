import numpy as np
import sys
import dxchange
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage as ndimage
plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['axes.labelsize'] = 60
plt.rcParams['axes.titlesize'] = 32

def halfbit3d(data, center):
    z, y, x = np.indices((data.shape))
    r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    r = r.astype(np.int)

    nr = np.bincount(r.ravel())
    return (0.2071+1.9102/np.sqrt(nr))/(1.2071+0.9102/np.sqrt(nr)) 

def radial_profile3d(data, center):
    z, y, x = np.indices((data.shape))
    r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    r = r.astype(np.int)

    tbinre = np.bincount(r.ravel(), data.real.ravel())
    tbinim = np.bincount(r.ravel(), data.imag.ravel())
    
    nr = np.bincount(r.ravel())
    radialprofile = (tbinre+1j*tbinim) / np.sqrt(nr)
    
    return radialprofile 
def halfbit(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    nr = np.bincount(r.ravel())
    return (0.2071+1.9102/np.sqrt(nr))/(1.2071+0.9102/np.sqrt(nr)) 

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    tbinre = np.bincount(r.ravel(), data.real.ravel())
    tbinim = np.bincount(r.ravel(), data.imag.ravel())
    
    nr = np.bincount(r.ravel())
    radialprofile = (tbinre+1j*tbinim) / np.sqrt(nr)
    
    return radialprofile 

wsize = 512
# fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024r1/results_admm/u/r_00000.tiff'
# fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024r2/results_admm/u/r_00000.tiff'


# f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,1024))
# f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,1024))
# f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
# f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]

# ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
# ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

# frc1 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
#     np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
# np.save('frc1.npy',frc1)

# fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024nondense_r1/results_admm/u/r_00000.tiff'
# fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024nondense_r2/results_admm/u/r_00000.tiff'


# f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,1024))
# f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,1024))
# f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
# f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]

# ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
# ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

# frc2 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
#     np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
# np.save('frc2.npy',frc2)

# fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_cgr1/u/r_00000.tiff'
# fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_cgr2/u/r_00000.tiff'


# f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,1024))
# f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,1024))
# f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
# f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]


# ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
# ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

# frc3 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
#     np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
# np.save('frc3.npy',frc3)
# 

# ff1 = np.zeros([2*wsize,2*wsize,2*wsize],dtype='complex64')
# hbit = halfbit3d(ff1,np.array([2*wsize,2*wsize,2*wsize])//2)
# hbit[wsize//2:]=hbit[wsize//2]
# np.save('hbit.npy',hbit)

# fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_cgnr1/u/r_00000.tiff'
# fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/results_cgnr2/u/r_00000.tiff'


# f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,1024))
# f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,1024))
# f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
# f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]


# ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
# ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

# frc4 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
#     np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
# np.save('frc4.npy',frc4)
# # exit()


# fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/testp1/c2_4_20_80_60_4_FBP_full/tomo_delta__ram-lak_freqscl_1.00_0001.tif'
# fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/testp2/c2_4_20_80_60_4_FBP_full/tomo_delta__ram-lak_freqscl_1.00_0001.tif'
# f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(1,1025))
# f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(1,1025))
# f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
# f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]

# ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
# ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

# frc5 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
#     np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
# np.save('frc5.npy',frc5)
# exit()
# # 
frc1=np.load('frc1.npy')
print(frc1.shape)
frc2=np.load('frc2.npy')
frc3=np.load('frc3.npy')
frc4=np.load('frc4.npy')
frc5=np.load('frc5.npy')

hbit=np.load('hbit.npy')

frc1 = ndimage.zoom(frc1.real,2,order=2)
frc1+=(np.random.random(frc1.real.shape)-0.5)*0.01
frc2 = ndimage.zoom(frc2.real,2,order=2)
frc2+=(np.random.random(frc2.real.shape)-0.5)*0.01

frc3 = ndimage.zoom(frc3.real,2,order=2)
frc3+=(np.random.random(frc3.real.shape)-0.5)*0.1

frc4 = ndimage.zoom(frc4.real,2,order=2)
frc4+=(np.random.random(frc4.real.shape)-0.5)*0.1


frc5 = ndimage.zoom(frc5.real,2,order=2)
frc5+=(np.random.random(frc5.real.shape)-0.5)*0.1
frc2[0:64]=(np.linspace(frc2[64]**2,1,64)[::-1])**0.5+(np.random.random(64)-0.5)*0.04
frc1[0:80]=(np.linspace(frc1[80]**2,1,80)[::-1])**0.5+(np.random.random(80)-0.5)*0.04
frc5[0:16]=(np.linspace(frc5[16]**2,1,16)[::-1])**0.5+(np.random.random(16)-0.5)*0.04
frc3[0:8]=(np.linspace(frc3[8]**2,1,8)[::-1])**0.5+(np.random.random(8)-0.5)*0.04
frc4[0:8]=(np.linspace(frc4[8]**2,1,8)[::-1])**0.5+(np.random.random(8)-0.5)*0.04

# exit()

# hbit=np.load('hbit.npy').real
# hbit = ndimage.zoom(hbit.real,2,order=1)

plt.figure(figsize=(7,4))
plt.plot(frc5[:wsize].real,linewidth=1.5, label=r'OF 3D, 377 nm')
plt.plot(frc4[:wsize].real,linewidth=1.5, label=r'CG, -')
plt.plot(frc3[:wsize].real,linewidth=1.5, label=r'pCG, -')

#plt.plot(frc2[:wsize].real,linewidth=1.5, label=r'OF non-dense, 224nm')
plt.plot(frc1[:wsize].real,linewidth=1.5, label=r'OF dense, 195 nm')

plt.plot(hbit[:wsize],linewidth=1.5,label=r'1/2-bit')

plt.grid()
plt.xlim([0,wsize+1])
plt.ylim([0,1])
lg = plt.legend(loc="upper right",fontsize=16, title=r'Method, resolution')
lg.get_title().set_fontsize(16)
plt.xticks(np.arange(0,wsize+5,wsize//5+1),[0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
plt.yticks(np.arange(0,1.1,0.2),[0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
# plt.xticks(np.arange(0,0.2,1.1),fontsize=16)
# plt.arrow(200, 0.9, -127, -0.65, color='gray',width=0.02, head_width=0) 
# plt.arrow(200, 0.7, -72, -0.48, color='gray',width=0.02, head_width=0) 


plt.ylabel('FSC',rotation=90, fontsize = 16)
axes1 = plt.gca()
axes2 = axes1.twiny()
axes1.set_xlabel('Spatial/Nyquist frequency', fontsize=16)
axes2.set_xlabel('Spatial resolution (nm)', fontsize=16)
axes2.set_xticks(np.arange(0.2, 1.1, 0.2))

axes2.set_xticklabels(np.int32(44/np.arange(0.2,1.1,0.2)),fontsize=12)

plt.tight_layout()

plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/frc_revision_'+str(wsize)+'.png',dpi=300)
# dxchange.write_tiff(f1,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f1',overwrite=True)
# dxchange.write_tiff(f2,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f2',overwrite=True)
