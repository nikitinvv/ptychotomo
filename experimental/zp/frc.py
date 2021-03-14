import numpy as np
import sys
import dxchange
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage as ndimage
# plt.rc('text', usetex=True)
# matplotlib.rc('font', family='serif', serif='cm10')
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
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

binning = np.int(sys.argv[1])
pre = sys.argv[2]
idslice = np.int(sys.argv[3])

wsize = 512//pow(2,binning)
fname1 = '/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090/dense_0p1/results_admm/u/r_00000.tiff'
fname2 = '/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090/dense_0p2/results_admm/u/r_00000.tiff'

f1 = dxchange.read_tiff_stack(fname1, ind = (idslice,idslice+1))[0]
f2 = dxchange.read_tiff_stack(fname2, ind = (idslice,idslice+1))[0]

# fname1 = '/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090/r1_1227c512/results_admm/u/r_00000.tiff'
# fname2 = '/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090/r2_1227c512/results_admm/u/r_00000.tiff'

# f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,512))#[0]
# f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,512))#[0]
# dxchange.write_tiff_stack(f1.swapaxes(0,1),'/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090_nfalign2nonalignedrecp1r/r',overwrite=True)
# dxchange.write_tiff_stack(f2.swapaxes(0,1),'/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090_nfalign2nonalignedrecp1r/r',overwrite=True)
# exit()
f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize]
f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize]

ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

frc1 = radial_profile(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile(np.abs(ff2)**2,np.array(ff1.shape)//2))
# np.save('frc1.npy',frc1)


hbit = halfbit(ff1,np.array(ff1.shape)//2)
# np.save('halfbit.npy',hbit)

plt.figure(figsize=(12,6))
print(idslice)
plt.subplot(1,2,1)
plt.plot(frc1[:wsize].real,linewidth=1.5, label=r'OF dense')
plt.plot(hbit[:wsize],linewidth=1.5,label=r'1/2-bit')

plt.grid()
plt.xlim([0,wsize+1])
plt.ylim([0,1])
lg = plt.legend(loc="upper right",fontsize=16, title=r'Method, resolution')
lg.get_title().set_fontsize(16)
plt.xticks(np.int32(np.arange(0.05,1.01,0.05)*100)/100*(wsize+1),np.int32(np.arange(0.05,1.01,0.05)*100)/100,fontsize=8)
plt.yticks(np.arange(0,1.1,0.2),[0,0.2,0.4,0.6,0.8,1.0],fontsize=10)


plt.ylabel('FSC',rotation=90, fontsize = 16)
axes1 = plt.gca()
plt.title('slice '+str(idslice))
axes2 = axes1.twiny()
axes1.set_xlabel('Spatial/Nyquist frequency', fontsize=16)
axes2.set_xlabel('Spatial resolution (nm)', fontsize=16)
axes2.set_xticks(np.int32(np.arange(0.05,1.01,0.05)*100)/100)

axes2.set_xticklabels(np.int32(590*pow(2,binning)/np.arange(0.05,1.01,0.05))/100,fontsize=8)
plt.subplot(1,2,2)
mmean = np.mean(f1)
mstd = np.std(f1)
plt.imshow(f1,cmap='gray',clim=[mmean-2*mstd,mmean+2*mstd])
plt.colorbar()

plt.tight_layout()

plt.savefig('/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090/frc'+pre+str(binning)+'_'+str(wsize)+'_'+str(idslice)+'.png',dpi=300)
# dxchange.write_tiff(f1,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f1',overwrite=True)
# dxchange.write_tiff(f2,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f2',overwrite=True)
