import dxchange
import numpy as np
import scipy.ndimage as ndimage

n=128
nz=128
f = dxchange.read_tiff_stack('lspci/psi_00000.tiff',ind=np.arange(0,166))   
frec = f*0 
for k in range(f.shape[0]):
    a = f[k]#.copy()
    a[a<4e-5]=0
    cm = ndimage.center_of_mass(a)   
    print(cm)
    frec[k] = np.roll(f[k],(-int(cm[0]-nz//2+0.5),-int(cm[1]-n//2+0.5)),axis=(0,1))      
dxchange.write_tiff_stack(frec, 'clspci/psi', overwrite=True)         
dxchange.write_tiff_stack(f-frec, 'clspcidif/psi', overwrite=True)         
