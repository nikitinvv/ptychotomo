import numpy as np 
import dxchange
import sys
import matplotlib.pyplot as plt

name = sys.argv[1]
id = sys.argv[2]
a = dxchange.read_tiff_stack(name+'/r_00000.tiff',ind=np.arange(0,512))[:,64:-64,64:-64]
print(a.shape)
print(np.linalg.norm(a))
m = np.mean(a[128:-128,128:-128,128:-128])
s = np.std(a[128:-128,128:-128,128:-128])
a[a>m+2.5*s]=m+2.5*s
a[a<m-2.5*s]=m-2.5*s
plt.imsave('figs/z'+str(id)+'.png',a[a.shape[0]//2],cmap='gray')
plt.imsave('figs/y'+str(id)+'.png',a[:,a.shape[1]//2],cmap='gray')
plt.imsave('figs/x'+str(id)+'.png',a[:,:,a.shape[2]//2],cmap='gray')
