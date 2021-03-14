import dxchange
import numpy as np
import matplotlib.pyplot as plt
import sys

name = sys.argv[1]
nameout = sys.argv[2]
u = dxchange.read_tiff_stack(name,ind=np.arange(0,700))
plt.imsave(nameout+'z.png',u[u.shape[0]//2],vmin=float(sys.argv[3]),vmax=float(sys.argv[4]),cmap='gray')
plt.imsave(nameout+'y.png',u[:,u.shape[1]//2-16],vmin=float(sys.argv[3]),vmax=float(sys.argv[4]),cmap='gray')
plt.imsave(nameout+'x.png',u[:,:,u.shape[2]//2+16],vmin=float(sys.argv[3]),vmax=float(sys.argv[4]),cmap='gray')
