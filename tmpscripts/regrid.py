import dxchange
import numpy as np
import scipy 
beta = np.zeros([512+64,512+64,512+64],dtype='float32')
delta = np.zeros([512+64,512+64,512+64],dtype='float32')
beta[32:32+512,32:32+512,32:32+512] = dxchange.read_tiff(
    'data/BETA.tiff').astype('float32')
delta[32:32+512,32:32+512,32:32+512] = dxchange.read_tiff(
    'data/DELTA.tiff').astype('float32')
   
x=np.linspace(0,1,512+64)
y=np.linspace(0,1,512+64)
z=np.linspace(0,1,512+64)
[xi,yi,zi] = np.meshgrid(np.linspace(0,1,256),np.linspace(0,1,256),np.linspace(0,1,256))


xi = np.ndarray.flatten(xi)
yi = np.ndarray.flatten(yi)
zi = np.ndarray.flatten(zi)

pts = np.array([yi,xi,zi]).swapaxes(0,1)

f = scipy.interpolate.RegularGridInterpolator((x, y, z), beta)
betap=f(np.array(pts))
betap = np.reshape(betap,[256,256,256])
f = scipy.interpolate.RegularGridInterpolator((x, y, z), delta)
deltap=f(np.array(pts))
deltap = np.reshape(deltap,[256,256,256])

dxchange.write_tiff(betap.astype('float32'),  'data/BETA256',overwrite=True)
dxchange.write_tiff(deltap.astype('float32'),  'data/DELTA256',overwrite=True)