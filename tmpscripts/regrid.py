import dxchange
import numpy as np
import scipy 

sz = 256

beta = np.zeros([512+128,512+128,512+128],dtype='float32')
delta = np.zeros([512+128,512+128,512+128],dtype='float32')
beta[64:64+512,64:64+512,64:64+512] = dxchange.read_tiff(
    'data/BETA.tiff').astype('float32')
delta[64:64+512,64:64+512,64:64+512] = dxchange.read_tiff(
    'data/DELTA.tiff').astype('float32')

x=np.linspace(0,1,512+128)
y=np.linspace(0,1,512+128)
z=np.linspace(0,1,512+128)
[xi,yi,zi] = np.meshgrid(np.linspace(0,1,sz),np.linspace(0,1,sz),np.linspace(0,1,sz))


xi = np.ndarray.flatten(xi)
yi = np.ndarray.flatten(yi)
zi = np.ndarray.flatten(zi)

pts = np.array([yi,xi,zi]).swapaxes(0,1)

f = scipy.interpolate.RegularGridInterpolator((x, y, z), beta)
betap=f(np.array(pts))
betap = np.reshape(betap,[sz,sz,sz])
f = scipy.interpolate.RegularGridInterpolator((x, y, z), delta)
deltap=f(np.array(pts))
deltap = np.reshape(deltap,[sz,sz,sz])

print(betap.shape)
dxchange.write_tiff(betap.astype('float32'),  'data/betacirc-256',overwrite=True)
dxchange.write_tiff(deltap.astype('float32'),  'data/deltacirc-256',overwrite=True)




















# beta = np.zeros([80+48,80+48,80+48],dtype='float32')
# delta = np.zeros([80+48,80+48,80+48],dtype='float32')
# beta[24:24+80,24:24+80,24:24+80] = dxchange.read_tiff(
#     'data/beta-pad.tiff')[50:50+80,100:100+80,100:100+80].astype('float32')
# delta[24:24+80,24:24+80,24:24+80] = dxchange.read_tiff(
#     'data/delta-pad.tiff')[50:50+80,100:100+80,100:100+80].astype('float32')
   
# # x=np.linspace(0,1,96+32)
# # y=np.linspace(0,1,96+32)
# # z=np.linspace(0,1,96+32)
# # [xi,yi,zi] = np.meshgrid(np.linspace(0,1,256),np.linspace(0,1,256),np.linspace(0,1,256))


# # xi = np.ndarray.flatten(xi)
# # yi = np.ndarray.flatten(yi)
# # zi = np.ndarray.flatten(zi)

# # pts = np.array([yi,xi,zi]).swapaxes(0,1)

# # f = scipy.interpolate.RegularGridInterpolator((x, y, z), beta)
# # betap=f(np.array(pts))
# # betap = np.reshape(betap,[256,256,256])
# # f = scipy.interpolate.RegularGridInterpolator((x, y, z), delta)
# # deltap=f(np.array(pts))
# # deltap = np.reshape(deltap,[256,256,256])

# dxchange.write_tiff(beta.astype('float32'),  'data/beta-pad2-128',overwrite=True)
# dxchange.write_tiff(delta.astype('float32'),  'data/delta-pad2-128',overwrite=True)






