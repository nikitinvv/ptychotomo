import tomopy
import numpy as np
import radonusfft
N = 128
Ntheta = 180
Ns = 128
theta = np.float32(np.arange(0,Ntheta)*np.pi/Ntheta)


#create class for the transform
cl = radonusfft.radonusfft(Ntheta,Ns,N)
cl.setobj(theta)
#swig does not work with complex numbers, so array sizes are doubled 2*N
f = np.float32(np.zeros([Ns,N,2*N]),dtype="float32")

# real part
f[:,:,::2] = tomopy.misc.phantom.shepp2d(size=N, dtype=u'float32')
# complex part
f[:,:,1::2] = np.fliplr(tomopy.misc.phantom.shepp2d(size=N, dtype=u'float32'))
#f[-1]*=0

# fwd
# memory for result
g = np.float32(np.zeros([Ntheta,Ns,2*N]),dtype="float32")
# run
cl.fwd(g,f)

# adj
# memory for result
ff = np.float32(np.zeros([Ns,N,2*N]),dtype="float32")
#run
cl.adj(ff,g)

f = np.float64(f)
ff = np.float64(ff)
g = np.float64(g)


# adj test
print((np.sum(ff*f)-np.sum(g*g))/np.sum(ff*f))


import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.imshow(np.squeeze(g[:,4,::2]))
plt.subplot(2,2,2)
plt.imshow(np.squeeze(g[:,4,1::2]))

plt.subplot(2,2,3)
plt.imshow(np.squeeze(ff[4,:,::2]))
plt.subplot(2,2,4)
plt.imshow(np.squeeze(ff[4,:,1::2]))

plt.show()

