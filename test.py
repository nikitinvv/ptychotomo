import matplotlib.pyplot as plt
import tomopy
import numpy as np
import radonusfft
N = 128
Ntheta = 128
Ns = 16
theta = np.float32(np.arange(0, Ntheta)*np.pi/Ntheta)


# create class for the transform
cl = radonusfft.radonusfft(Ntheta, Ns, N)
cl.setobj(theta)
# swig does not work with complex numbers, so array sizes are doubled 2*N
f = np.zeros([Ns, N, N], dtype="complex64")

# real part
f[:] = tomopy.misc.phantom.shepp2d(size=2*N, dtype=u'float32')[0, N//2:3*N//2, N//2:3*N//2]+1j*np.fliplr(
    tomopy.misc.phantom.shepp2d(size=2*N, dtype=u'float32'))[0, N//2:3*N//2, N//2:3*N//2]

# fwd
# memory for result
g = np.zeros([Ntheta, Ns, N], dtype="complex64")
# run
cl.fwd(g, f)

# adj
# memory for result
ff = np.zeros([Ns, N, N], dtype="complex64")
# run
cl.adj(ff, g)

f = np.complex128(f)
ff = np.complex128(ff)
g = np.complex128(g)


# adj test
print((np.sum(f*np.conj(ff))-np.sum(g*np.conj(g)))/np.sum(f*np.conj(ff)))


plt.subplot(2, 2, 1)
plt.imshow(np.squeeze(g[:, 4, :].real))
plt.subplot(2, 2, 2)
plt.imshow(np.squeeze(g[:, 4, :].imag))

plt.subplot(2, 2, 3)
plt.imshow(np.squeeze(ff[4, :, :].real))
plt.subplot(2, 2, 4)
plt.imshow(np.squeeze(ff[4, :, :].imag))

plt.show()
