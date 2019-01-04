from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
image = data.coins()+1j*data.coins()/10
fimage = fft.fftshift(fft.fft2(fft.ifftshift(np.exp(1j*image))))
fimage = fimage/np.amax(np.abs(fimage))*1000
fimagen = np.random.poisson(np.abs(fimage))*fimage/np.abs(fimage)

imagerec = np.log(fft.fftshift(fft.ifft2(fft.ifftshift(fimagen))))/1j


plt.subplot(2,2,1)
plt.imshow(image.real)
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(np.log(np.abs(fimage)))
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(imagerec.real)
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(imagerec.imag)
plt.colorbar()

plt.show()