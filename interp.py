import dxchange
import numpy as np
import scipy as sc
from skimage.transform import resize    
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)    
    
beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')#[::2, ::2, ::2]
delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')#[::2, ::2, ::2]

#delta1 = resize(delta,(512,512,512),anti_aliasing=True).astype('float32')
#beta1 = resize(beta,(512,512,512),anti_aliasing=True).astype('float32')
delta1 = denoise_tv_chambolle(delta, weight=1e-5, multichannel=False)
beta1 = denoise_tv_chambolle(beta, weight=1e-5, multichannel=False)

dxchange.write_tiff(delta1,
        'data/test-delta-512.tiff')
dxchange.write_tiff(beta1,
        'data/test-beta-512.tiff')