from __future__ import print_function
import dxchange
import os
import scipy.misc

files = os.listdir('../rec_ptycho/delta_s2')
for name in files:
	a = dxchange.read_tiff('../rec_ptycho/delta_s2/'+name)
	scipy.misc.toimage(a, cmin=-0.00013467284/4, cmax=0.00013467284).save('../rec_ptycho/png/delta_s2'+str(os.path.splitext(name)[0])+'.png')

files = os.listdir('../rec_ptycho/beta_s2')
for name in files:
	a = dxchange.read_tiff('../rec_ptycho/beta_s2'+name)
	scipy.misc.toimage(a, cmin=-2.7094511e-05/4, cmax=2.7094511e-05).save('../rec_ptycho/png/beta_s2'+str(os.path.splitext(name)[0])+'.png')


