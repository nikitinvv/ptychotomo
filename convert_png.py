from __future__ import print_function
import dxchange
import os
import scipy.misc

files = os.listdir('delta')
for name in files:
	print(name)
	a = dxchange.read_tiff('delta/'+name)[7]
	print(a.shape)
	scipy.misc.toimage(a, cmin=-0.00013467284/4, cmax=0.00013467284).save('png/delta/'+str(os.path.splitext(name)[0])+'.png')

files = os.listdir('beta')
for name in files:
	a = dxchange.read_tiff('beta/'+name)[7]
	scipy.misc.toimage(a, cmin=-2.7094511e-05/4, cmax=2.7094511e-05).save('png/beta/'+str(os.path.splitext(name)[0])+'.png')


files = os.listdir('deltaml')
for name in files:
	a = dxchange.read_tiff('deltaml/'+name)[7]
	scipy.misc.toimage(a, cmin=-0.00013467284/4, cmax=0.00013467284).save('png/deltaml/'+str(os.path.splitext(name)[0])+'.png')

files = os.listdir('betaml')
for name in files:
	a = dxchange.read_tiff('betaml/'+name)[7]
	scipy.misc.toimage(a, cmin=-2.7094511e-05/4, cmax=2.7094511e-05).save('png/betaml/'+str(os.path.splitext(name)[0])+'.png')


