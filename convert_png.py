from __future__ import print_function
import dxchange
import os
import scipy.misc
import numpy as np


# files = os.listdir('gendata')
# for name in files:
# 	b = dxchange.read_tiff('gendata/'+name)[0,:,:]
# 	a = b.copy()
# 	a[a==0]=1
# 	print(np.amax(np.log(a)))
# 	scipy.misc.toimage(np.log(a), cmin=0, cmax=9).save('png/gendata/'+str(os.path.splitext(name)[0])+'.png')



files = os.listdir('old/delta')
for name in files:
	a = dxchange.read_tiff('old/delta/'+name)[24,45:-40,45:-40]
	scipy.misc.toimage(a, cmin=0, cmax=4e-5).save('png/delta/'+str(os.path.splitext(name)[0])+'.png')


files = os.listdir('old/beta')
for name in files:
	a = dxchange.read_tiff('old/beta/'+name)[24,45:-40,45:-40]
	scipy.misc.toimage(a, cmin=0, cmax=3e-6).save('png/beta/'+str(os.path.splitext(name)[0])+'.png')


