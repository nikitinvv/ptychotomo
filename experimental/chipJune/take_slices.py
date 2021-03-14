
import dxchange
import numpy as np
import dxchange
from scipy.ndimage import rotate
import sys
import matplotlib.pyplot as plt
if __name__ == "__main__":

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    #mul=2
    binning=0
    data = dxchange.read_tiff_stack(in_file+'/r_00000.tiff', ind=[691,657,672])
    
      #bin0 cutes    
    vmin = -0.001*2/3
    vmax = 0.002*2/3
    plt.imsave(out_file+'2000bin0p1.png',data[0],vmin=vmin,vmax=vmax,cmap='gray')
    plt.imsave(out_file+'2000bin0p2.png',data[1],vmin=vmin,vmax=vmax,cmap='gray')
    plt.imsave(out_file+'2000bin0p3.png',data[2],vmin=vmin,vmax=vmax,cmap='gray')