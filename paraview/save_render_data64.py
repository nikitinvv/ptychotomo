#/local/vnikitin/software/ParaView-5.6.0-MPI-Linux-64bit/bin/pvpython --force-offscreen-rendering save_render_data.py
from shutil import copyfile
import os
from paraview.simple import *
arr = os.listdir("/home/beams/VNIKITIN/data_ptycho/data_noise64")
print(arr)
for f in arr:
    copyfile("/home/beams/VNIKITIN/data_ptycho/data_noise64/"+f,"tmp_data.tiff") 
    LoadState("datas64.pvsm")
    WriteImage(os.path.splitext("/home/beams/VNIKITIN/data_ptycho/png/data_noise64/"+f)[0]+'.png')

#for f in *.png; do  echo "Converting $f"; convert -trim "$f" "$f"; done