#/local/vnikitin/software/ParaView-5.5.2-Qt5-MPI-Linux-64bit/bin/pvpython --force-offscreen-rendering save_render_beta.py
from shutil import copyfile
import os
from paraview.simple import *
arr = os.listdir("/home/beams/VNIKITIN/maxiv_rec/beta")
for f in arr:
    copyfile("/home/beams/VNIKITIN/maxiv_rec/beta/"+f,"tmp.tiff") 
    LoadState("betas.pvsm")
    WriteImage(os.path.splitext("/home/beams/VNIKITIN/data_ptycho/png/beta/"+f)[0]+'.png')

#for f in *.png; do  echo "Converting $f"; convert -trim "$f" "$f"; done