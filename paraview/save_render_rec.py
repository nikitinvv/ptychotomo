#/local/vnikitin/software/ParaView-5.6.0-MPI-Linux-64bit/bin/pvpython --force-offscreen-rendering save_render_rec.py
from shutil import copyfile
import os
from paraview.simple import *
# f = "beta128.tiff"
# copyfile("/home/beams/VNIKITIN/ptychotomo/data/"+f,"tmp_beta.tiff") 
# LoadState("betas2.pvsm")
# WriteImage(os.path.splitext("/home/beams/VNIKITIN/data_ptycho/png/beta/"+f)[0]+'.png')
# f = "delta128.tiff"
# copyfile("/home/beams/VNIKITIN/ptychotomo/data/"+f,"tmp_delta.tiff") 
# LoadState("deltas3.pvsm")
# WriteImage(os.path.splitext("/home/beams/VNIKITIN/data_ptycho/png/delta/"+f)[0]+'.png')
# arr = os.listdir("/home/beams/VNIKITIN/data_ptycho/max_iv2/beta")
# for f in arr:
#     copyfile("/home/beams/VNIKITIN/data_ptycho/max_iv2/beta/"+f,"tmp_beta.tiff") 
#     LoadState("betas2.pvsm")
#     WriteImage(os.path.splitext("/home/beams/VNIKITIN/data_ptycho/png/beta/"+f)[0]+'.png')

arr = os.listdir("/home/beams/VNIKITIN/data_ptycho/max_iv2/deltam")

for f in arr:
    copyfile("/home/beams/VNIKITIN/data_ptycho/max_iv2/deltam/"+f,"tmp_delta.tiff") 
    LoadState("deltas3.pvsm")
    WriteImage(os.path.splitext("/home/beams/VNIKITIN/data_ptycho/png/delta/"+f)[0]+'.png')
# for f in *.png; do  echo "Converting $f"; convert -trim "$f" "$f"; done