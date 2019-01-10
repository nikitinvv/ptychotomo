import dxchange
import os
arr = os.listdir("/home/beams/VNIKITIN/data_ptycho/max_iv2/delta")

for f in arr:
    a = dxchange.read_tiff("/home/beams/VNIKITIN/data_ptycho/max_iv2/delta/"+f).copy()
    a+=1e-5
    dxchange.write_tiff(a,"/home/beams/VNIKITIN/data_ptycho/max_iv2/deltam/"+f,overwrite=True)
    
# for f in *.png; do  echo "Converting $f"; convert -trim "$f" "$f"; done