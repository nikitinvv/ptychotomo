#!/usr/bin/bash
# #SBATCH -n 40
# #SBATCH -p v100
# ##SBATCH --exclude gn1
# #SBATCH --mem 160G
# #SBATCH -t 40:00:00
# # module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
# cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign_develop2/experimental/zp

python admm.py 12 200 /local/data/vnikitin/Kenan_ZP_ROI3_8keV_interlaced_5000prj_2s_003 0 &
# python admm.py 12 200 /local/data/vnikitin/Kenan_ZP_8keV_interlaced_5000prj_3s_001 0 &

# python admm.py 24 200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/DeAndrade/16nmZP/Kenan_ZP_ROI2_8keV_interlaced_5000prj_3s_002 0 0;
# python admm.py 24 200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/DeAndrade/16nmZP/Kenan_ZP_ROI2_8keV_interlaced_5000prj_3s_002 1 0;

# python admm.py 24 200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/DeAndrade/16nmZP/Kenan_ZP_8keV_interlaced_5000prj_3s_001 0 0;
# python admm.py 24 200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/DeAndrade/16nmZP/Kenan_ZP_8keV_interlaced_5000prj_3s_001 1 0;