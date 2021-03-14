# #!/usr/bin/bash
# #SBATCH -n 64
# #SBATCH -p v100
# #SBATCH --exclude gn1
# #SBATCH --mem 160G
# #SBATCH -t 80:00:00
# nvidia-smi
# module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
#cd /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/
# python admm.py 1 5 200 0 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
# python admm.py 1 5 200 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
# python admm.py 1 10 200 0 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
# python admm.py 1 10 200 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
# python admm.py 2 10 200 0 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
# python admm.py 2 10 200 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
# python admm.py 3 10 200 0 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
# python admm.py 3 10 200 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098
#

python admm94.py 1 10 200 0 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_4000prj_3s_094
python admm94.py 1 5 200 0 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_4000prj_3s_094
python admm94.py 1 15 200 0 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_4000prj_3s_094
