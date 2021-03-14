#!/usr/bin/bash
#SBATCH -n 40
#SBATCH -p v100
##SBATCH --exclude gn1
#SBATCH --mem 160G
#SBATCH -t 40:00:00
# module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign_develop2/experimental/mask_new
python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Gould/Fibers_Phase_1201prj_interlaced_1s_015
python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Gould/Fibers_Lower_part_Phase_1201prj_interlaced_1s_016
#python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Gould/Fibers_particles_Abs_1201prj_interlaced_1s_018
python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Gould/Fibers_particles_Phase_1201prj_interlaced_1s_017
