#!/usr/bin/bash
#SBATCH -n 40
#SBATCH -p v100
##SBATCH --exclude gn1
#SBATCH --mem 160G
#SBATCH -t 20:00:00
# module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign_develop2/experimental/nmc
python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Wolfman/LMR-NMC_925C_8600eV_Interlaced_1201prj_087
python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Wolfman/LMR-NMC_925C_8600eV_Interlaced_1201prj_087

# python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Wolfman/LMR-NMC_950C_8600eV_Interlaced_1201prj_097
# python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Wolfman/LMR-NMC_950C_8600eV_Interlaced_1201prj_097

# python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Wolfman/LMR-NMC_950C_8600eV_Interlaced_1201prj_107
# python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Wolfman/LMR-NMC_950C_8600eV_Interlaced_1201prj_107
