#!/usr/bin/bash
#SBATCH -n 40
#SBATCH -p v100
##SBATCH --exclude gn1
#SBATCH --mem 160G
#SBATCH -t 40:00:00
# module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign_develop2/experimental/soil
# python cg.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T10_22micro_0001
python cg.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T5_36micro_0002
python cg.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_33micro_0001
python cg.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_24macro_0001


# python admm.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T10_22micro_0001
# python admm.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T5_36micro_0002
# python admm.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_33micro_0001
# python admm.py /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_24macro_0001


# python cg.py  /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T10_31macro_0001
# python cg.py  /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_24macro_0001

# python admm.py  /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T10_22micro_0001

# python admm.py  /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T10_31macro_0001

# python admm.py  /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T15_38macro;

# python admm.py  /data/staff/tomograms/vviknik/tomoalign_vincent_data/soil/T30_24macro_0001
