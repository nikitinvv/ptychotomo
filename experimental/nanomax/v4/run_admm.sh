#!/bin/bash
source /home/vvnikitin/anaconda3/etc/profile.d/conda.sh
conda activate tomoalign

if [[ $1 -eq 1 ]]
then
    NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 3500 192 1 1
fi
if [[ $1 -eq 1 ]]
then
    NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 3500 192 0 1
fi
if [[ $1 -eq 1 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 1700 192 1 2
fi
if [[ $1 -eq 1 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 1700 192 0 2
fi
if [[ $1 -eq 1 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 900 192 1 4
fi
if [[ $1 -eq 1 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 900 192 0 4
fi
if [[ $1 -eq 1 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 500 192 1 8
fi
if [[ $1 -eq 1 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py /gdata/RAVEN/vnikitin/nanomax/ 500 192 0 8
fi
    