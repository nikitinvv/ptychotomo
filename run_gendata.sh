#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -J ptychotomo_gendata
#SBATCH -p v100
#SBATCH --exclude gn1
##SBATCH --nodelist gn2
#SBATCH --mem 165G

#SBATCH -o gendata.out
#SBATCH -e gendata.err

module add GCC/8.2.0-2.31.1 CUDA/10.1.105

source ~/.bashrc-2019-04-27
source activate ptychotomo


python -u gendata.py $1 >resb$1$2 &
#python -u gendata.py 1 >resb1 &
#python -u gendata.py 2 >resb2 &
#python -u gendata.py 2 >resb2 &
wait
