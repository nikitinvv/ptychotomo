#!/bin/bash
# source /home/vvnikitin/anaconda3/etc/profile.d/conda.sh
# conda activate tomoalign

python read_data.py /gdata/RAVEN/vnikitin/nanomax/
python sort.py /gdata/RAVEN/vnikitin/nanomax/
for k in {0..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_full.py /gdata/RAVEN/vnikitin/nanomax/ $(($k+$j)) 2000 1 16 &                
    done
    wait
done
python prealign_sift1.py /gdata/RAVEN/vnikitin/nanomax/ 2000


for k in {0..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop.py /gdata/RAVEN/vnikitin/nanomax/ $(($k+$j)) 2700 1 64 &                
    done
    wait
done
python prealign_sift2.py /gdata/RAVEN/vnikitin/nanomax/ 2700

for k in {0..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop_final.py /gdata/RAVEN/vnikitin/nanomax/  $(($k+$j)) 3500 1 32 &                
    done
    wait
done

