
#!/bin/bash
for k in {0..173..4}; 
do
    for j in {0..3}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_ptycho_full.py $(($k+$j)) &        
        # CUDA_VISIBLE_DEVICES=$j python rec_ptycho_crop_part.py $(($k+$j)) &        
    done
    wait
done