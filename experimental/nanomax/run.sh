
#!/bin/bash
for k in {0..173..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_ptycho.py $(($k+$j)) 1 &        
    done
    wait
done