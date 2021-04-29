for k in {0..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop.py $(($k+$j)) 5400 1 512 &                
    done
    wait
done

# python find_center.py 4000
