for k in {8..174..16}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop.py $(($k+$j)) 2700 1 32 &                
    done
    wait
done

# python find_center.py 4000
