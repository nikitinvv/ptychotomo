for k in {16..174..24}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop.py $(($k+$j)) 5400 1 128 &                
    done
    wait
done

# python find_center.py 4000
