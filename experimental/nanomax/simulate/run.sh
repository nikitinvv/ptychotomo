##ecover_prb = sys.argv[1]
##swap_prb = sys.argv[2]
##align = sys.argv[3]    
##shake = sys.argv[4]    
##nmodes = int(sys.argv[5])




CUDA_VISIBLE_DEVICES=0 python testnanomax.py False False True True 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py False False True False 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py False False False False 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py False False False True 1 

# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True True True True 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True True True False 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True True False False 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True True False True 1 

# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True False True True 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True False True False 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True False False False 1 
# CUDA_VISIBLE_DEVICES=0 python testnanomax.py True False False True 1 

