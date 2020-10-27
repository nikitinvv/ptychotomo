##ecover_prb = sys.argv[1]
##swap_prb = sys.argv[2]
##align = sys.argv[3]    
##shake = sys.argv[4]    
##nmodes = int(sys.argv[5])

python testnanomax.py False False False False 2
python testnanomax.py False False True True 2
python testnanomax.py True True False False 2
python testnanomax.py True True True True 2


python testnanomax.py False False False False 1
python testnanomax.py False False True True 1
python testnanomax.py True True False False 1
python testnanomax.py True True True True 1
