CUDA_VISIBLE_DEVICES=0 nohup python -u test_standard.py True 0 >res0 &
CUDA_VISIBLE_DEVICES=1 nohup python -u test_standard.py 0 1 >res1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u test_standard.py 0 2 >res2 &
CUDA_VISIBLE_DEVICES=3 nohup python -u test_standard.py 0 3 >res3 &
