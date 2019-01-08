nohup python -u test_s.py 1 0 >res0 &
CUDA_VISIBLE_DEVICES=4 nohup python -u test_s.py 0 1 >res1 &
CUDA_VISIBLE_DEVICES=6 nohup python -u test_s.py 0 2 >res2 &
CUDA_VISIBLE_DEVICES=5 nohup python -u test_s.py 0 3 >res3 &
