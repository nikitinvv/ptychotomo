CUDA_VISIBLE_DEVICES=0 nohup python -u test_noise.py 0 0 >res0 &
CUDA_VISIBLE_DEVICES=1 nohup python -u test_noise.py 0 2 >res1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u test_noise.py 0 3 >res2 &
