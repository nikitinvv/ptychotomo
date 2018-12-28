# ptycho

#specify path to CUDA:

export CUDAHOME=/sw/pkg/cuda_x86/cuda-9.1

#then run:

git clone https://github.com/math-vrn/ptychotomo

git clone https://github.com/math-vrn/ptychofft

git clone https://github.com/math-vrn/radonusfft

cd ptychofft; python setup.py install; cd - 

cd radonusfft; python setup.py install; cd - 

cd ptycho

python test.py
