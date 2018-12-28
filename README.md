# ptycho
## set cuda path
export CUDAHOME=/sw/pkg/cuda_x86/cuda-9.1

git clone https://github.com/math-vrn/ptycho

git clone https://github.com/math-vrn/ptychofft

git clone https://github.com/math-vrn/radonusfft

cd ptychofft; python setup.py install; cd - 

cd radonusfft; python setup.py install; cd - 

cd ptycho

python test.py
