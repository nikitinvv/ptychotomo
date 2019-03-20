#specify path to CUDA, e.g.

export CUDAHOME=/sw/pkg/cuda_x86/cuda-9.1

#Install cupy, e.g.

pip install cupy-cuda91

#Install ptychography and tomography solvers:

git clone https://github.com/math-vrn/ptychotomo

git clone https://github.com/math-vrn/ptychofft

git clone https://github.com/math-vrn/radonusfft

cd ptychofft; python setup.py install; cd -

cd radonusfft; python setup.py install; cd -

cd ptychotomo

#Run test on gpu with id=0

python test.py 0


# Notes:

# Error: ....undefined symbol: __intel_sse2_strcpy
# If using intel compilers during the compilation then python binary requires
# intel's libraries when running ptychotomo, otherwise above error happens. 
# To eliminate this requirement above setup.py file can be executed with gcc:
#
# $ CC=gcc python setup.py install; cd -
