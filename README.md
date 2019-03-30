#Install ptychotomo, e.g.

## Clone the repository

$ git clone https://github.com/math-vrn/ptychotomo

cd ptychotomo

## Install dependencies using conda

conda env create -n ptychotomo -f envs/python-3.6.yaml

## Install ptychotomo

python setup.py install

#Run test on gpu with id=0

python test.py 0

#Notes:

#Error: ....undefined symbol: __intel_sse2_strcpy
#If using intel compilers during the compilation then python binary requires
#intel's libraries when running ptychotomo, otherwise above error happens.
#To eliminate this requirement above setup.py file can be executed with gcc:

#$ CC=gcc python setup.py install; cd -
