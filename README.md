# ptychotomo
A CuPy and CUDA FFT based library for ptychography and tomography ADMM solvers.

## Installation from source
```bash
export CUDACXX=path-to-cuda-nvcc
pip install .
```

## Dependency
CuPy - for GPU acceleration of linear algebra operations in iterative schemes.
See (https://cupy.chainer.org/). For installation use

```bash
conda install cupy
```

## Tests
Run python test.py 
