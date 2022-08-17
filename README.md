# Neural Network CUDA Example
![logo](./image/logo.png)

Several simple examples for neural network toolkits (PyTorch, TensorFlow, etc.) calling custom CUDA operators.

We provide several ways to compile the CUDA kernels and their cpp wrappers, including jit, setuptools and cmake.

We also provide several python codes to call the CUDA kernels, including kernel time statistics and model training.

*For more accurate time statistics, you'd best use **nvprof** or **nsys** to run the code.*

## Environments
* NVIDIA Driver: 418.116.00
* CUDA: 11.0
* Python: 3.7.3
* PyTorch: 1.7.0+cu110
* TensorFlow: 2.4.1
* CMake: 3.16.3
* Ninja: 1.10.0
* GCC: 8.3.0

*Cannot ensure successful running in other environments.*

## Code structure
```shell
.
├── add2.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── build
│   ├── bdist.linux-x86_64
│   ├── lib.linux-x86_64-cpython-38
│   │   ├── add2.cpython-38-x86_64-linux-gnu.so
│   │   └── nn_cuda.cpython-38-x86_64-linux-gnu.so
│   └── temp.linux-x86_64-cpython-38
│       ├── build.ninja
│       ├── kernel
│       │   ├── add2_kernel.o
│       │   └── compute_distance.o
│       └── pytorch
│           ├── add2_ops.o
│           └── nn_cuda_ops.o
├── dist
│   ├── add2-0.0.0-py3.8-linux-x86_64.egg
│   └── nn_cuda-0.0.0-py3.8-linux-x86_64.egg
├── image
│   └── logo.png
├── include
│   └── nn_cuda.h
├── kernel
│   ├── add2_kernel.cu
│   └── compute_distance.cu
├── LICENSE
├── nn_cuda.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── pytorch
│   ├── nn_cuda_ops.cpp
│   └── setup.py
└── README.md

13 directories, 25 files

```
## PyTorch
### Compile cpp and cuda
**JIT**  
Directly run the python code.

**Setuptools**  
```shell
python3 pytorch/setup.py install
```

