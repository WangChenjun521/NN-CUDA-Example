from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="nn_cuda",
    include_dirs=["include","/home/wangchenjun/open3d_install/include","/usr/local/include/eigen3"],  

    # library_dirs = ['/home/wangchenjun/code/NTracking/Open3D-0.14.1/build/lib/Release'],
    # runtime_library_dirs=["/home/wangchenjun/code/NTracking/Open3D-0.14.1/build/lib/Release"],

    ext_modules=[
        CUDAExtension(
            "nn_cuda",
            ["pytorch/nn_cuda_ops.cpp", "kernel/add2_kernel.cu","kernel/compute_distance.cu","kernel/graph.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)