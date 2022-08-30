from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from setuptools import setup, Extension
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
setup(
    name="nn_cuda",
    include_dirs=["include","/usr/local/include/eigen3"],  
    library_dirs = ["/home/wangchenjun/open3d_install/lib"],
    
    # runtime_library_dirs=["/home/wangchenjun/code/NTracking/Open3D-0.14.1/build/lib/Release"],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "nn_cuda",
            ["pytorch/nn_cuda_ops.cpp", "kernel/add2_kernel.cu","kernel/compute_distance.cu","kernel/graph.cu"],
            # include_dirs=["include"," /home/wangchenjun/open3d_install/include","/usr/local/include/eigen3"],
            
            library_dirs = ["/home/wangchenjun/open3d_install/lib"],
            include_dirs = ['/home/wangchenjun/open3d_install/include'],
            # extra_ldflags=['libOpen3D.so'],
            libraries = ['Open3D'],
            extra_compile_args={'cxx': ['-g'],
                                        'nvcc': ['--extended-lambda']} ,    
        ),

        # Extension(
        #     "nn_cuda2",
        #     [ "kernel/add2_kernel.cu"],
        #     include_dirs=["include"," /home/wangchenjun/open3d_install/include","/usr/local/include/eigen3"],
        #     library_dirs = ["/home/wangchenjun/open3d_install/lib"],
        #     extra_compile_args={'cxx': ['-static'],
        #                          'nvcc': ['-O2']}   
        # ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)