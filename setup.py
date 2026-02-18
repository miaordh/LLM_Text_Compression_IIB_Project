# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='exact_softmax_cpp',
    ext_modules=[
        CppExtension('exact_softmax_cpp', ['exact_softmax.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)