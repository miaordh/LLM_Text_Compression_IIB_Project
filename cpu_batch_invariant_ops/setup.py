from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cpu_batch_invariant_ops',
    ext_modules=[
        CppExtension(
            name='cpu_batch_invariant_ops_c',
            sources=['csrc/cpu_batch_invariant.cpp'],
            extra_compile_args=['-O3']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
