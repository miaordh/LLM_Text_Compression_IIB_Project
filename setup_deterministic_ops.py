from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="deterministic_ops_cpp",
    ext_modules=[
        CppExtension("deterministic_ops_cpp", ["deterministic_ops.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
