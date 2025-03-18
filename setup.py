from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "mcts_cy",  # Output module name
        ["MCTS.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  # Maximum optimization
    )
]

setup(
    name="mcts_cy",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
        }
    )
)