from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy  # Make sure numpy is installed

extensions = [
    Extension(
        "MCTS",
        ["MCTS.pyx"],
        language="c++",  # This is the crucial change
        include_dirs=[numpy.get_include()]  # Include numpy headers
    )
]

setup(
    name="MCTS",
    ext_modules=cythonize(extensions),
)