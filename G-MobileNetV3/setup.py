# setup.py

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("ATT/_hidden_impl.pyx", compiler_directives={'language_level': "3"}),
)