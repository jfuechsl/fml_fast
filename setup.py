from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        "filter/_cumsum.pyx",
        "fracdiff/_fracdiff.pyx",
        "sampleweights/_sequentialbootstrap.pyx"
    ]),
)
