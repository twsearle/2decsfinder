#!/bin/bash

# This compiles this dirty fortran routine into a shared object
# so that python can use it.

# f2py should come with numpy/scipy

python -m numpy.f2py -c RStransform.f95 -m RStransform

