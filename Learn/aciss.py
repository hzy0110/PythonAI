from PIL import Image
import argparse
import scipy
import numpy
import matplotlib

import scipy.optimize
import sympy
from sympy import *


a = symbols('a')
print(solve(Eq(2 ** a, 64), a))


print(scipy.optimize.fsolve(lambda x: 2+x, 3))

print(2**6)


