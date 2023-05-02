if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def sphere(x: Variable, y: Variable) -> Variable:
    z = x ** 2 + y ** 2
    return z

def matyas(x: Variable, y: Variable) -> Variable:
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z

def goldstein_price(x: Variable, y: Variable) -> Variable:
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *\
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))

z = goldstein_price(x, y)
z.backward()
print('<goldstein_price(1.0, 1.0)>')
print('x.grad:', x.grad)
print('y.grad:', y.grad)