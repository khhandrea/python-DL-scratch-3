if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Function

def f(x: Variable) -> Variable:
    y = x**4 - 2*x**2
    return y

# 2nd derivative of f(x)
# (1st derivative : from back propagation)
def gx2(x: np.ndarray) -> np.ndarray:
    y = 12*x**2 - 4
    return y

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print('x :', x)

    y = f(x)

    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
