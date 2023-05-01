import numpy as np
from numpy import ndarray
from typing import Callable

class Variable:
    def __init__(self, data: ndarray):
        self.data = data

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, in_data: ndarray) -> ndarray:
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x: ndarray) -> ndarray:
        return x ** 2

class Exp(Function):
    def forward(self, x: ndarray) -> ndarray:
        return np.exp(x)
    
def numerical_diff(f: Callable[[Variable], Variable],
                   x: Variable,
                   eps: float=1e-4) -> ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return A(B(C(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)