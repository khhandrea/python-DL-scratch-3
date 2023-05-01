import numpy as np
from numpy import ndarray
from typing import Callable

class Variable:
    def __init__(self, data: ndarray):
        self.data = data
        self.grad = None
        self.creator = None

    # def set_creator(self, func: Function) -> None:
    def set_creator(self, func) -> None:
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, in_data: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def backward(self, gy) -> ndarray:
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x: ndarray) -> ndarray:
        y = x ** 2
        return y
    
    def backward(self, gy: ndarray) -> ndarray:
        x = self.input.data
        gx = gy * 2 * x
        return gx

class Exp(Function):
    def forward(self, x: ndarray) -> ndarray:
        return np.exp(x)
    
    def backward(self, gy: ndarray) -> ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)