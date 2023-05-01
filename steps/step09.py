import numpy as np
from numpy import ndarray
from typing import Callable

class Variable:
    def __init__(self, data: ndarray):
        if data is not None:
            if not isinstance(data, ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None

    # def set_creator(self, func: Function) -> None:
    def set_creator(self, func) -> None:
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x) -> ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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

def square(x: Variable) -> Variable:
    return Square()(x)

def exp(x: Variable) -> Variable:
    return Exp()(x)

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(f'x.grad : {x.grad}')

x = np.array(1.0)
x = Variable(None)
x = Variable(1.0)