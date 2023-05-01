from typing import Callable, List
import numpy as np
from numpy import ndarray

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
    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        xs = [input.data for input in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        [output.set_creator(self) for output in outputs]
        self.inputs = inputs
        self.outputs = outputs
        return outputs
    
    def forward(self, xs: List[ndarray]) -> List[ndarray]:
        raise NotImplementedError()
    
    def backward(self, gys: List[ndarray]) -> List[ndarray]:
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y, )
    
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)