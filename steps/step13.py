from typing import Callable, List, Union
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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                
                if x.creator is not None:
                    funcs.append(x.creator)

def as_array(x) -> ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs: Variable) -> List[Variable]:
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        [output.set_creator(self) for output in outputs]
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs: List[ndarray]) -> Union[ndarray, List[ndarray]]:
        raise NotImplementedError()
    
    def backward(self, gys: List[ndarray]) -> List[ndarray]:
        raise NotImplementedError()

class Square(Function):
    def forward(self, x: ndarray) -> ndarray:
        y = x ** 2
        return y

    def backward(self, gy: ndarray) -> ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)

class Add(Function):
    def forward(self, x0: ndarray, x1:ndarray) -> ndarray:
        y = x0 + x1
        return y
    
    def backward(self, gy:ndarray) -> List[ndarray]:
        return gy, gy
    
def add(x0, x1):
    return Add()(x0, x1)

x = Variable(np.array(2))
y = Variable(np.array(3))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)