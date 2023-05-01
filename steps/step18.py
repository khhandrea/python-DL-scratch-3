import contextlib
import heapq # Backward is implemented with heapq differently than the book.
from typing import List, Union
import weakref
import numpy as np
from numpy import ndarray

class Config:
    enable_backdrop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backdrop', False)


class Variable:
    def __init__(self, data: ndarray):
        if data is not None:
            if not isinstance(data, ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    # def set_creator(self, func: Function) -> None:
    def set_creator(self, func) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, (-f.generation, f))
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            f = heapq.heappop(funcs)[1]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

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

        if Config.enable_backdrop:
            self.generation = max([x.generation for x in inputs])
            [output.set_creator(self) for output in outputs]
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def __lt__(self, other):
        return self.generation >= other.generation

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

x0 = Variable(np.array(2.0))
x1 = Variable(np.array(3.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad, x0.grad, x1.grad)

with no_grad():
    x = Variable(np.array(2.0))
    a = square(x)
    y = square(a)
    y.backward()
    print('x.grad:', x.grad)
    print('a.grad:', a.grad)
