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
    __array_priority__ = 200

    def __init__(self, data: ndarray, name:str=None):
        if data is not None:
            if not isinstance(data, ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

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

def as_variable(obj: any) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x) -> ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs: Variable) -> List[Variable]:
        inputs = [as_variable(input) for input in inputs]
        
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
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0:ndarray, x1:ndarray) -> ndarray:
        y = x0 * x1
        return y
    
    def backward(self, gy:ndarray) -> List[ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gy0, gy1 = gy * x1, gy * x0
        return gy0, gy1
    
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x: ndarray) -> ndarray:
        return -x

    def forward(self, gy: ndarray) -> ndarray:
        return -gy
    
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0: ndarray, x1: ndarray) -> ndarray:
        y = x0 - x1
        return y
    
    def backward(self, gy: ndarray) -> ndarray:
        return gy, -gy
    
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Div(Function):
    def forward(self, x0: ndarray, x1: ndarray) -> ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: ndarray) -> ndarray:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0: ndarray, x1: ndarray) -> ndarray:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: ndarray, x1: ndarray) -> ndarray:
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c: float):
        self.c = c

    def forward(self, x: ndarray) -> ndarray:
        y = x ** self.c
        return y

    def backward(self, gy: ndarray) -> ndarray:
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow

x = Variable(np.array(2.0))
print('x:', x)

y = -x
print('-x:', y) 

y1 = 2.0 - x
y2 = x - 1.0
print('2.0 - x:', y1) 
print('x - 1.0:', y2) 

y = 3.0 / x
print('3.0 / x:', y) 

y = x ** 3
y.backward()
print('x^2:', y)
print('x.grad:', x.grad)