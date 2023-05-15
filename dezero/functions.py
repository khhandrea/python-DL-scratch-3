import numpy as np
from numpy import ndarray
from dezero.core import Variable, Function

class Sin(Function):
    def forward(self, x: Variable) -> Variable:
        y = np.sin(x)
        return y
    
    def backward(self, gy: Variable) -> None:
        x, = self.inputs
        gx = gy * np.cos(x)
        return gx
    
def sin(x: Variable) -> Variable:
    return Sin()(x)

class Cos(Function):
    def forward(self, x: Variable) -> Variable:
        y = np.cos(x)
        return y
    
    def backward(self, gy: Variable) -> None:
        x, = self.inputs
        gx = gy * -np.sin(x)
        return gx
    
def cos(x: Variable) -> Variable:
    return Cos()(x)