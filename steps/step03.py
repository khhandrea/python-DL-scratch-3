import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, in_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)
    
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)