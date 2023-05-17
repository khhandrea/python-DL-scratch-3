if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero import functions as F

x = Variable(np.array())