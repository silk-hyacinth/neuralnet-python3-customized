import numpy as np
from PIL import Image

class relu:
    def eval(x):
        return x * (x > 0)

    def eval_prime(x):
        return 1.*(x>0)
        
class sigmoid:
    def eval(x):
        return 1.0/(1.0+np.exp(-x))

    def eval_prime(x):
        sigmoid.eval(x)*(1-sigmoid.eval(x))

class test:
    def __init__(self, x):
        self.x=x

    def do(self, activation):
        return activation.eval_prime(self.x)
    

#t=test(-1)

# print(t.do(relu))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# print(relu.eval_prime(np.array([[1,0],[3,4]])))
target = "pictures/test/"

img = Image.open(target + "0/3.jpg")
print(np.array(img))
