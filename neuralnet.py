import numpy as np
from PIL import Image
import random

class Network:
    def __init__(self, layer_sizes, activation_function='sigmoid'):
        '''
        THE ONLY OPTIONS FOR ACTIVATION FUNCTION ARE 'sigmoid' AND 'relu'
        '''

        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y,1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(layer_sizes[:-1], layer_sizes[1:])]
        # self note for index chasing: it is flipped since y is next layer, x is this layer, and therefore there are x in each inner array since there are x neurons that go into each neuron in the y layer
        self.activation_function=activation_function

    def feedforward(self, vector):
        '''
        returns the output of the network given an input vector

        ACTIVATION: the activation function!!
        '''
        for b, w in zip(self.biases, self.weights):
            vector = self.eval((np.dot(w, vector) + b))

        return vector
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        '''
        training_data is list of tuples (x,y) of training inputs and desired output

        will be evaluated against test_data if test_data is provided
        '''
        if test_data: test_data_length = len(test_data)
        training_data_length = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, training_data_length, mini_batch_size)] # thanks python for ignoring out of bounds lol
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print("Epoch {} accuracy: {}/{}".format(j, self.evaluate(test_data), test_data_length))
            else:
                print("Epoch {} complete".format(j))

        #print(self.biases)
        #np.save('trainedbiases', self.biases)
        #np.save('trainedweights', self.weights)

    
    def update_mini_batch(self, mini_batch, learning_rate):
        '''
        updates network (the actual class itself) weights and biases based on a single mini batch
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases] # why not np.zeros(self.biases.shape) ?
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # backprop on a single training example
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b-((learning_rate/len(mini_batch))*nb) for b, nb in zip(self.biases, nabla_b)] # update weights and biases normalized around size of the batch
        self.weights = [w-((learning_rate/len(mini_batch))*nw) for w, nw in zip(self.weights, nabla_w)] 

    def backprop(self, x, y):
        '''
        returns a gradient vector of bias and weight given x is input and y is desired output
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases] # same question as in update mini batch
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward
        activation = x
        activations = [x] # stores each activation (which just means output of each individual neuron) layer by layer
        zs = [] # list of all z, which are the wa+b without being plugged into the activation function
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b 
            # print(z)
            zs.append(z)
            #activation = activation_function.eval(z)
            activation = self.eval(z)
            activations.append(activation)

        # backward
        delta = self.cost_derivative(activations[-1], y) * self.eval_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l is minus indexing 
        for l in range(2, self.num_layers):
            z = zs[-l]
            ap = self.eval_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * ap 
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    
    def identify(self, image):
        return np.argmax(self.feedforward(image))
    
    def identify_from_path(self, path):
        return np.argmax(self.feedforward(np.array([[x] for x in np.ndarray.flatten(np.array(Image.open(path)))])/255))
    
    def cost_derivative(self, output, y):
        '''
        output = list of output 'activations'
        '''
        # print(output)
        return (self.normalize(output)-y)
    
    def eval(self, z):
        if self.activation_function=='relu':
            return z * (z > 0)
        else:
            return 1.0/(1.0+np.exp(-z))

    def eval_prime(self, z):
        if self.activation_function=='relu':
            return 1.*(z>0)
        else:
            return self.eval(z)*(1-self.eval(z))
        
    def normalize(self, x):
        return x/(max(1,np.max(x)))

    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return Network.sigmoid(z)*(1-Network.sigmoid(z))

    
