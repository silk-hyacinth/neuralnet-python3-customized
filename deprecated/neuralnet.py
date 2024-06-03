import numpy as np
import random

class Network:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y,1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(layer_sizes[:-1], layer_sizes[1:])]
        # self note for index chasing: it is flipped since y is next layer, x is this layer, and therefore there are x in each inner array since there are x neurons that go into each neuron in the y layer

    def feedforward(self, vector, activation_function):
        '''
        returns the output of the network given an input vector

        ACTIVATION: the activation function!!
        '''
        for b, w in zip(self.biases, self.weights):
            vector = activation_function.eval((np.dot(w, vector) + b))

        return vector
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, activation_function, test_data=None):
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
                self.update_mini_batch(mini_batch, learning_rate, activation_function)

            if test_data:
                print("Epoch {} accuracy: {}/{}".format(j, self.evaluate(test_data, activation_function), test_data_length))
            else:
                print("Epoch {} complete".format(j))

    
    def update_mini_batch(self, mini_batch, learning_rate, activation_function):
        '''
        updates network (the actual class itself) weights and biases based on a single mini batch
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases] # why not np.zeros(self.biases.shape) ?
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, activation_function) # backprop on a single training example
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b-((learning_rate/len(mini_batch))*nb) for b, nb in zip(self.biases, nabla_b)] # update weights and biases normalized around size of the batch
        self.weights = [w-((learning_rate/len(mini_batch))*nw) for w, nw in zip(self.weights, nabla_w)] 

    def backprop(self, x, y, activation_function):
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
            activation = relu.eval(z)
            activations.append(activation)

        # backward
        delta = self.cost_derivative(activations[-1], y) * activation_function.eval_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l is minus indexing 
        for l in range(2, self.num_layers):
            z = zs[-l]
            #print(zs)
            ap = activation_function.eval_prime(z)
            #print("ap", ap)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * ap 
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data, activation_function):
        test_results = [(np.argmax(self.feedforward(x, activation_function)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    def cost_derivative(self, output, y):
        '''
        output = list of output 'activations'
        '''
        return (output-y)


class relu:
    def eval(x):
        return x * (x > 0)

    def eval_prime(x):
        return 1.*(x>0)
    

class sigmoid:
    def eval(x):
        return 1.0/(1.0+np.exp(-x))

    def eval_prime(x):
        (1.0/(1.0+np.exp(-x)))*(1-(1.0/(1.0+np.exp(-x))))


    
