import numpy
from scipy.special import expit
import matplotlib.pyplot as plt

# NetWork Framework
class NN:

    # init function
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, reg, ):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.reg = reg
        
        # init weight between input layer and hidden layer, hidden layer and output layer, the weights obbey normal distribution.
        self.weight_ih = numpy.random.normal(0.0, pow(self.hnodes, 0.5), (self.hnodes, self.inodes))
        self.weight_ho = numpy.random.normal(0.0, pow(self.onodes, 0.5), (self.onodes, self.hnodes))
        
        # activation function
        self.activation = lambda x: expit(x)

    # NetWork training
    def train(self, input_list, target_list):
        # init inputs and target
        inputs = numpy.array(input_list, ndmin=2).T
        target = numpy.array(target_list, ndmin=1)

        # calculate hidden layer inputs and hidden layer outputs
        hidden_inputs = numpy.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        # calculate final layer inputs and outputs
        final_inputs = numpy.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        
        # calculate final layer errors and hidden layer errors, use BackPropagation algorithm.
        output_errors = target - final_outputs
        hidden_errors = numpy.dot(self.weight_ho.T, output_errors)
        
        # update weight
        self.weight_ho += self.lr * (numpy.dot(output_errors * final_outputs * (1 - final_outputs), hidden_outputs.T) + self.reg * numpy.sum(numpy.sign(self.weight_ho))) 
        self.weight_ih += self.lr * (numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T) + self.reg * numpy.sum(numpy.sign(self.weight_ih)))
    
    # Query NN
    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = numpy.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        return final_outputs