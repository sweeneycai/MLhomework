import numpy
from scipy.special import expit
import matplotlib.pyplot as plt

# NetWork Framework
class NN:

    # init function
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, reg):
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
    
def main():
    input_nodes = 2
    hidden_nodes = 200
    output_nodes = 1
    learningrate = 0.1
    reg = 0.00000000001

    n = NN(input_nodes, hidden_nodes, output_nodes, learningrate, reg)

    data_file = open('shuffle_data.dat', 'r')
    raw_data = data_file.readlines()
    data_file.close()

    train_data = raw_data[0:64]
    test_data = raw_data[64:]

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # 处理读取到的数据，将其分割为x，y两组数据
    for data in train_data:
        data = data.split('   ')
        train_x.append([data[0], data[1]])
        train_y.append(data[2])

    for data in test_data:
        data = data.split('   ')
        test_x.append([data[0], data[1]])
        test_y.append(data[2])

    # 对验证集的真实结果进行处理，使其更易读
    test_y = numpy.asfarray(test_y).tolist()
    test_y = [int(y) for y in test_y]


    # 对训练集的真实结果进行处理
    train_y = numpy.asfarray(train_y)

    # training network
    epoches = 10000
    for i in range(epoches):
        for [j, x] in enumerate(train_x):
            inputs = numpy.asfarray(x) / 100
            targets = train_y[j]
            n.train(inputs, targets)

    output = []
    predict = []
    for i in test_x:
        output.append(n.query(numpy.asfarray(i) / 100))
        if (n.query(numpy.asfarray(i) / 100)) > 0.5:
            predict.append(int(1))
        else:
            predict.append(int(0))
    
    print(output)
    print(test_y)
    print(predict)

    predict_true = 0
    for i in range(16):
        if test_y[i] == predict[i]:
            predict_true += 1
    
    print(predict_true / 16)
    
    train_y = train_y.tolist()
    train_y = [int(y) for y in train_y] 

    predict_training_set = []
    for i in train_x:
        if (n.query(numpy.asfarray(i) / 100)) > 0.5:
            predict_training_set.append(int(1))
        else:
            predict_training_set.append(int(0))

    training_true = 0
    for i in range(64):
        if predict_training_set[i] == train_y[i]:
            training_true += 1
    
    print(training_true / 64)


if __name__ == '__main__':
    main()