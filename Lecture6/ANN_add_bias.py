import numpy
from scipy.special import expit
from shuffle import Data
from sklearn.model_selection import KFold


# NetWork Framework
class NN:

    # init function
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, reg):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.reg = reg
        
        # init weight between input layer and hidden layer, hidden layer and output layer,
        # the weights obey normal distribution.
        self.weight_ih = numpy.random.normal(0.0, pow(self.hnodes, 0.5), (self.hnodes, self.inodes))
        self.weight_ho = numpy.random.normal(0.0, pow(self.onodes, 0.5), (self.onodes, self.hnodes))
        self.bias_1 = numpy.ones((self.hnodes, 1)) / 10
        self.bias_2 = numpy.ones((self.onodes, 1)) / 10
        
        # activation function
        self.activation = lambda x: expit(x)

    # NetWork training
    def train(self, input_list, target_list):
        # init inputs and target
        inputs = numpy.array(input_list, ndmin=2).T
        print("shape of inputs:", inputs.shape)
        target = numpy.array(target_list, ndmin=1)
        print("shape of hi:", self.weight_ih.shape)

        # calculate hidden layer inputs and hidden layer outputs
        hidden_inputs = numpy.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs + self.bias_1)
        print("shape of hidden inputs:", hidden_inputs.shape)
        print("shape of b1:", self.bias_1.shape)
        # calculate final layer inputs and outputs
        final_inputs = numpy.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs + self.bias_2)

        # calculate final layer errors and hidden layer errors, use BackPropagation algorithm.
        output_errors = target - final_outputs
        hidden_errors = numpy.dot(self.weight_ho.T, output_errors)
        
        # update weight and bias simultaneously
        self.weight_ho += self.lr * (numpy.dot(output_errors * final_outputs * (1 - final_outputs), hidden_outputs.T)
                                     + self.reg * numpy.sum(numpy.sign(self.weight_ho)))
        print(final_outputs)
        print(target)
        self.bias_2 += (final_outputs - target) * final_outputs * (1 - final_outputs)
        self.weight_ih += self.lr * (numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)
                                     + self.reg * numpy.sum(numpy.sign(self.weight_ih)))
        self.bias_1 += self.bias_2 * self.weight_ho * hidden_outputs * (1 - hidden_outputs)

    # Query NN
    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = numpy.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        return final_outputs


def main():
    x, y = Data.getAllData()
    learning_rate = 0.3
    reg = 1e-8
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 1

    prediction = 0  # 最后预测结果
    kf = KFold(n_splits=5, shuffle=False)
    for [train_index, test_index] in kf.split(x):
        x_train, x_test, y_train, y_test = [], [], [], []

        for i in train_index:
            x_train.append(x[i])
            y_train.append(y[i])

        for i in test_index:
            x_test.append(x[i])
            y_test.append(y[i])

        for i in range(len(x_train)):
            x[i].insert(0, 100)

        epoch = 5000
        n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)
        for i in range(epoch):
            for [j, x_temp] in enumerate(x_train):
                print(x_temp)
                inputs = numpy.array(x_temp) / 100   # 添加偏置节点
                targets = y_train[j]
                n.train(inputs, targets)

        predict_true = 0
        result = []
        query_result = []
        for j in range(16):
            temp = n.query(numpy.array(x_test[j]) / 100)
            if (temp > 0.5) & (y_test[j] == 1.0):
                predict_true += 1
                result.append(float(1.0))
            elif (temp < 0.5) & (y_test[j] == 0.0):
                predict_true += 1
                result.append(float(1.0))
            else:
                result.append(float(0.0))
            query_result.append(temp)

        predict_true /= 16
        prediction += predict_true
        print(predict_true)
        print("result:", result)
        print("y_test", y_test)

    prediction /= 5
    print("cross validation result:", prediction)


if __name__ == '__main__':
    main()
