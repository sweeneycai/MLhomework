import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from shuffle import Data


# NetWork Framework
class NN:

    # init function
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, activation_type):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        
        # init weight between input layer and hidden layer, hidden layer and output layer, the weights obbey normal distribution.
        self.weight_ih = np.random.normal(0.0, pow(self.hnodes, 0.5), (self.hnodes, self.inodes))
        self.weight_ho = np.random.normal(0.0, pow(self.onodes, 0.5), (self.onodes, self.hnodes))
        
        if activation_type == 1:
            self.activation = lambda x: np.tanh(x)
            self.activation_deriv = lambda x: self.tanh_deriv(x)
        else:
            self.activation = lambda x: self.softplus(x)
            self.activation_deriv = lambda x: self.softplus_deriv(x)

    def tanh_deriv(self, x):  
        return 1.0 - np.tanh(x) * np.tanh(x)

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def softplus_deriv(self, x):
        return expit(x)

    # NetWork training
    def train(self, input_list, target_list):
        # init inputs and target
        inputs = np.array(input_list, ndmin=2).T
        target = np.array(target_list, ndmin=1)

        # calculate hidden layer inputs and hidden layer outputs
        hidden_inputs = np.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        # calculate final layer inputs and outputs
        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        # calculate final layer errors and hidden layer errors, use BackPropagation algorithm.
        output_errors = target - final_outputs
        hidden_errors = np.dot(self.weight_ho.T, output_errors)
        
        # update weight
        self.weight_ho += self.lr * (np.dot(output_errors * self.activation_deriv(final_outputs), hidden_outputs.T))
        self.weight_ih += self.lr * (np.dot(hidden_errors * self.activation_deriv(hidden_outputs), inputs.T))
    
    # Query NN
    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        return final_outputs

def main():
    x, y = Data.getAllData()
    learning_rate = 0.1
    input_nodes = 2
    hidden_nodes = 5
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

        epoch = 1000
        n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, 1)
        for i in range(epoch):
            for [j, x_temp] in enumerate(x_train):
                inputs = np.array(x_temp) / 100
                targets = y_train[j]
                n.train(inputs, targets)
        
        query_result = []
        for j in range(16):
            query_result.append(n.query(np.array(x_test[j]) / 100))

        predict_true = Data.calAc(query_result, y_test)
        prediction += predict_true
        print(predict_true)

    prediction /= 5
    print("cross validation result:", prediction)


if __name__ == "__main__":
    main()

