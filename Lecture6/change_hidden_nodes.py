from ANN import NN
from shuffle import Data
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import KFold


def main():
    x, y = Data.getAllData()
    learning_rate = 0.1
    reg = 1e-8
    input_nodes = 2
    hidden_nodes = 5
    output_nodes = 1
    plt.ylim(0, 1)
    plt.grid(True)
    hidden = []
    test_rate = []

    while hidden_nodes < 500:
        hidden.append(hidden_nodes)
        kf = KFold(n_splits=5, shuffle=False)
        predict_count = 0
        for [train_index, test_index] in kf.split(x):
            x_train, x_test, y_train, y_test = [], [], [], []

            for i in train_index:
                x_train.append(x[i])
                y_train.append(y[i])

            for i in test_index:
                x_test.append(x[i])
                y_test.append(y[i])

            epoch = 5000
            n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)
            for i in range(epoch):
                # training
                for[j, x_temp] in enumerate(x_train):
                    inputs = np.array(x_temp) / 100
                    targets = y_train[j]
                    n.train(inputs, targets)

            predict_test_data = []
            predict_test_true = 0

            for j in x_test:
                if n.query(np.array(j) / 100) > 0.5:
                    predict_test_data.append(float(1.0))
                else:
                    predict_test_data.append(float(0.0))
          
            for j in range(16):
                if predict_test_data[j] == y_test[j]:
                    predict_test_true += 1

            predict_test_true /= 16
            predict_count += predict_test_true

        test_rate.append(predict_count / 5)
        hidden_nodes += 5
        print(hidden_nodes, predict_count / 5)

    file = open('nodes.dat', 'a')
    file.writelines(str(test_rate))
    plt.plot(hidden, test_rate, 'b.')
    plt.savefig('change_hidden_nodes.png')


if __name__ == "__main__":
    main()