from ANN import NN
from shuffle import Data
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import KFold


def main():
    x, y = Data.getAllData()
    learning_rate = 0.1
    reg = 0
    input_nodes = 2
    hidden_nodes = 0
    output_nodes = 1
    test_rate = []
    epoch = 3000

    plt.ylim(0, 1)
    plt.xlabel("Hidden layer nodes")
    plt.ylabel("Accuracy")
    plt.title("Relationship between hidden layer nodes and accuracy")
    plt.grid(True) 

    while hidden_nodes < 100:
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

            n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)

            for i in range(epoch):
                # training
                for[j, x_temp] in enumerate(x_train):
                    inputs = np.array(x_temp) / 100
                    targets = y_train[j]
                    n.train(inputs, targets)

            predict_test_data = []
            # calculate query result 
            for x_temp in x_test:
                predict_test_data.append(n.query(np.array(x_temp) / 100))                
            
            # calculate test data set accuracy
            predict_test_true = Data.calAc(predict_test_data, y_test)
            predict_count += predict_test_true

        test_rate.append(predict_count / 5)
        hidden_nodes += 1
        print(hidden_nodes, predict_count / 5)

    plt.plot(range(0, 100, 1), test_rate, 'b-')
    plt.savefig('change_hidden_nodes.png')


if __name__ == "__main__":
    main()