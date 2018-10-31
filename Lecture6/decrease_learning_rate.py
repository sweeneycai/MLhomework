from ANN_change_learning_rate import NN
from shuffle import Data
import matplotlib.pyplot as plt
import numpy as np

def main():
    training_x, training_y, test_x, test_y = Data.getData()
    learning_rate = 1
    reg = 0
    input_nodes = 2
    hidden_nodes = 5
    output_nodes = 1
    epoch = 2000  
    loss_train_all = []
    loss_test_all = []

    plt.ion()
    plt.subplot(121)
    plt.grid(True)
    plt.xlim(0, epoch)
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.title("The relationship between generations and loss")
    plt.subplot(122)
    plt.grid(True)
    plt.xlim(0, epoch)
    plt.xlabel("Generation")
    plt.ylabel("Learning rate")

    n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)

    for i in range(epoch):
        # training
        for[j, x] in enumerate(training_x):
            inputs = np.array(x) / 100
            targets = training_y[j]
            n.train(inputs, targets, learning_rate, i)
        print(n.lr)

        # calculate training set loss and accuracy
        loss_train = 0
        training_result = []
        for j, x in enumerate(training_x):
            result = n.query(np.array(x) / 100)
            loss_train += pow((result - training_y[j]), 2) / 2
            training_result.append(result)
                    
        # calculate test set loss and accuracy
        loss_test = 0
        testing_result = []
        for j, x in enumerate(test_x):
            result = n.query(np.array(x) / 100)
            loss_test += pow((result - test_y[j]), 2) / 2
            testing_result.append(result)

        loss_train_all.append(float(loss_train / 8))
        loss_test_all.append(float(loss_test / 2))

        if i % 10 == 0:
            plt.subplot(121)
            plt.scatter(i, (loss_train / 64), marker="^", c='b')
            plt.scatter(i, (loss_test / 16), marker='o', c='r')
            plt.subplot(122)
            plt.scatter(i, n.lr, marker='o', c='r')
            plt.pause(0.1)
            plt.savefig('change_learning_rate.png')


if __name__ == "__main__":
    main()