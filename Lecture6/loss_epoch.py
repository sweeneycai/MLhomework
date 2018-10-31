from ANN import NN
from shuffle import Data
import matplotlib.pyplot as plt
import numpy as np

def main():
    training_x, training_y, test_x, test_y = Data.getData()
    learning_rate = 0.1
    reg = 0
    input_nodes = 2
    hidden_nodes = 5
    output_nodes = 1
    epoch = 5000  
    loss_train_all = []
    loss_test_all = []
    accu_train = []
    accu_test = []

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
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("The relationship between generations and accuracy")

    n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)

    for i in range(epoch):
        # training
        for[j, x] in enumerate(training_x):
            inputs = np.array(x) / 100
            targets = training_y[j]
            n.train(inputs, targets)

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

        ac_train = Data.calAc(training_result, training_y)
        ac_test = Data.calAc(testing_result, test_y)

        print("test ac:", ac_test, "  epoch:", i)
        loss_train_all.append(float(loss_train / 64))
        loss_test_all.append(float(loss_test / 16))
        accu_test.append(ac_test)
        accu_train.append(ac_train)

        if i % 50 == 0:
            plt.subplot(121)
            plt.scatter(i, loss_train / 64, c='b', marker='^')
            plt.scatter(i, loss_test / 16, c='r', marker='o')
            plt.subplot(122)
            plt.scatter(i, ac_train, c='b', marker='^')
            plt.scatter(i, ac_test, c='r', marker='o')
            plt.savefig('loss.png')
            plt.pause(0.1)
        


if __name__ == "__main__":
    main()