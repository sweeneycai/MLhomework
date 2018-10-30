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

    plt.grid(True)
    # plt.ylim(0, 10)
    plt.xlim(0, epoch)
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.title("The relationship between generations and loss")

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
        loss_train_all.append(float(loss_train / 8))
        loss_test_all.append(float(loss_test / 2))
        accu_test.append(ac_test)
        accu_train.append(ac_train)

    plt.plot(range(0, epoch, 1), loss_train_all, 'r-', label='Training set loss')
    plt.plot(range(0, epoch, 1), loss_test_all, 'b-', label='Test set loss')
    plt.plot(range(0, epoch, 1), accu_train, 'r--', label='Accuracy of training set')
    plt.plot(range(0, epoch, 1), accu_test, 'b--', label='Accuracy of test set')
    plt.legend(loc='upper right')
    plt.savefig('loss.png')


if __name__ == "__main__":
    main()