from ANN import NN
from shuffle import Data
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import KFold

def main():
    X, Y = Data.getAllData()
    learning_rate = 0.3
    reg = 1
    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 1
    times = []
    train_all_loss = []
    test_all_loss = []
    epoch = 100
    k = 1

    # plt.xlim(0, 20)
    plt.grid(True)
    plt.xlabel("λ range from 0.1 to 1e-7")
    plt.ylabel("Loss")
    plt.title("The relationship between λ and loss")

    while reg > 1e-7:
        train_loss = 0
        test_loss = 0
        kf = KFold(n_splits=5, shuffle=False)

        for [train_index, test_index] in kf.split(X):
            n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)

            # Generate training and testing data
            x_train, x_test, y_train, y_test = [], [], [], []

            for i in train_index:
                x_train.append(X[i])
                y_train.append(Y[i])

            for i in test_index:
                x_test.append(X[i])
                y_test.append(Y[i])

            # Training            
            for i in range(epoch):
                # training
                for[j, x] in enumerate(x_train):
                    inputs = np.array(x) / 100
                    targets = y_train[j]
                    n.train(inputs, targets)

            # compute the loss of training set
            for j, x in enumerate(x_train):
                result = n.query(np.array(x) / 100)
                train_loss += pow((result - y_train[j]), 2) / 2

            # compute the loss of testing set
            for j, x in enumerate(x_test):
                result = n.query(np.array(x) / 100)
                test_loss += pow(result - y_test[j], 2) / 2

        # compute the average loss in this cross validation
        train_loss = train_loss * 100 / (64 * 5)  # 进行放缩处理，使数据更易读
        test_loss = test_loss * 100 / (16 * 5)
        train_all_loss.append(float(train_loss))
        test_all_loss.append(float(test_loss))
        times.append(k)
        print("reg = ", reg, "   training set loss:", train_loss, "   test set loss:", test_loss)
        k += 1
        reg /= 1.2

    print(len(times), len(test_all_loss), len(train_all_loss))
    plt.plot(times, test_all_loss, 'r-', label='testing set loss')
    plt.plot(times, train_all_loss, 'b--', label='training set loss')
    plt.legend(loc='upper right')
    plt.savefig("changeReg.png")


if __name__ == '__main__':
    main()
