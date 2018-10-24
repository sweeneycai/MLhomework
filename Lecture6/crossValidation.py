from ANN import NN
from shuffle import Data
import numpy as np
from sklearn.model_selection import KFold


def main():
    x, y = Data.getAllData()
    learning_rate = 0.3
    reg = 1e-8
    input_nodes = 2
    hidden_nodes = 150
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

        epoch = 5000
        n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)
        for i in range(epoch):
            for [j, x_temp] in enumerate(x_train):
                inputs = np.array(x_temp) / 100
                targets = y_train[j]
                n.train(inputs, targets)
        
        predict_true = 0
        result = []
        query_result = []
        for j in range(16):
            temp = n.query(np.array(x_test[j]) / 100)
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


if __name__ == "__main__":
    main()
