from ANN import NN
from shuffle import Data
import numpy as np
from sklearn.model_selection import KFold


def main():
    x, y = Data.getAllData()
    learning_rate = 0.1
    reg = 0
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
        n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate, reg)
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
