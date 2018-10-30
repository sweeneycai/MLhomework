import numpy as np
import matplotlib.pyplot as plt
from math import e

# x[2, 14],y[14]


def cost_func(x, y, theta):
    """
    计算代价函数
    """
    m = x.shape[1]
    J = (1 / (2 * m)) * np.sum((np.dot(theta, x) - y)**2)
    return J


def gradientDecent(x, y, theta, alpha, iters=100000000):
    """
    梯度下降
    """
    m = x.shape[1]
    iters = 0  # 循环次数
    count = 0  # 绘图计数
    grad = np.zeros(2)
    J = cost_func(x, y, theta)

    # 绘图
    plt.ion()
    plt.subplot(121)
    plt.xlim(2.0, 2.014)
    plt.scatter(x[1, :], y)

    while J > e-3:
        grad[0] = (1 / m) * np.sum(np.dot(theta, x) - y)
        grad[1] = (1 / m) * np.sum((np.dot(theta, x) - y) * x[1, :])
        theta = theta - alpha * grad

        # 设置间隔一定循环次数绘一次图，100000比较合适
        if count == 100000:
            J = cost_func(x, y, theta)
            plt.subplot(122)
            plt.scatter(iters, J)
            plt.subplot(121)
            plt.plot(x[1, :], np.dot(theta, x))
            plt.pause(1)
            count = 0
            #print(iters, J)
        count += 1
        iters += 1

    return theta


def main():
    x = np.ones([2, 14])
    x[1, :] = np.arange(2000, 2014, 1)
    # 做特征缩放
    x[1, :] = x[1, :] / 1000
    y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365,
                  5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900])
    theta = np.random.rand(2)
    alpha = 0.3
    theta = gradientDecent(x, y, theta, alpha)
    """
    plt.scatter(x[1, :], y)
    plt.plot(x[1, :], np.dot(theta, x))
    plt.show()
    """
    print(theta)
    # 预测2014年房价
    print((theta[1] * 2.014 + theta[0]) * 1000)


if __name__ == "__main__":
    main()
