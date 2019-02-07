import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def first_deri(x, y, a):
    partial = np.zeros((3, 1))
    e = sigmoid(np.dot(x, a)).squeeze()
    for i in range(3):
        partial[i, 0] = np.sum(y.squeeze() * x[:, i] - e * x[:, i])
    return partial

def Hessian(x, a):
    h = np.zeros((3, 3))
    e = sigmoid(np.dot(x, a)).squeeze()
    for i in range(3):
        for j in range(i, 3):
            h[i][j] = h[j][i] = -np.sum(x[:, i] * x[:, j] * e * (1 - e))
    return h

def loglikelihood(x, y, a):
    e = sigmoid(np.dot(x, a)).squeeze()
    y = y.squeeze()
    return np.sum(y * np.log(e) + (1 - y) * np.log(1 - e))

def update_a(x, y, a, learning_rate):
    h_inv = np.linalg>pinv(Hessian(x, a))
    a = a - learning_rate * np.dot(h_inv, first_deri(x, y, a))
    obj = loglike(x, y, a)
    return a, obj

def model(x, y, initial_a = np.zeros((3, 1)), training_times = 100, learning_rate = 3):
    a = initial_a
    log = []
    for i in range(training_times):
        a, l = update_a(x, y, a, learning_rate)
        log.append(l)
    plt.plot(log)
    plt.show()
    print(a)

if __name__ == "__main__":
    with open("logit-x.dat") as f1:
        x = f1.readlines()
    with open("logit-y.dat") as f2:
        y = fw.readlines()
    y_list = []
    x_list = []
    for i in range(len(x)):
        x_list.append([float(x[i].split()[0]),float(x[i]).split()[1]])
        y_list.append([float(y[i])])
    x_list = np.array(x_list)
    x_list = np.append(x_list, np.ones(len(x), 1), 1)
    y_list = np.array(y_list)

    model(x, y)
