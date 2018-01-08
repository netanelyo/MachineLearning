from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt


mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_size = 2000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = (labels[train_idx[:train_data_size]] == pos)*2-1

#validation_data_unscaled = data[train_idx[6000:], :].astype(float)
#validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_size = 2000
test_data_unscaled = data[60000+test_idx[:test_data_size], :].astype(float)
test_labels = (labels[60000+test_idx[:test_data_size]] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
#validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


def weak_learner(dist, samples, labels):
    m = len(dist)
    d = len(samples[0])
    Fs = inf
    b = 0
    best_theta = best_j = 0
    samples_and_dist = zip(samples, labels, dist)
    for j in range(d):
        sorted_samples_and_dist = sorted(samples_and_dist, key=lambda x: x[0][j])
        sorted_samples_and_dist.append((sorted_samples_and_dist[-1][0].copy(), 0, 0))
        sorted_samples_and_dist[-1][0][j] += 1
        F1 = numpy.sum(dist[where(labels == 1)])
        F2 = numpy.sum(dist[where(labels == -1)])

        if F1 < Fs:
            Fs = F1
            best_theta = sorted_samples_and_dist[0][0][j] - 1
            best_j = j
            b = 1

        if F2 < Fs:
            Fs = F2
            best_theta = sorted_samples_and_dist[0][0][j] - 1
            best_j = j
            b = -1

        for i in range(m):
            F1 = F1 - sorted_samples_and_dist[i][1] * sorted_samples_and_dist[i][2]
            F2 = F2 + sorted_samples_and_dist[i][1] * sorted_samples_and_dist[i][2]
            x_i_j = sorted_samples_and_dist[i][0][j]
            x_i1_j = sorted_samples_and_dist[i+1][0][j]
            if F1 < Fs and x_i_j != x_i1_j:
                Fs = F1
                best_theta = 0.5*(x_i_j + x_i1_j)
                best_j = j
                b = 1

            if F2 < Fs and x_i_j != x_i1_j:
                Fs = F2
                best_theta = 0.5 * (x_i_j + x_i1_j)
                best_j = j
                b = -1

    return (best_theta, best_j, b)


def check_hypothesis(samples, labels, dist, theta, j, b):
    n = len(samples)
    hypot = array([-1 for i in range(n)])
    for i in range(n):
        if (b == 1 and samples[i][j] <= theta) or (b == -1 and samples[i][j] > theta):
            hypot[i] = 1

    epsilon = numpy.sum(dist[numpy.where(numpy.multiply(hypot, labels) == -1)])

    return epsilon, hypot


def predict(sample, g):
    prediction = []
    for theta, j, first_hypot in g:
        if first_hypot:
            v = 1 if sample[j] <= theta else -1
        else:
            v = 1 if sample[j] > theta else -1

        prediction.append(v)

    return array(prediction)


def check_error(data, labels, classifier):
    error = 0.0
    for sample, label in zip(data, labels):
        if classifier(sample) != label:
            error += 1

    return error/len(data)


def adaboost(samples, labels, T):
    n = len(samples)
    dist = array([1.0/n for i in range(n)])
    g = []
    weights = []
    train_error = []
    test_error = []
    for t in range(T):
        # print("iteration: {}".format(t+1))
        theta, j, b = weak_learner(dist, samples, labels)
        epsilon, hypot = check_hypothesis(samples, labels, dist, theta, j, b)
        first_hypot = (b == 1)
        weights.append(0.5*log((1-epsilon)/epsilon))
        wt = weights[-1]
        mult = multiply(labels, hypot)
        Zt = 2*sqrt(epsilon * (1-epsilon))
        dist = multiply(dist, exp(-wt*mult))/Zt
        g.append((theta, j, first_hypot))
        classifier = lambda x: sign(sum(multiply(predict(x, g), weights)))
        train_error.append(check_error(samples, labels, classifier))
        test_error.append(check_error(test_data, test_labels, classifier))

    return classifier, train_error, test_error


def plot_multiple_graphs(X, Ys, title, filename):
    colors = ['yellow', 'blue']
    legends = ['train', 'test']
    fig = plt.figure(figsize=(9, 7))
    plt.title(title)
    for Y, col, leg in zip(Ys, colors, legends):
        plt.plot(X, Y, color=col, label=leg)
    plt.legend()

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()


def q_b():
    T = 100
    classifier, train_error, test_error = adaboost(train_data, train_labels, T)
    Ys = [train_error, test_error]
    plot_multiple_graphs(range(T), Ys, 'Errors vs iteration of AdaBoost', 'out_1b.png')
    return 0


if __name__=='__main__':
    q_b()