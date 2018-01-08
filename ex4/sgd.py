from numpy import random as nprand
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import os


mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = np.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = np.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos)*2-1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_unscaled = data[60000+test_idx, :].astype(float)
test_labels = (labels[60000+test_idx] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


def sgd_svm(samples, labels, C, eta_0, T):
    """SGD-SVM implementation"""
    n = len(samples[0])
    m = len(samples)
    w = np.zeros(n)
    for t in range(1,T+1):
        eta_t = eta_0 / float(t)    #eta_t = eta_0/t
        i = nprand.randint(m)       #Stochastic GD - random sample
        y_i = labels[i]
        x_i = samples[i]
        prod = y_i * np.dot(x_i, w)
        np.seterr(over='ignore')
        u = (1 - eta_t) * w
        if prod < 1:
            w = u + eta_t * C * y_i * x_i
        else:
            w = u

    return w


def calc_accuracy(w, samples, samples_labels):
    acc = 0.0
    for x_i, y_i in zip(samples, samples_labels):
        if y_i * np.dot(w, x_i) >= 1:
            acc += 1

    return acc / len(samples)


def assess_perf(train, labels, valid, valid_labels, eta_0, C, T, runs=10):
    avg_acc = 0.0
    for i in range(runs):
        w = sgd_svm(train, labels, C, eta_0, T)
        avg_acc += calc_accuracy(w, valid, valid_labels)

    return 100*avg_acc / runs


def plot_graph(X, Y, title, filename, scale):
    fig = plt.figure(figsize=(9,7))
    plt.title(title)
    plt.xscale(scale)
    plt.plot(X, Y, color='blue')

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()


def q_a():
    eta = [10**i for i in range(-5, 6)]
    accuracy = []
    for eta_0 in eta:
        acc = assess_perf(train_data, train_labels, validation_data, validation_labels, eta_0, 1, 1000)
        accuracy.append(acc)

    plot_graph(eta, accuracy, 'Accuracy of SGD SVM vs eta_0', 'out_1a.png', 'log')
    print('Best eta_0 = ' + str(eta[np.argmax(accuracy)]))


def q_b(best_eta_0=1):
    Cs = [(5**i)*(10**-2) for i in range(0, 9)]
    accuracy = []
    for C in Cs:
        acc = assess_perf(train_data, train_labels, validation_data, validation_labels, best_eta_0, C, 1000)
        accuracy.append(acc)

    plot_graph(Cs, accuracy, 'Accuracy of SGD SVM vs C', 'out_1b.png', 'log')
    print('Best C = ' + str(Cs[np.argmax(accuracy)]))


def plot_weights_vector_as_image(w, filename):
    fig = plt.figure()
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()


def q_c(best_eta_0=1, best_C=0.25, to_plot=True):
    w = sgd_svm(train_data, train_labels, best_C, best_eta_0, 20000)
    if to_plot:
        plot_weights_vector_as_image(w, 'out_1c.png')
    return w


def q_d():
    best_w = q_c(to_plot=False)
    acc = calc_accuracy(best_w, test_data, test_labels)
    print('Accuracy for 20k runs on test set with best classifier: ' + str(100*acc) + '%')


def main(subsection):
    if subsection == 'a':
        q_a()

    elif subsection == 'b':
        q_b()

    elif subsection == 'c':
        q_c()

    elif subsection == 'd':
        q_d()

    else:
        print('[ERROR]\nsgd:')
        print('No subsection \'' + subsection + '\' in this module')
        return 1

    return 0