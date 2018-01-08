from numpy import *
import numpy as np
import numpy.random
import sklearn
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from sklearn.svm import LinearSVC, SVC
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

def plot_graph(X, Y, title, filename, scale, Y2=None):
    fig = plt.figure(figsize=(9,7))
    plt.title(title)
    plt.xscale(scale)
    label_Y = None

    if Y2 is not None:
        label_Y = 'Validation'
        plt.plot(X, Y2, color='yellow', label='Training')

    plt.plot(X, Y, color='blue', label=label_Y)

    if Y2 is not None:
        plt.legend(loc='upper right')

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()

def find_accuracies(Cs, accuracy=[], train_accuracy=[]):
    for C in Cs:
        model = LinearSVC(loss='hinge', fit_intercept=False, C=C)
        model.fit(train_data, train_labels)
        accuracy.append(model.score(validation_data, validation_labels))
        train_accuracy.append(model.score(train_data, train_labels))

    return np.argmax(accuracy)

def save_img_to_file(arr, filename):
    fig = plt.figure()
    plt.imshow(reshape(arr, (28, 28)), interpolation='nearest')
    fig.savefig(filename)
    plt.close()

def q_a():
    Cs = [10**i for i in xrange(-10, 11)]
    accuracy = []
    train_acc = []
    c = find_accuracies(Cs, accuracy, train_acc)
    for i in xrange(len(accuracy)):
        accuracy[i] *= 100
        train_acc[i] *= 100

    plot_graph(Cs, accuracy, 'Accuracy vs C in SVM', 'out_3a.png', 'log', train_acc)
    print('Best C: ' + str(Cs[c]))

def q_c():
    Cs = [10**i for i in xrange(-10, 11)]
    best_C = Cs[find_accuracies(Cs)]
    model = LinearSVC(loss='hinge', fit_intercept=False, C=best_C)
    model.fit(train_data, train_labels)
    save_img_to_file(model.coef_, 'out_3c.png')

def q_d():
    Cs = [10**i for i in xrange(-10, 11)]
    best_C = Cs[find_accuracies(Cs)]
    model = LinearSVC(loss='hinge', fit_intercept=False, C=best_C)
    model.fit(train_data, train_labels)
    accuracy = model.score(test_data, test_labels)
    print('Accuracy of test set for C = ' +  str(best_C) + ' is: ' + str(100*accuracy) + '%')

def q_e():
    model = SVC(kernel='rbf', C=10, gamma=5*(10**-7))
    model.fit(train_data, train_labels)
    train_accuracy = model.score(train_data, train_labels)
    test_accuracy = model.score(test_data, test_labels)
    print('Accuracy of train set is: ' + str(100*train_accuracy) + '%')
    print('Accuracy of test set is: ' + str(100*test_accuracy) + '%')

def main(subsection):
    if subsection == 'a':
        q_a()

    elif subsection == 'e':
        q_e()

    elif subsection == 'c':
        q_c()

    elif subsection == 'd':
        q_d()

    else:
        print('[ERROR]\nSVM:')
        print('No subsection \'' + subsection + '\' in this module')
        return 1

    return 0