from numpy import *
import numpy as np
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


def perceptron(train_set, train_labels):
	weights = np.zeros(len(train_set[0]))
	
	for img, true_label in zip(train_set, train_labels):
		img = img / np.linalg.norm(img)
		dot_prod = np.dot(weights, img)
		prediction_label = 1 if dot_prod >= 0 else -1

		if int(true_label) != prediction_label:
			weights += true_label*img

	return weights

def get_first_n_samples(n=1000):
    return (train_data[:n], train_labels[:n])

def check_test_error(w, mis=False):
    error = 0.0
    i = 0
    mis_imgs = [0, 0]
    j = 0
    for query_img in test_data:
        dot_prod = np.dot(query_img, w)
        pred_label = 1 if dot_prod >= 0 else -1
        true_label = test_labels[i]
        i += 1
        if int(true_label) != pred_label:
            error += 1
            if mis:
                mis_imgs[j] = i - 1
                j += 1
                if j == 2:
                    return mis_imgs


    return 100*(1 - error / len(test_data))

def print_table(stats):
    print('%-10s  %-10s  %-10s  %-10s' % ('#samples', 'mean (%)', '5-th %', '95-th %'))
    for stat in stats:
        print('%-10d  %-10f  %-10f  %-10f' % stat)

def q_a(runs=100):
    samples = [5, 10, 50, 100, 500, 1000, 5000]
    statistics = []
    for n in samples:
        accuracy = []
        train_set, train_labels = get_first_n_samples(n)
        train_and_labels = zip(train_set, train_labels)
        for i in xrange(runs):
            np.random.shuffle(train_and_labels)
            train_set, train_labels = zip(*train_and_labels)
            w = perceptron(np.array(train_set), np.array(train_labels))
            accuracy.append(check_test_error(w))

        statistics.append((n, np.mean(accuracy), np.percentile(accuracy, 5), np.percentile(accuracy, 95)))

    print_table(statistics)

def save_img_to_file(arr, filename):
    fig = plt.figure()
    plt.imshow(reshape(arr, (28, 28)), interpolation='nearest')

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()


def q_b():
    w = perceptron(train_data, train_labels)
    save_img_to_file(w, 'out_2b.png')

def q_c():
    w = perceptron(train_data, train_labels)
    print('accuracy = ' + str(check_test_error(w)) + '%')

def q_d():
    w = perceptron(train_data, train_labels)
    mis_imgs = check_test_error(w, True)
    save_img_to_file(test_data_unscaled[mis_imgs[0]], 'out_2d_1.png')
    save_img_to_file(test_data_unscaled[mis_imgs[1]], 'out_2d_2.png')

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
        print('[ERROR]\nperceptron:')
        print('No subsection \'' + subsection + '\' in this module')
        return 1

    return 0