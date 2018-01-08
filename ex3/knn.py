from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

### Aux function ###
def get_argmax(k, dist, labels):
    '''
    Returns the digit that's closest to the test image
    '''
    digits = [0 for i in range(10)]
    for j in range(k):
        orig_idx = dist[j][1]
        digit = int(labels[orig_idx])
        digits[digit] += 1

    return np.argmax(digits)

### Aux function ###
def build_dataset(train_set, test_set, to_sort=True):
    '''
    Builds a data set of distances between training images and test images
    '''
    distances_from_test_images = []
    for query_img in test_set:
        dist = []
        i = 0
        for img in train_set:
            d = np.linalg.norm(query_img - img)
            dist.append((d, i))
            i += 1
        if to_sort:
            distances_from_test_images.append(sorted(dist, key = lambda p : p[0]))
        else:
            distances_from_test_images.append(dist)

    return distances_from_test_images

def q_a(image_set=train[:1000], labels=train_labels[:1000], query=test[0], k=10, distances=None):
    query_img = np.array(query)
    dist = []
    i = 0
    
    if distances is None:
        for img in image_set:
            dist.append((np.linalg.norm(query_img - img), i))
            i += 1

        dist.sort(key=lambda p : p[0])
    else:
        dist = distances

    return get_argmax(k, dist, labels)

def get_first_n_samples(n=1000):
    return (train[:n], train_labels[:n])

def q_b(n=1000, k=10, tup=get_first_n_samples(), distances_dataset=None):
    error = 0
    num_of_samples = n
    train_set, labels = tup[0], tup[1]
    distances = None
    m = len(test)
    for i in xrange(m):
        if distances_dataset is not None:
            distances = distances_dataset[i]
        prediction = q_a(train_set, labels, test[i], k, distances)
        error += (prediction != int(test_labels[i]))

    return 100*(1 - float(error)/m)

### Aux function ###
def plot_graph(X, Y, title, filename):
    fig = plt.figure(figsize=(9,7))
    plt.title(title)
    plt.plot(X, Y, color='blue')

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()

def q_c(n=1000, max_k=100):
    ks = [k for k in xrange(1, max_k + 1)]
    accuracy = []
    train_and_labels = get_first_n_samples(n)

    dist_dataset = build_dataset(train_and_labels[0], test)

    for k in ks:
        accuracy.append(q_b(k=k, tup=train_and_labels, distances_dataset=dist_dataset))

    plot_graph(ks, accuracy, 'Accuracy of predictions vs k', 'out_1c.png')
    return np.argmax(accuracy) + 1

def q_d(max_n=5000, min_n=100, stride_n=100, best_k=1):
    samples = [n for n in range(min_n, max_n + 1, stride_n)]
    accuracy = []
    train_and_labels = get_first_n_samples(max_n)
    dist_dataset = build_dataset(train_and_labels[0], test, False)
    m = len(test)
    for n in samples:
        train_set = train_and_labels[0][:n]
        train_labels = train_and_labels[1][:n]
        dist_dataset_n = [dist_dataset[i][:n].sort(key=lambda p : p[0]) for i in xrange(m)]
        accuracy.append(q_b(n, k=best_k, tup=(train_set, train_labels), distances_dataset=dist_dataset_n))

    plot_graph(samples, accuracy, 'Accuracy of predictions vs number of samples', 'out_1d.png')


def main(subsection, args):
    if subsection == 'a':
        if len(args) > 0:
            q_a(k=int(args[0]))
        else:
            print('[ERROR]\nkNN:')
            print('For subsection \'a\' please enter number of neighbours')
            return 1

    elif subsection == 'b':
        q_b()

    elif subsection == 'c':
        q_c()

    elif subsection == 'd':
        q_d()

    else:
        print('[ERROR]\nkNN:')
        print('No subsection \'' + subsection + '\' in this module')
        return 1

    return 0