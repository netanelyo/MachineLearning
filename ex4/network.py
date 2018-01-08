"""
network.py
"""

import os
import random
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data, is_subsection_b=False, is_subsection_d=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  """
        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))
        n = len(training_data)

        X = range(epochs)
        if is_subsection_b:
            train_accuracies, train_losses, test_accuracies = [], [], []
        if is_subsection_d:
            all_norms = []

        for j in range(epochs):
            random.shuffle(list(training_data))
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                deriv_norms = self.update_mini_batch(mini_batch, learning_rate, is_subsection_d)
            test_acc = self.one_label_accuracy(test_data)
            if is_subsection_b:
                # Calculate accuracies and losses #
                train_acc = self.one_hot_accuracy(training_data)
                train_loss = self.loss(training_data)
                test_accuracies.append(test_acc)
                train_accuracies.append(train_acc)
                train_losses.append(train_loss)
                ###################################
            print ("Epoch {0} test accuracy: {1}".format(j, test_acc))

            if is_subsection_d:
                all_norms.append([norm/np.float32(n) for norm in deriv_norms])

        if is_subsection_b:
            # Plot graphs of accuracies and losses #
            plot_graph(X, train_accuracies, 'Train Accuracies', 'out_2b_1.png')
            plot_graph(X, train_losses,     'Train Losses',     'out_2b_2.png')
            plot_graph(X, test_accuracies,  'Test Accuracies',  'out_2b_3.png')

        if is_subsection_d:
            plot_multiple_graphs(X, np.transpose(all_norms), 'Biases Norms', 'out_2d.png')

    def update_mini_batch(self, mini_batch, learning_rate, is_subsection_d=False):
        """Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        deriv_norms = [0 for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            if is_subsection_d:
                deriv_norms = [dnorm + np.linalg.norm(db) for dnorm, db
                               in zip(deriv_norms, delta_nabla_b)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

        return deriv_norms

    def backprop(self, x, y):
        softmax = [x]       # Will contain a_l for each layer 0<=l<=L-1
        before_softmax = [] # Will contain z_l for each layer 1<=l<=L-1
        out_before_smax = self.network_output_before_softmax(x, softmax, before_softmax)
        delta = [self.loss_derivative_wr_output_activations(out_before_smax, y)]
        # Create delta_l as defined in recitation
        for W, z in reversed(zip(self.weights[1:], before_softmax)):
            sig_z_deriv = sigmoid_derivative(z)
            d = delta[-1]
            W_T = W.transpose()
            delta.append(np.multiply(np.dot(W_T, d), sig_z_deriv))

        db = delta[::-1] # Reversed deltas --> d/db(loss)
        # Create d/dw(loss) as defined in recitation
        dw = []
        for a, d in zip(softmax, db):
            dw.append(np.dot(d, a.transpose()))

        return db, dw

    def one_label_accuracy(self, data):
        """Return accuracy of network on data with numeric labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)
         for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results)/float(len(data))

    def one_hot_accuracy(self,data):
        """Return accuracy of network on data with one-hot labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))
                          for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results) / float(len(data))

    ### Added arguments for backprop solution ###
    def network_output_before_softmax(self, x, after_softmax=None, before_softmax=None):
        """Return the output of the network before softmax if ``x`` is input."""
        layer = 0
        for b, w in zip(self.biases, self.weights):
            if layer == len(self.weights) - 1:
                x = np.dot(w, x) + b
            else:
                """Changed functionality to keep a_l and z_l for each layer l"""
                z_l = np.dot(w, x) + b
                x = sigmoid(z_l) # a_l = x
                if after_softmax is not None and before_softmax is not None:
                    after_softmax.append(x)
                    before_softmax.append(z_l)
            layer += 1
        return x

    def loss(self, data):
        """Return the loss of the network on the data"""
        loss_list = []
        for (x, y) in data:
            net_output_before_softmax = self.network_output_before_softmax(x)
            net_output_after_softmax = self.output_softmax(net_output_before_softmax)
            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])
        return sum(loss_list) / float(len(data))

    def output_softmax(self, output_activations):
        """Return output after softmax given output before softmax"""
        output_exp = np.exp(output_activations)
        return output_exp/output_exp.sum()

    def loss_derivative_wr_output_activations(self, output_activations, y):
        """Return derivative of loss with respect to the output activations before softmax"""
        return self.output_softmax(output_activations) - y


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


### Auxiliary functions
def plot_graph(X, Y, title, filename):
    fig = plt.figure(figsize=(9,7))
    plt.title(title)
    plt.plot(X, Y, color='blue')

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()


def plot_multiple_graphs(X, Ys, title, filename):
    colors = ['red', 'yellow', 'blue', 'green', 'black']
    fig = plt.figure(figsize=(9, 7))
    plt.title(title)
    i = 1
    for Y, col in zip(Ys, colors):
        plt.plot(X, Y, color=col, label='layer {}'.format(i))
        i += 1
    plt.legend()

    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
    fig.savefig(path + filename)
    plt.close()