from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
from math import sqrt

from conv_net import deepnn

message = 'Choose initialization method (insert number):\n' + \
    '\t0 zero initialization\n' + \
    '\t1 normal distribution N(0, 0.1)\n' + \
    '\t2 uniform distibution U[-1/sqrt(784),1/sqrt(784)] = U[-1/28,1/28]\n' + \
    '\t3 Xavier initialization (normal distribution)\n' + \
    '\t4 Xavier initialization (uniform distribution)\n' + \
    '\t5 RELU initialization (normal distribution)\n' + \
    '\t6 Xavier initialization (uniform distribution)\n'
init = int(input(message))

learning_rate = 0.5
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

n_in = 784
n_out = 10
s = n_in + n_out
x = tf.placeholder(tf.float32, [None, n_in])
if init == 0:
    W = tf.Variable(tf.zeros([n_in, n_out]))
    b = tf.Variable(tf.zeros([n_out]))

elif init == 1:
    W = tf.Variable(tf.random_normal([n_in, n_out], stddev=0.1))
    b = tf.Variable(tf.random_normal([n_out], stddev=0.1))

elif init == 2:
    W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-1.0/28, maxval=1.0/28))
    b = tf.Variable(tf.random_uniform([n_out], minval=-1.0/28, maxval=1.0/28))

elif init == 3:
    W = tf.Variable(tf.random_normal([n_in, n_out], stddev=sqrt(2.0/s)))
    b = tf.Variable(tf.random_normal([n_out], stddev=sqrt(2.0/s)))

elif init == 4:
    W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-sqrt(6.0/s), maxval=sqrt(6.0/s)))
    b = tf.Variable(tf.random_uniform([n_out], minval=-sqrt(6.0/s), maxval=sqrt(6.0/s)))

elif init == 5:
    W = tf.Variable(tf.random_normal([n_in, n_out], stddev=sqrt(2.0/n_in)))
    b = tf.Variable(tf.random_normal([n_out], stddev=sqrt(2.0/n_in)))

elif init == 6:
    W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-sqrt(6.0/n_in), maxval=sqrt(6.0/n_in)))
    b = tf.Variable(tf.random_uniform([n_out], minval=-sqrt(6.0/n_in), maxval=sqrt(6.0/n_in)))

else:
    print('[Error] please enter a number between 0 and 6')
    exit(1)

# Output tensor.
y_pred = tf.matmul(x, W) + b

# Define loss and optimizer
y = tf.placeholder(tf.float32, [None, 10])

softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
cross_entropy = tf.reduce_mean(softmax)

# Define a gradient step operation.
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define a TensorFlow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(200)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if _ % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
        validation_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
        print('step %d, training accuracy %g, validation accuracy %g' % (_, train_accuracy, validation_accuracy))

# Test trained model
print('Test Accuracy')
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))