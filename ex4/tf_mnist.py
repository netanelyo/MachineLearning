from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from conv_net import deepnn

learning_rate = 0.5
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.random_normal(([784, 10]), stddev=0.1))
b = tf.Variable(tf.random_normal(([10]), stddev=0.1))

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