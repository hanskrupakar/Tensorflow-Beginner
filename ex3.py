import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

ip = 28
rows = 28
hidd = 140
op = 10 

x = tf.placeholder("float", [None, rows, ip])
y = tf.placeholder("float", [None, op])

W_h = tf.Variable(tf.random_normal([ip, hidd]))
W_o = tf.Variable(tf.random_normal([hidd, op]))
B_h = tf.Variable(tf.random_normal([hidd]))
B_o = tf.Variable(tf.random_normal([op]))

def RNN(x, weights, biases):

    # This is the formatting required for Tensorflow RNN Input
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, ip])
    x = tf.split(0, rows, x)

    lstm_cell = rnn_cell.BasicLSTMCell(hidd, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], W_o) + B_o

pred = RNN((tf.matmul(x, W_h) + B_h), weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 1

    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, rows, ip))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.4f}".format(acc)
        step += 1
    print "Optimization Finished!"

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, rows, ip))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label})