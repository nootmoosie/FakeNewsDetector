import tensorflow as tf
import gensim
import numpy as np

from tensorflow.contrib import rnn
from process_data import process_train_data

hm_epochs = 50
n_classes = 2
# batch_size = 128

chunk_size = 300
n_chunks = 1
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])  # changes dimensionality to comply w/ rnn_cell API
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']  # final output multiplied with weights + biases

    return output


def train_neural_network(x):

    # train_data, test_data = process_csv_data('../data/train.csv', '../data/test.csv', 10, 5)
    # epoch_x, epoch_y = train_data[0], train_data[1]

    epoch_x, epoch_y = process_train_data('../data/train.csv', 50, n_chunks)
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # (epoch_x, epoch_y), _ = process_csv_data('../data/train.csv', '../data/test.csv', 10, 5)

            # for _ in range(int(mnist.train.num_examples / batch_size)):
            #     # print("Reshaping input into chunks:")
            #     epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            #     # print(tf.shape(epoch_x))
            #     epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
            #     # print(tf.shape(epoch_x))
            #     # print("DONE.")

            epoch_x = epoch_x.reshape((epoch_x.shape[0], n_chunks, chunk_size))
            pred, _, c = sess.run([prediction, optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            print(pred)
            epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:',
        #       accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


train_neural_network(x)
