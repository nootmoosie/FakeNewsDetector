import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = 3
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
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
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                print("Reshaping input into chunks:")
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                print(type(epoch_y))
                print(type(epoch_x))
                print(np.array(epoch_x).shape)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                # grabs next batch of images, 128 of the 1x784 vectors -> 128x784
                # then reshapes into chunks of 128x28, grabbing 28 of them -> 128x28x28

                # for arictle: grabs 128 articles of the #wordsx300 embeddings (might have to flatten)
                # reshapes into chunks of 128xn_chunksxchunk_size
                # n_chunks would be amount of lines in article, chunk_size would be # words in line

                print(np.array(epoch_x).shape)
                print("DONE.")

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',
              accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


train_neural_network(x)

tf.logging.set_verbosity(old_v)