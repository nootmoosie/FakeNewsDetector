import tensorflow as tf
import gensim
import numpy as np
import re

from tensorflow.contrib import rnn
from process_data import process_train_data, train_test_split, get_original_test_data
from make_plot import plot_acc

num_articles = 20800
hm_epochs = 50
n_classes = 2
# batch_size = 128

chunk_size = 300
n_chunks = 50
rnn_size = 256
l_rate = .0001

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    ''' Caclulates output of RNN cell '''

    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])  # changes dimensionality to comply w/ rnn_cell API
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    gru_cell = rnn.GRUCell(rnn_size)
    rnn_cell = rnn.RNNCell()
    outputs, states = rnn.static_rnn(gru_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']  # final output multiplied with weights + biases

    return output


def train_neural_network(x):
    ''' Loads data and trains the RNN '''

    #epoch_x, epoch_y = process_train_data('../data/train.csv', num_articles, n_chunks)
    (train_x, train_y), (test_x, test_y) = process_train_data('../data/train.csv', num_articles, n_chunks, .8)
    #train, test = train_test_split(epoch_x, epoch_y, .8)
    epoch_x, epoch_y = train_x, train_y
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        train_accuracies = []
        test_accuracies = []

        for epoch in range(hm_epochs):

            epoch_loss = 0
            epoch_x = epoch_x.reshape((epoch_x.shape[0], n_chunks, chunk_size))
            pred, _, c = sess.run([prediction, optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            ##### FOR PLOTTING ACCURACIES AT EACH EPOCH #####
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            train_accuracies.append(accuracy.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: epoch_y}))
            #test_x, test_y = test[0], test[1]
            test_accuracies.append(accuracy.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}))
            #################################################

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        print('correct!!!!; ', correct.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: epoch_y}))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        train_acc = accuracy.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: epoch_y})
        print('Training Accuracy:', train_acc)
        # print('Training Correct Article Indices: ',
        #       correct.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: epoch_y}))

        #test_x, test_y = test[0], test[1]
        eval_acc = accuracy.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y})
        print('Test Accuracy:', eval_acc)

        ##### FOR PLOTTING ACCURACIES AT EACH EPOCH #####
        plot_acc(train_accuracies, test_accuracies, hm_epochs, rnn_size, n_chunks, train_acc, eval_acc)
        #################################################

        # print('Test Correct Article Indices: ',
        #       correct.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}))

        # incorrect_indeces = tf.where(tf.logical_not(correct.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y})))
        # print(incorrect_indeces.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}))
        #
        # for i in incorrect_indeces.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}):
        #     index = i[0]
        #     data = get_original_test_data('../data/train.csv', .8, num_articles)
        #     print("index: ", index, "article: ", data[index], "article length: ", len(re.sub(r'[^\w\s]', '', data[index][3]).split()))



train_neural_network(x)
