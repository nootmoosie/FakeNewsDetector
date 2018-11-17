import tensorflow as tf
import gensim
import numpy as np
import re

from tensorflow.contrib import rnn
from process_data import process_train_data, train_test_split, get_original_test_data

num_articles = 1000
hm_epochs = 50
n_classes = 2
# batch_size = 128

chunk_size = 300
n_chunks = 100
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
    epoch_x, epoch_y = process_train_data('../data/train.csv', num_articles, n_chunks)
    train, test = train_test_split(epoch_x, epoch_y, .8)
    epoch_x, epoch_y = train[0], train[1]
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # predictions = []
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
            # predictions.append(pred)
            epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        print('correct!!!!; ', correct.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: epoch_y}))

        # num_correct = 0
        # print(epoch_y.shape)
        # for i, p in enumerate(pred):
        #     print("prediction: {}    true value: {}".format(p, epoch_y[i]))
        #     if np.argmax(p) == np.argmax(epoch_y):
        #         print("CORRECT BITCH!!!")
        #         num_correct += 1
        #
        # accuracy = num_correct/len(pred)
        # print("TEST ACCURACY: ", accuracy)

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Training Accuracy:',
              # accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))
              accuracy.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: epoch_y}))

        test_x, test_y = test[0], test[1]
        print('Test Accuracy:',
              # accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))
              accuracy.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}))

        incorrect_indeces = tf.where(tf.logical_not(correct.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y})))
        print(incorrect_indeces.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}))

        for i in incorrect_indeces.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}):
            index = i[0]
            data = get_original_test_data('../data/train.csv', .8, num_articles)
            print("index: ", index, "article: ", data[index], "article length: ", len(re.sub(r'[^\w\s]', '', data[index][3]).split()))



train_neural_network(x)
