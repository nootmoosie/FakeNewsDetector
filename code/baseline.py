import numpy as np

from sklearn.neural_network import MLPClassifier
from process_data import process_train_data, train_test_split, get_original_test_data
from make_plot import plot_acc

num_articles = 20800
n_chunks = 50

print("processing data...")
#epoch_x, epoch_y = process_train_data('../data/train.csv', num_articles, n_chunks, .8)
(train_x, train_y), (test_x, test_y) = process_train_data('../data/train.csv', num_articles, n_chunks, .8)
#train, test = train_test_split(epoch_x, epoch_y, .8)

#epoch_x, epoch_y = train[0], train[1]
print("train data shape: ", train_x.shape)
print("train label shape: ", train_y.shape)

#test_x, test_y = test[0], test[1]

neural_net = MLPClassifier()
neural_net.fit(train_x, train_y)
preds = neural_net.predict(test_x)

correct = 0
for i, p in enumerate(preds):
    if np.array_equal(p, test_y[i]):
        correct += 1

acc = correct/len(test_y)

print("Baseline accuracy achieved with sklearn standard neural network classifier:")
print("Eval acc: ", acc)