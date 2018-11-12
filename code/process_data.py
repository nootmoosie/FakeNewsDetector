import os
import numpy as np

def process_csv_data(train_path, test_path):
    train_data = []
    test_data = []

    train_file = open(train_path, 'r', encoding='utf8')
    test_file = open(test_path, 'r', encoding='utf8')

    for article in train_file:
        train_data.append(article)

    for article in test_file:
        test_data.append(article)

    return train_data, test_data


train_data, test_data = process_csv_data('../data/train.csv', '../data/test.csv')

print('train data format: ' + train_data[0])
print('test data format: ' + test_data[0])

