import os
import csv
import numpy as np


def process_csv_data(train_path, test_path):
    train_data = []
    test_data = []

    train_file = open(train_path, 'r', encoding='utf8')
    test_file = open(test_path, 'r', encoding='utf8')

    train_str = train_file.read().splitlines()
    test_str = test_file.read().splitlines()


    # SASHAS:
    # for data_point in train_str:
    #     line = csv.reader([data_point], quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    #     entry = list(i for i in line)
    #     print(entry)
    #     #print('id: ', entry[0])
    #     #print('title: ', entry[1])
    #     #print('author: ', entry[2])
    #     #print('text: ', entry[3])
    #     #print('label: ', entry[4])
    #     #train_data.append(entry)

    # print(type(train_str))
    # batch = train_str[:3]
    # print(batch)
    # NATES:

    for entry in csv.reader(train_str, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        print('id: ', entry[0])
        print('title: ', entry[1])
        print('author: ', entry[2])
        print('text: ', entry[3])
        print('label: ', entry[4])
        train_data.append(entry)

    for entry in csv.reader(test_str, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        # print('id: ', entry[0])
        # print('title: ', entry[1])
        # print('author: ', entry[2])
        # print('text: ', entry[3])
        # print('label: ', entry[4])
        test_data.append(entry)


    return train_data, test_data


def withhold_data(training_data, percentage):
    train_size = int(percentage * len(training_data))
    train_data = training_data[:train_size]
    withheld_data = training_data[train_size:]

    return train_data, withheld_data


train_data, test_data = process_csv_data('../data/train.csv', '../data/test.csv')



