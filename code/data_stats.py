''' module to get stats of the training data, for figuring out good hyperparameters '''
import os
import csv
import numpy as np
import re
import statistics
from matplotlib import pyplot

def get_stats(train_path, attribute='articles', n_rows=5000, extra_plot=None, chars=False):
    ''' method to process the data and
    calculate all of the useful
    statistics '''

    spot = None
    if attribute == 'articles':
        spot = 3
    elif attribute == 'authors':
        spot = 2
    elif attribute == 'titles':
        spot = 1

    # read in all of the training data
    train_data = read_train_data(train_path, n_rows, attribute)

    # get the lengths of the articles, in words
    if not chars:
        lengths = [len(row[spot].split()) for row in train_data]

    if chars:
        lengths = [len(row[spot]) for row in train_data]

    if n_rows == 'all':
        n_rows = len(train_data)


    # calculate and print all of the statistics for the article lengths
    print("Statistics of ", attribute, " Lengths")
    print("Number of ", attribute, ": ", n_rows)
    print("Mean: ", statistics.mean(lengths))
    print("Median: ", statistics.median(lengths))
    print("Mode: ", statistics.mode(lengths))
    print("Minimum: ", min(lengths))
    print("Maximum: ", max(lengths))
    print("Standard Deviation: ", statistics.stdev(lengths))

    ids = [i for i in range(n_rows)]

    lengths.sort()

    title = 'Lengths of ' + attribute
    label = 'length of ' + attribute

    pyplot.title(title)
    pyplot.xlabel(label)
    pyplot.ylabel('frequency')
    pyplot.bar(ids, lengths)
    pyplot.show()

    idx = int(n_rows/2)

    title2 = title + ' (smaller half)'
    pyplot.title(title2)
    pyplot.xlabel(label)
    pyplot.ylabel('frequency')
    pyplot.bar(ids[:idx], lengths[:idx])
    pyplot.show()

    title3 = title + ' (larger half)'
    pyplot.title(title3)
    pyplot.xlabel(label)
    pyplot.ylabel('frequency')
    pyplot.bar(ids[idx:], lengths[idx:])
    pyplot.show()

    if extra_plot is not None:
        title4 = title + ' (up until index ' + extra_plot + ')'
        pyplot.title(title4)
        pyplot.xlabel(label)
        pyplot.ylabel('frequency')
        pyplot.bar(ids[:extra_plot], lengths[:extra_plot])
        pyplot.show()



def read_train_data(train_path, n_rows, attribute='articles'):
    ''' method to read in the training data
    and shape it into the correct dimensions
    for processing it and collecting stats '''

    spot = None
    if attribute == 'articles':
        spot = 3
    elif attribute == 'authors':
        spot = 2
    elif attribute == 'titles':
        spot = 1

    train_data = []
    train_file = open(train_path, 'r', encoding='utf8')
    train_str = train_file.readlines()

    csv.field_size_limit(100000000)

    if n_rows == 'all':
        n_rows = len(train_str)


    n_1 = 0
    for entry in csv.reader(train_str, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if n_1 < n_rows:
            train_data.append(entry)
            train_data[n_1][spot] = re.sub(r'[^\w\s]', '', train_data[n_1][spot])
            n_1 += 1
        else:
            break

    return train_data