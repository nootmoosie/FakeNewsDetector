''' module to get stats of the training data, for figuring out good hyperparameters '''
import os
import csv
import numpy as np
import re
import statistics
from matplotlib import pyplot

def get_stats(train_path, n_articles=5000, extra_plot=None):
    ''' method to process the data and
    calculate all of the useful
    statistics '''

    # read in all of the training data
    train_data = read_train_data(train_path, n_articles)

    # get the lengths of the articles, in words
    lengths = [len(row[3].split()) for row in train_data]


    # calculate and print all of the statistics for the article lengths
    print("Statistics of Article Lengths")
    print("Number of Articles: ", n_articles)
    print("Mean: ", statistics.mean(lengths))
    print("Median: ", statistics.median(lengths))
    print("Mode: ", statistics.mode(lengths))
    print("Minimum: ", min(lengths))
    print("Maximum: ", max(lengths))
    print("Standard Deviation: ", statistics.stdev(lengths))

    ids = [i for i in range(n_articles)]

    lengths.sort()

    pyplot.title('Lengths of articles')
    pyplot.xlabel('length of article')
    pyplot.ylabel('frequency')
    pyplot.bar(ids, lengths)
    pyplot.show()

    idx = int(n_articles/2)

    pyplot.bar(ids[:idx], lengths[:idx])
    pyplot.show()

    pyplot.bar(ids[idx:], lengths[idx:])
    pyplot.show()

    if extra_plot is not None:
        pyplot.bar(ids[:extra_plot], lengths[:extra_plot])
        pyplot.show()



def read_train_data(train_path, n_articles):
    ''' method to read in the training data
    and shape it into the correct dimensions
    for processing it and collecting stats '''

    train_data = []
    train_file = open(train_path, 'r', encoding='utf8')
    train_str = train_file.readlines()

    n_1 = 0
    for entry in csv.reader(train_str, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if n_1 < n_articles:
            train_data.append(entry)
            train_data[n_1][3] = re.sub(r'[^\w\s]', '', train_data[n_1][3])
            n_1 += 1
        else:
            break

    return train_data