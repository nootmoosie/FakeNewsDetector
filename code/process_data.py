import os
import csv
import numpy as np
import gensim
import re
from scipy import stats


def process_train_data(train_path, n_articles, n_words, split_percentage, max_words=2000, min_words=50, split_shuffle=False):
    ''' Method to read in the training data and shape it into the correct dimensions.
        Takes in file path, # articles, # words, and a range of article lengths
        Returns two tuples (train_x, train_y), (test_x, test_y) '''


    # if shuffle:
    #     zipped = zip(x, y)
    #     zipped = list(zipped)
    #     np.random.shuffle(zipped)
    #     x, y = zip(*zipped)
    #     print(type(x))



    # train_x = x[:split_idx]
    # test_x = x[split_idx:]

    # train_y = y[:split_idx]
    # test_y = y[split_idx:]

    # return (train_x, train_y), (test_x, test_y)

    # load in the pre trained word vectors
    word_embeddings = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    train_data = []
    train_file = open(train_path, 'r', encoding='utf8')
    train_str = train_file.readlines()

    csv.field_size_limit(100000000)
    n_1 = 0
    for entry in csv.reader(train_str, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if n_1 < n_articles:
            train_data.append(entry)
            n_1 += 1
        else:
            break

    split_idx = int(split_percentage*len(train_data))

    train_x = np.zeros((split_idx, n_words*300))
    train_y = np.zeros((split_idx, 2))
    test_x = np.zeros((n_articles-split_idx, n_words*300))
    test_y = np.zeros((n_articles-split_idx, 2))
    # print(train_x.shape)
    # print(train_y.shape)

    if split_shuffle:
        np.random.shuffle(train_data)

    removed = 0
    ignored = 0

    train_set = True

    for i, data in enumerate(train_data[1:]):
        rmv_pnc = re.sub(r'[^\w\s]', '', data[3])
        words = rmv_pnc.split()

        if (min_words < len(words) < max_words) or not train_set:
            n_2 = 0
            word_matrix = []
            total_words = len(words)
            found_words = 0
            for word in words:
                if n_2 < n_words:
                    if word in word_embeddings:
                        embedding = word_embeddings[word]
                        word_matrix.extend(embedding)
                        n_2 += 1
                        found_words += 1

            words_used = min(total_words, n_words)

            if not train_set or ((found_words/words_used) > 0.5):
                if len(word_matrix) < train_x.shape[1]:
                    padding = np.zeros(train_x.shape[1]-len(word_matrix))
                    word_matrix = np.append(padding, word_matrix)
                    # word_matrix.extend(padding)

                label = data[4]

                if int(label) == 0:
                    if train_set:
                        train_y[i] = [1, 0]
                    else:
                        test_y[i-split_idx] = [1, 0]
                else:
                    if train_set:
                        train_y[i] = [0, 1]
                    else:
                        test_y[i-split_idx] = [0, 1]
                if train_set:
                    train_x[i] = word_matrix
                else:
                    test_x[i-split_idx] = word_matrix
            else:
                ignored += 1
        else:
            removed += 1

        if i == split_idx-1:
            train_set = False

    print("train x shape: ", train_x.shape)
    print("train y shape: ", train_y.shape)
    print("test x shape: ", test_x.shape)
    print("test y shape: ", test_y.shape)

    print("Articles removed because of length: ", removed)
    print("Articles removed because of unseen words: ", ignored)

    # Returns two tuples (train_x, train_y), (test_x, test_y)
    return (train_x, train_y), (test_x, test_y)




# ---------------------------------

def get_original_test_data(path, percentage, n_articles):

    split_idx = int(percentage*n_articles-1)
 
    test_data = []
 
    file = open(path, 'r', encoding='utf8')
 
    string = file.readlines()
 
    n_1 = 0  # TEMP FIX, WANT TO PASS BATCHES INTO READER TO GET ALL DATA.
    for entry in csv.reader(string, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if n_1 < n_articles:
            if n_1 < split_idx:
                # print('id: ', entry[0])
                # print('title: ', entry[1])
                # print('author: ', entry[2])
                # print('text: ', entry[3])
                # print('label: ', entry[4])
                n_1 += 1
            else:
                test_data.append(entry)
                n_1 += 1
        else:
            break

    test_data.pop(0)
    print(np.array(test_data).shape)
     # Data now organized in nested list, now need to embed the words for each article.
 
    return test_data


#---------------------------------

# TODO: Implement optional shuffle
def train_test_split(x, y, percentage, shuffle=False):
    if shuffle:
        zipped = zip(x, y)
        zipped = list(zipped)
        np.random.shuffle(zipped)
        x, y = zip(*zipped)
        print(type(x))

    split_idx = int(percentage*len(x))

    train_x = x[:split_idx]
    test_x = x[split_idx:]

    train_y = y[:split_idx]
    test_y = y[split_idx:]

    return (train_x, train_y), (test_x, test_y)


def withhold_data(training_data, percentage):
    train_size = int(percentage * len(training_data))
    train_data = training_data[:train_size]
    withheld_data = training_data[train_size:]

    return train_data, withheld_data


#train_data, test_data = process_csv_data('../data/train.csv', '../data/test.csv', 10, 5)

# tx, ty = process_train_data('../data/train.csv', 10, 100)

# x, y = process_train_data('../data/train.csv', 50, 100)

# train, test = train_test_split(x, y, 0.6)

#test_data = get_original_test_data('../data/train.csv', .8, 1000)
#print(test_data)




