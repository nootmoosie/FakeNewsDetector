import os
import csv
import numpy as np
import gensim
import re

def process_csv_data(train_path, test_path, n_train, n_test):
    WORD_NUM = 100

    # Load Google's pre-trained Word2Vec model.
    word_embeddings = model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    train_data = []
    test_data = []

    train_file = open(train_path, 'r', encoding='utf8')
    test_file = open(test_path, 'r', encoding='utf8')

    train_str = train_file.readlines()
    test_str = test_file.readlines()

    n_1 = 0  # TEMP FIX, WANT TO PASS BATCHES INTO READER TO GET ALL DATA.
    for entry in csv.reader(train_str, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if n_1 < n_train:
            # print('id: ', entry[0])
            # print('title: ', entry[1])
            # print('author: ', entry[2])
            # print('text: ', entry[3])
            # print('label: ', entry[4])
            train_data.append(entry)
            n_1 += 1
        else:
            break

    n_2 = 0
    for entry in csv.reader(test_str, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if n_2 < n_test:
            test_data.append(entry)
            n_2 += 1
        else:
            break

    print(np.array(train_data).shape)
    print(np.array(test_data).shape)
    # Data now organized in nested list, now need to embed the words for each article.

    training = []
    for data in train_data[1:]:
        print("Converting article: ", data[1], "...")
        rmv_pnc = re.sub(r'[^\w\s]', '', data[3])
        words = rmv_pnc.split()
        word_matrix = []
        label = data[4]
        w_cnt = 0
        for word in words:
            # print(word)
            if w_cnt >= WORD_NUM:
                break
            if word in word_embeddings:
                # print(word, " was in the embedding")
                embedding = word_embeddings[word]
                word_matrix.extend(embedding)
                w_cnt += 1
        training.append(word_matrix)
        # processed_data = np.array(word_matrix).flatten()
        # training.append((processed_data, label))
        print("words in article: ", len(data[3].split()))
        print("shape of word matrix: ", np.array(word_matrix).shape)

    training = np.array(training)
    print(training.shape)

    # # testing size of training data
    # for article in training:
    #     print(article[0].shape, " ", article[1])
    #     if article[0].shape[0]%300 is not 0:
    #         print("bruh its rong")

    testing = []
    for data in test_data[1:]:
        # print("Converting article: ", data[1], "...")
        rmv_pnc = re.sub(r'[^\w\s]', '', data[3])
        words = rmv_pnc.split()
        word_matrix = []
        for word in words:
            # print(word)
            if word in word_embeddings:
                # print(word, " was in the embedding")
                embedding = word_embeddings[word]
                word_matrix.append(embedding)

        processed_data = np.array(word_matrix).flatten()
        testing.append(processed_data)
        # print("words in article: ", len(data[3].split()))
        # print("shape of word matrix: ", np.array(word_matrix).shape)

    return training, testing


def withhold_data(training_data, percentage):
    train_size = int(percentage * len(training_data))
    train_data = training_data[:train_size]
    withheld_data = training_data[train_size:]

    return train_data, withheld_data


train_data, test_data = process_csv_data('../data/train.csv', '../data/test.csv', 10, 5)



