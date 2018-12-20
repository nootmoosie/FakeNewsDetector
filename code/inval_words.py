import csv
import gensim
import re


def get_inval_words(train_path, n_articles):
    ''' Method to read in the training data and shape it into the correct dimensions.
        Takes in file path, # articles, # words, and a range of article lengths
        Returns two tuples (train_x, train_y), (test_x, test_y) '''

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

    found_words = 0
    words_not_found = 0
    words_not_embedded = []
    for i, data in enumerate(train_data[1:]):
        rmv_pnc = re.sub(r'[^\w\s]', '', data[3])
        words = rmv_pnc.split()

        for word in words:
            if word in word_embeddings:
                found_words += 1
            elif word not in words_not_embedded:
                words_not_embedded.append(word)
                words_not_found += 1
            else:
                words_not_found += 1

    print("Words FOUND: ", found_words)
    print("Words NOT FOUND: ", words_not_found)
    return words_not_embedded, found_words, words_not_found

num_articles = 20800
w, words_found, words_not = get_inval_words('../data/train.csv', num_articles)
for i in w:
    print(i)

print("Words FOUND: ", words_found)
print("Words NOT FOUND: ", words_not)
print("Percentage of words embedded: ", (words_found/(words_found+words_not)))