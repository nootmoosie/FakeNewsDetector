from process_data import process_csv_data
from process_data import withhold_data
# add import statement for neural network module
import numpy as np
import gensim

# load in the word embeddings
word_emb = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

# create a neural network from our module


# load in the training and testing data



# matplotlib or pyplot - some type of plot of accuracies vs. some type of parameter/epochs
# baseline accuracy
# accuracy based on how much data is withheld

def accuracy_withheld(test_data, predictions):
	''' function for calculating accuracy 
	of predictions based on some withheld 
	data '''
	true_vals = list(label for (id_, title, author, test, label) in test_data)
	true_vals = np.array(true_vals)
	preds = np.array(predictions)
	total = len(test_data)
	correct = np.sum(true_vals == preds)
	return correct/total



