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
# error analysis - what kind of examples did the classifier fail at

def incorrect_classifications(test_data, predictions):
	''' function for extracting all of the data that
	was incorrectly classified for further analysis
	and evaluation '''
	true_vals = np.array(list(label for (id_, title, author, test, label) in test_data))
	preds = np.array(predictions)
	incorrect = np.where(true_vals != preds)[0]
	vals = []
	for i in incorrect:
		vals.append(test_data[i])
	# print(vals)
	return vals

def accuracy_withheld(test_data, predictions):
	''' function for calculating accuracy 
	of predictions based on some withheld 
	data '''
	true_vals = np.array(list(label for (id_, title, author, test, label) in test_data))
	preds = np.array(predictions)
	total = len(test_data)
	correct = np.sum(true_vals == preds)
	return correct/total



