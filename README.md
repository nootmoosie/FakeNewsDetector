# FakeNewsDetector
This repo contains the Kaggle fake news dataset, https://www.kaggle.com/c/fake-news/data, and files for a classifier to use on this data.

The classifier uses pre-trained word embeddings, which are not included in this repo due to the size of the file. To run the files in the repo, the word embeddings file will have to be downloaded (https://drive.google.com/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM) and added into the code directory. 

To run our RNN model, just run `python RNN.py`.

# File breakdown:
baseline.py: running this file will run our baseline on our dataset and output the accuracy.

data_stats.py: this file was used for collecting statistics about the dataset

inval_words.py: this file was also used for collecting statistics about the words in the dataset vs. the words in the pre trained word embeddings

make_plot.py: this file contains the methods we use inside of RNN.py to create plots of the accuracy over the amount of epochs

process_data.py: this file loads the pre trained word embeddings, processes the data files, splits the data into two sets, gets the word embeddings for both sets, and filters the training set

RNN.py: this file contains our RNN, and running it takes a while. It will output the loss over all epochs, and the accuracy at the end, as well as plot the accuracy over all the epochs. Towards the beginning of the file, there are parameters that can be changed, such as the number of articles we want to use (num_articles), how many epochs we want to run (hm_epochs), the number of classes (n_classes), the number of words we take from each article (n_chunks), the size of the hidden layers (rnn_size), and the learning rate (l_rate). 