
from matplotlib import pyplot as plt


def plot_acc(train_accuracies, test_accuracies, n_epochs, rnn_size, words_evaluated, t_acc, eval_acc):

    plt.plot(range(n_epochs), train_accuracies, 'b-', label='Train')
    plt.plot(range(n_epochs), test_accuracies, 'r-', label='Eval')
    plt.title("Train and Evaluation Accuracy with RNN size {} and {} words evaluated".format(rnn_size, words_evaluated))
    plt.xlabel("Epoch\nFinal Train Accuracy: {}%\nFinal Eval Accuracy: {}%".format(int(t_acc*100), int(eval_acc*100)))
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
