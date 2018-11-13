import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable

np.random.seed(69420)

MAX_LEN = 500
EMBEDDING_SIZE = 300
BATCH_SIZE = 32
EPOCH = 40
DATA_SIZE = 5200
INPUT_SIZE = 300

def batch(tensor, batch_size):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: ()])
        i += 1

class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, X_list, y_list):

        # train one epoch
        loss_list = []
        acc_list = []
        for X, y in zip(X_list, y_list):
            X_v = Variable(torch.from_numpy(np.swapaxes(X,0,1)).float())
            y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)

            self.optimizer.zero_grad()
            y_pred = self.model(X_v, self.model.initHidden(X_v.size[1]))
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            # for log
            loss_list.append(loss.data[0])
            classes = torch.topk(y_pred, 1)[1].data.numpy()


