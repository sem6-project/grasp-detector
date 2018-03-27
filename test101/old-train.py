import math
import itertools
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import utils


def batchDataset(datapoints, batch_size=50, test_size=0.2):
    '''Batches given datapoints and yields dataset for each batch.
    Yields: tuple : X_train, X_test, Y_train, Y_test
                       for each batch
    '''
    for idx in itertools.count():
        batch = datapoints[idx*batch_size:(idx+1)*batch_size]
        if not batch:
            break
        yield prepareDataset(batch, test_size)


def prepareDataset(datapoints, test_size=0.2):
    '''Prepares dataset to feed to Keras from given datapoints
    Returns: tuple : X_train, X_test, Y_train, Y_test
    '''
    X = list(map(lambda x: x.image_path, datapoints))
    Y = list(map(lambda y: y.rect.to_numpy(), datapoints))
    return train_test_split(X, Y, test_size=test_size)


def prepareModel(x_shape, y_shape):
    model = Sequential([
        Dense(200, input_dim=x_shape),
        Activation('relu'),
        Dropout(0.2),
        Dense(y_shape)
    ])
    model.compile('adadelta', 'mse')
    return model


class DataSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
        return batch_x, batch_y


if __name__ == '__main__':
    datapoints = utils.prepareDataPoints('../../DataRaw')
    batches = batchDataset(datapoints)
    for (X_train, X_test, Y_train, Y_test) in batches:

        print((X_train, X_test, Y_train ,Y_test))


    # train_batch = list(train_batch)
    # test_batch = list(test_batch)

    # print(len(train_batch), len(test_batch))
    # print(train_batch[0])
    # print(test_batch[0])

    # x, y = next(train_batch), next(test_batch)
    # print('train', x)
    # print('test', y)
