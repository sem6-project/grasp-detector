import math
import itertools
from copy import deepcopy
from random import shuffle
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


def prepareModel(x_shape, y_shape):
    print('preparing model: x_shape >', x_shape, 'y_shape >', y_shape)
    model = Sequential([
        Dense(200, input_dim=x_shape),
        Activation('relu'),
        Dropout(0.2),
        Dense(y_shape)
    ])
    model.compile('adadelta', 'mse')
    return model


class DataBatcher(Sequence):
    def __init__(self, datapoints, batch_size):
        # self.x, self.y = x_set, y_set
        self.datapoints = datapoints
        self.batch_size = batch_size

    def __len__(self):
        # return math.floor(len(self.x) / self.batch_size)
        return int(len(self.datapoints) / self.batch_size)

    def __getitem__(self, idx):
        # batch_x = self.x[idx * self.batch_size : (idx+1) * self.batch_size]
        # batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
        batch = self.datapoints[idx * self.batch_size : (idx+1) * self.batch_size]
        # batch_x = np.array(list(map(lambda x: x.image_path, batch)))
        batch_x = np.array(list(map(lambda x: utils.readImage(x.image_path),
                                    batch)))
        batch_y = np.array(list(map(lambda y: y.rect.to_numpy(),
                                    batch)))
        return batch_x, batch_y


def trainTestSplitDatapoints(datapoints, test_size=0.2):
    marker = int(len(datapoints) * (1 - test_size))
    dp_copy = deepcopy(datapoints)
    shuffle(dp_copy)
    return dp_copy[:marker], dp_copy[marker:]


if __name__ == '__main__':
    datapoints = utils.prepareDataPoints('../../DataRaw')[:100]
    n_batches = 10

    train_set, test_set = trainTestSplitDatapoints(datapoints)

    train_seq = DataBatcher(train_set, batch_size=int(len(train_set) / n_batches))
    test_seq = DataBatcher(test_set, batch_size=int(len(test_set) / n_batches))

    x_shape = utils.readImage(datapoints[0].image_path).shape[-1]
    y_shape = datapoints[0].rect.to_numpy().shape[-1]
    model = prepareModel(x_shape, y_shape)

    retval = model.fit_generator(train_seq,
                                 validation_data=test_seq,
                                 epochs=1, verbose=2)
    print(retval)
