import math
import itertools
from copy import deepcopy
from random import shuffle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, rmsprop
from keras.utils import Sequence
import keras.backend as K

from sklearn.model_selection import train_test_split

import utils


def rectangle_guessing_loss(y_true, y_pred):
    y_true_np = K.eval(y_true)
    y_pred_np = K.eval(y_pred)
    y_true_iter = np.nditer(y_true_np, order='C')
    y_pred_iter = np.nditer(y_pred_np, order='C')
    loss = 0.0
    count = 0
    for y1, y2 in zip(y_true_iter, y_pred_iter):
        loss += utils.IOU(y1, y2)
        count += 1

    return (loss / count)



def prepareExampleModel(x_shape, y_shape):
    print('preparing model: x_shape >', x_shape, 'y_shape >', y_shape)
    model = Sequential([
        Dense(200, input_dim=x_shape),
        Activation('relu'),
        Dropout(0.2),
        Dense(y_shape)
    ])
    # model.compile('adadelta', loss=rectangle_guessing_loss)
    # model.compile('adadelta', 'cosine_proximity')
    model.compile('adadelta', 'mse')
    return model


def prepareConvModel(x_shape, y_shape, batch_size):
    print('preparing model with shape', x_shape, y_shape)

    model = Sequential([
    Conv2D(batch_size, (3, 3), padding='same', input_shape=(486,))
        # Conv2D(32, (3, 3), padding='same', input_shape=x_shape),
        # Activation('relu'),

        # Conv2D(32, (3, 3)),
        # Activation('relu'),

        # MaxPooling2D(pool_size=(2,2)),
        # Dropout(0.25),

        # Conv2D(64, (3, 3), padding='same'),
        # Activation('relu'),

        # Conv2D(64, (3, 3)),
        # Activation('relu'),

        # MaxPooling2D(pool_size=(2, 2)),
        # Dropout(0.25),

        # Flatten(),
        # Dense(512),
        # Activation('relu'),
        # Dropout(0.25),
        # Dense(y_shape),
        # Activation('softmax')
    ])

    opt = rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    return model


class DataBatcher(Sequence):
    def __init__(self, datapoints, batch_size):
        # self.x, self.y = x_set, y_set
        self.datapoints = datapoints

        if batch_size == 0:
            self.batch_size = len(self.datapoints)
        else:
            self.batch_size = batch_size

        print('batcher created of batch_size', self.batch_size)

    def __len__(self):
        # return math.floor(len(self.x) / self.batch_size)
        return int(len(self.datapoints) / self.batch_size)

    def __getitem__(self, idx):
        # batch_x = self.x[idx * self.batch_size : (idx+1) * self.batch_size]
        # batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
        # print('batch > __getitem__ for idx', idx)
        batch = self.datapoints[idx * self.batch_size : (idx+1) * self.batch_size]
        # batch_x = np.array(list(map(lambda x: x.image_path, batch)))
        images = list(map(lambda x: x.get_image(), batch))
        rectangles = list(map(lambda x: x.rect.to_numpy(), batch))
        # batch_x = np.array(list(map(lambda x: utils.readImage(x.image_path),
        #                             batch)))
        # batch_y = np.array(list(map(lambda y: y.rect.to_numpy(),
        #                             batch)))

        batch_x = np.array(images)
        # batch_x = batch_x.reshape(len(batch), -1)
        # batch_x = (batch_x.reshape(self.batch_size, -1) - batch_x.mean()) / batch_x.std()

        batch_y = np.array(rectangles)
        # print('getitem (debug) :', type(batch_x), len(batch_x))
        return batch_x, batch_y


def trainTestSplitDatapoints(datapoints, test_size=0.2):
    marker = int(len(datapoints) * (1 - test_size))
    dp_copy = deepcopy(datapoints)
    shuffle(dp_copy)
    return dp_copy[:marker], dp_copy[marker:]


def determineBatchSize(train_size, test_size):
    n_total = min(train_size, test_size)
    # try to keep 50 images in a batch
    n_per_batch = 50
    n_batches = math.ceil(n_total / n_per_batch)
    return n_batches, n_per_batch


def determineModelShape(some_batch):
    some_x_batch, some_y_batch = some_batch
    some_x, some_y = some_x_batch[0], some_y_batch[0]
    # x_shape, y_shape = some_x.shape[-1], some_y.shape[-1]
    x_shape, y_shape = some_x.shape, some_y.shape
    return x_shape, y_shape


def main():
    import sys

    try:
        dataRawDir = sys.argv[1]
        nImages = int(sys.argv[2])
        epochs = int(sys.argv[3])
    except IndexError:
        dataRawDir = '../../DataRaw'
        nImages = 10
        epochs = 1

    # shit happens here
    datapoints          = utils.prepareDataPoints(dataRawDir)[:nImages]

    train_set, test_set = trainTestSplitDatapoints(datapoints)
    n_batches, n_images_per_batch  = determineBatchSize(len(train_set), len(test_set))

    train_seq           = DataBatcher(train_set, batch_size=int(len(train_set) / n_batches))
    test_seq            = DataBatcher(test_set, batch_size=int(len(test_set) / n_batches))

    import pdb; pdb.set_trace()

    x_shape, y_shape    = determineModelShape(train_seq[0])

    # model               = prepareModel(x_shape, y_shape)
    # model               = prepareExampleModel(x_shape, y_shape)
    print('testing last element of batch sequence ', 'train_seq', not(not(train_seq[-1])), 'test_seq', not(not(test_seq[-1])))

    model               = prepareConvModel(x_shape, y_shape, n_images_per_batch)

    retval              = model.fit_generator(train_seq,
                                              validation_data=test_seq,
                                              epochs=epochs, verbose=2)
    print(retval)

    # evaluating the results
    y_pred = model.predict(test_seq[0][0])
    # calculate mean IOU
    total_IOU = 0
    for bbox_pred, bbox_test in zip(
            y_pred.reshape(-1, 5), test_seq[0][1].reshape(-1, 5)
    ):
        total_IOU += utils.IOU(bbox_pred, bbox_test)

    mean_IOU = total_IOU / len(y_pred)
    print('mean IOU', mean_IOU)



if __name__ == '__main__':
    main()
