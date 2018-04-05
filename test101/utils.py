import os
import cv2
import numpy as np
import math


def IOU(bbox1, bbox2):
    '''Calculates overlap b/w two rectangles'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    w_I = min(x1+w1, x2+w2) - max(x1, x2)
    h_I = min(y1+h1, y2+h2) - max(y1, y2)

    if w_I <= 0 or h_I <= 0: # no overlap
        return 0

    I = w_I * h_I
    U = w1*h1 + w2*h2 - I
    return I / U


class Rectangle(object):
    def __init__(self, x, y, w, h, t):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.t = t

    def to_tuple(self):
        return (self.x, self.y, self.w, self.h, self.t)

    def to_numpy(self):
        return np.array(self.to_tuple())

    def __str__(self):
        return f'Rectangle({self.to_tuple()})'

    def __repr__(self):
        return str(self)


class DataPoint(object):
    def __init__(self, image_path, rect):
        self.image_path = image_path
        self.rect = rect

    def get_image(self):
        # return cv2.imread(self.image_path)
        return cv2.imread(self.image_path, cv2.COLOR_BGR2GRAY)

    def __str__(self):
        return f'DataPoint({self.rect})'

    def __repr__(self):
        return str(self)


def readImage(image_path):
    # return cv2.imread(image_path)
    return cv2.imread(self.image_path, cv2.COLOR_BGR2GRAY)


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))


def getRectangles(coords):
    X = [0, 0, 0, 0]
    Y = [0, 0, 0, 0]
    for i, c in enumerate(coords):
        i = i % 4
        x, y = float(c[0]), float(c[1])
        X[i], Y[i] = x, y
        if (i+1) % 4 == 0:
            W = euclideanDistance(X[0], Y[0], X[1], Y[1])
            H = euclideanDistance(X[0], Y[0], X[3], Y[3])
            T = 0
            try:
                T = math.atan((Y[1] - Y[0]) / (X[1] - X[0]))
            except ZeroDivisionError:
                T = 0

            yield Rectangle(X[0], Y[0], W, H, T)


def prepareDataPoints(raw_data_path):
    raw_data_path = os.path.abspath(raw_data_path)

    isImageRGBFile = lambda f: f.startswith('pcd') and f.endswith('r.png')
    isImageGrayFile = lambda f: f.startswith('pcd') and f.endswith('gray.png')
    isPosRectFile = lambda f: f.startswith('pcd') and (f.endswith('pos.txt') or f.endswith('cpos.txt'))
    isRequiredDirectory = lambda d: d.isnumeric()  # to avoid backgrounds/
    isRequiredFile = lambda f: isImageRGBFile(f) or isImageGrayFile(f) or isPosRectFile(f)
    # take care to replace cpos.txt before pos.txt
    getImageId = lambda f: f.replace('pcd', '').replace('r.png', '').replace('gray.png', '').replace('cpos.txt', '').replace('d.txt', '').replace('pos.txt', '')

    fileMap = {}
    for dirpath, _, files in os.walk(raw_data_path):
        if isRequiredDirectory(os.path.basename(dirpath)):
            for file in filter(isRequiredFile, files):
                imageId = getImageId(file)
                print(imageId, file)
                if imageId not in fileMap:
                    fileMap[imageId] = {'rgb': None, 'gray': None, 'pos_rect': None}
                    # initial values applied because there are missing values causing error

                if isImageRGBFile(file):
                    fileMap[imageId]['rgb'] = os.path.join(dirpath, file)
                if isImageGrayFile(file):
                    fileMap[imageId]['gray'] = os.path.join(dirpath, file)
                if isPosRectFile(file):
                    fileMap[imageId]['pos_rect'] = os.path.join(dirpath, file)

    print('total', len(fileMap), 'images read')
    datapoints = []

    for each in fileMap.values():
        # rgb_img = each['rgb']
        gray_img = each['gray']
        pos_rect_file = each['pos_rect']
        # image = cv2.imread(gray_img)
        image = gray_img
        if  pos_rect_file:
            with open(pos_rect_file) as f:
                coords = map(lambda x: x.split(), f.readlines())
                for rect in getRectangles(coords):
                    datapoints.append(DataPoint(image, rect))

    return datapoints


def pickleDataPoints(datapoints, dest_file):
    import pickle
    with open(dest_file, 'w') as f:
        f.write(pickle.dumps(datapoints))


def loadPickledDataPoints(source_file):
    import pickle
    return pickle.load(source_file)


def storeDataPoints(datapoints, dest_file):
    import json
    with open(dest_file, 'w') as f:
        f.write(json.dumps(datapoints))


def loadJsonDataPoints(source_file):
    import json
    with open(source_file) as f:
        datapoints = json.loads(f.read())
    return datapoints



if __name__ == '__main__':
    data_path = '../../DataRaw'
    dest_file = 'datapoints.pickle'
    dest_json = 'datapoints.json'
    datapoints = prepareDataPoints(data_path)
    print('prepared datapoints of length', len(datapoints))

    # pickleDataPoints(datapoints, dest_file)
    # storeDataPoints(datapoints, dest_json)
    # print('datapoints read and stored to file')

    # dps = loadPickledDataPoints(dest_file)
    # dps = loadJsonDataPoints(dest_json)
    # print('read datapoints again from file')
    # print(dps[0])
