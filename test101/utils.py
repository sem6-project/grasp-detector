import os
import cv2
import numpy as np
import math
import itertools


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
    def __init__(self, x, y, w, h, t, coords):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.t = t
        # in case actual coordinates are required
        self.coordinates = coords

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


def reorderCoordinates(X, Y):
    sortingKey = lambda I: euclideanDistance(I[0][0], I[0][1], I[1][0], I[1][1])
    d1, d2 = sorted(itertools.combinations(zip(X,Y), 2),
                    key=sortingKey,
                    reverse=True)[:2]

    # have two diagonals going up to down
    # first diagonal,... top coordinate up
    d1 = sorted(d1, key=lambda I: I[1])   # sort with Y
    # second diagonal,... top coordinate up
    d2 = sorted(d2, key=lambda I: I[1])   # sort with Y

    # get the left point first
    d1, d2 = (d1, d2) if d1[0][0] < d2[0][0] else (d2, d1)

    # just return the coordinates from the ordered diagonals
    return [d1[0], d1[1], d2[0], d2[1]]


def getRectangles(coords):
    X = [0, 0, 0, 0]
    Y = [0, 0, 0, 0]
    for i, c in enumerate(coords):
        i = i % 4
        x, y = float(c[0]), float(c[1])
        X[i], Y[i] = x, y
        if (i+1) % 4 == 0:
            # the coordinates need to be reordered
            # ordering - clockwise starting from top left point
            coords = reorderCoordinates(X, Y)
            W = euclideanDistance(coords[0][0], coords[0][1],
                                  coords[1][0], coords[1][1])
            H = euclideanDistance(coords[0][0], coords[0][1],
                                  coords[3][0], coords[3][1])
            T = 0
            try:
                # T = (y2-y1) / (x2-x1)
                T = math.atan((coords[1][1] - coords[0][1]) / (coords[1][0] - coords[0][0]))
            except ZeroDivisionError:
                T = 0

            yield Rectangle(coords[0][0], coords[0][1], W, H, T, coords)


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
    data = []
    for each in datapoints:
        image_path = each.image_path
        data.append({
            'image_path': image_path,
            'rectangle': each.rect.to_tuple()
        })

    with open(dest_file, 'w') as f:
        f.write(json.dumps(data, indent=2))


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

    consent = input('Save datapoints to datapoints.json ? (y / n) ')
    if consent.lower() == 'y':
        print('saving')
        storeDataPoints(datapoints, dest_json)
    else:
        print('not saving')

    # pickleDataPoints(datapoints, dest_file)
    # storeDataPoints(datapoints, dest_json)
    # print('datapoints read and stored to file')

    # dps = loadPickledDataPoints(dest_file)
    # dps = loadJsonDataPoints(dest_json)
    # print('read datapoints again from file')
    # print(dps[0])
