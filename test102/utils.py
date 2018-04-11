'''Helper functions to be used throughout the project
'''

import glob
import os
import math
import itertools
import cv2
import numpy as np
from copy import deepcopy
import random


class Point(object):
    def __init__(self, x :float, y :float):
        self.x = x
        self.y = y

    def to_tuple(self) -> tuple:
        return (self.x, self.y)
        
    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y)
    
    def __mul__(self, scalar :float):
        return Point(self.x*scalar, self.y*scalar)
    
    def __str__(self):
        return f'Point({self.x}, {self.y})'

    def __repr__(self):
        return self.__str__()


class DataPoint(object):
    def __init__(self, image_path :str, vertices :list):
        '''Parameters:
            image_path : str : path to the image
            vertices: str : list of :Point:
        '''
        self.image_path = image_path
        self.vertices = vertices

    def get_image(self, force_gray=True) -> np.array:
        img = cv2.imread(self.image_path, cv2.COLOR_BGR2GRAY)
        if force_gray and len(img.shape) is 3:
            try:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img_gray
            except cv2.error:
                # happens when converting gray images to gray
                # as they don't have three dimension
                pass
        return img


    def get_rectangle(self) -> np.array:
        # could have done as mid point of a diagonal
        # but still, things are not much accurate
        center_of_rect = (self.vertices[0] + self.vertices[1] + self.vertices[2] + self.vertices[3]) * 0.25
        width = euclidean_distance(self.vertices[0], self.vertices[1])
        height = euclidean_distance(self.vertices[1], self.vertices[2])
        inclination = get_rectangle_inclination(self.vertices)

        # return np.array(
        #     [center_of_rect.x, center_of_rect.y, width, height, inclination]
        # )
        return np.array(
             [self.vertices[0].x, self.vertices[0].y, width, height, inclination]
        )

    @property
    def X(self):
        return self.get_image()

    @property
    def Y(self):
        return self.get_rectangle()


def get_rectangle_vertices(Y :tuple) -> tuple:
    x1, y1, w, h, inclination = Y
    radian = degree_to_radian(inclination)
    sin_theta, cos_theta = math.sin(radian), math.cos(radian)
    x2, y2 = (x1 + w*cos_theta), (y1 + w*sin_theta)
    x4, y4 = (x1 + h*sin_theta), (y1 + h*cos_theta)
    x3, y3 = (x4 + w*cos_theta), (y4 + w*sin_theta)

    return ((x1, y1), (x2, y2), (x3, y3), (x4, y4))


def euclidean_distance(pt_1 :Point, pt_2 :Point) -> float:
    square = lambda n: (n) * (n)
    return math.sqrt(square(pt_1.x - pt_2.x) + square(pt_1.y - pt_2.y))


def degree_to_radian(degree :float) -> float:
    return degree * math.pi / 180


def get_rectangle_inclination(vertices :list) -> float:
    '''Vertices is a list of Point'''
    pt_1, pt_2 = vertices[0], vertices[1]
    angle = np.arctan2((pt_2.y - pt_1.y), ((pt_2.x - pt_1.x))) * 180 / math.pi
    return angle


def reorder_vertices(vertices :tuple) -> tuple:
    '''Reorders coordinates clockwise with top-left coordinate at first
    '''
    sorting_key = lambda X: euclidean_distance(X[0], X[1])
    diag_1, diag_2 = sorted(itertools.combinations(vertices, 2),
                            key=sorting_key, reverse=True)[:2]
    
    # we have two diagonals. They should be going down
    # first diagonal, top coordinate up
    diag_1 = sorted(diag_1, key=lambda pt: pt.y)
    # second diagonal, top coordinate up
    diag_2 = sorted(diag_2, key=lambda pt: pt.y)

    # get the left point first
    diag_1, diag_2 = (diag_1, diag_2) if diag_1[0].x < diag_2[0].x else (diag_2, diag_1)

    # just the coordinates are to be returned
    return diag_1[0], diag_2[0], diag_1[1], diag_2[1]


def read_cpos_file(filepath :str) -> list:
    rectangles = []
    vertices = [None, None, None, None]
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            i = i % 4
            val = line.split()
            x, y = float(val[0]), float(val[1])
            vertices[i] = Point(x, y)
            if (i+1) % 4 == 0:
                new_vertices = reorder_vertices(tuple(vertices))
                # converting to tuple makes a copy
                # list is mutable, tuple is not
                rectangles.append(new_vertices)

    return rectangles


def prepare_datapoints(data_raw_path='../../DataRaw', force_gray=True) -> list:
    if force_gray:
        image_files = glob.glob(data_raw_path + '/*/*gray.png')
    else:
        image_files = glob.glob(data_raw_path + '/*/*r.png')

    cpos_files = glob.glob(data_raw_path + '/*/*cpos.txt')
    
    print('There are', len(image_files), 'image files and', 
          len(cpos_files), 'cpos files')

    # convert relative paths to absolute paths
    image_files = map(os.path.abspath, image_files)
    cpos_files = map(os.path.abspath, cpos_files)

    datapoints = list()
    for image_file, cpos_file in zip(image_files, cpos_files):
        for rect in read_cpos_file(cpos_file):
            datapoints.append(DataPoint(image_file, rect))

    print('Prepared', len(datapoints), 'datapoints')

    return datapoints


def train_test_split_datapoints(datapoints :list, test_size=0.2) -> list:
    dp_copy = deepcopy(datapoints)
    random.shuffle(dp_copy)
    n_test = math.floor(len(dp_copy) * 0.2)
    n_train = len(dp_copy) - n_test
    train_datapoints = dp_copy[:n_train]
    test_datapoints = dp_copy[n_train:]
    return train_datapoints, test_datapoints



def main():
    datapoints = prepare_datapoints()
    somedp = datapoints[0]
    print('image', somedp.X, 'rectangle', somedp.Y, sep=os.linesep*2)


if __name__ == '__main__':
    main()
