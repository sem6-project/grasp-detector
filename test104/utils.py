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
# from sympy.geometry import Point, Line
import sympy.geometry as SG
from shapely.geometry import Polygon as ShapelyPolygon


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

    def __str__(self) -> str:
        return str(self.to_tuple())

    def __repr__(self) -> str:
        return self.__str__()


class DataPoint(object):
    def __init__(self, rgb_image_path :str, gray_image_path :str, vertices :list):
        '''Parameters:
            rgb_image_path : str : path to the image
            gray_image_path : str : path to the image
            vertices: str : list of :Point:
        '''
        self.rgb_image_path = rgb_image_path
        self.gray_image_path = gray_image_path
        self.image_path = self.gray_image_path
        self.vertices = vertices
        self._rgb_image = None
        self._gray_image = None
        self._image = None

    @property
    def image_name(self) -> str:
        filename = os.path.basename(self.image_path)
        name, _ = os.path.splitext(filename)
        return name.replace('gray', '')

    def get_image(self, gray=True, background=None) -> np.array:
        if self._image is not None:
            return self._image
        if gray:
            image = cv2.imread(self.gray_image_path, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.imread(self.rgb_image_path)

        if background is not None:
            bg = deepcopy(background)
            if gray:
                bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
            # # background removal, but leaves salt and pepper noise
            # image = (image.astype('float') - bg.astype('float')).astype('uint8')
            # # 2 passes of median filter to remove salt and pepper noise
            # image = cv2.medianBlur(image, 7)
            # image = cv2.medianBlur(image, 7)

            # absdiff is better than normal subtraction.. awesome
            image = cv2.absdiff(image, bg)

        self._image = image
        return self._image


    def get_rectangle(self) -> np.array:
        # could have done as mid point of a diagonal
        # but still, things are not much accurate
        # center_of_rect = (self.vertices[0] + self.vertices[1] + self.vertices[2] + self.vertices[3]) * 0.25
        width = euclidean_distance(self.vertices[0], self.vertices[1])
        height = euclidean_distance(self.vertices[1], self.vertices[2])
        inclination = get_rectangle_inclination(self.vertices)

        # return np.array(
        #     [center_of_rect.x, center_of_rect.y, width, height, inclination]
        # )
        # return np.array(
        #      [self.vertices[0].x, self.vertices[0].y, width, height, inclination]
        # )
        return np.array(
            [(vertex.x, vertex.y) for vertex in self.vertices[:3]]
        ).reshape(-1)

    @property
    def X(self):
        return self.get_image(gray=True)

    @property
    def Y(self):
        return self.get_rectangle()

    def visualize_result(self, prediction :tuple, target_file :str,
                         gray=True,
                         actual_rect_color=(0, 0, 255),
                         predicted_rect_color=(0, 255, 0),
                         rect_thickness=2) -> None:
        '''Visualize the actual and predicted rectangle on RGB image
        '''
        # image = self.get_image(gray=False)
        image = self.get_image(gray=gray)
        predicted_rect = np.array(get_rectangle_vertices(prediction), np.int32)
        actual_rect = np.array(get_rectangle_vertices(self.Y), np.int32)

        cv2.polylines(image, [predicted_rect], True, predicted_rect_color, rect_thickness)
        cv2.polylines(image, [actual_rect], True, actual_rect_color, rect_thickness)

        cv2.imwrite(target_file, image)


    def __hash__(self):
        return hash(self.image_name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.image_name == other.image_name


def get_rectangle_vertices(Y :tuple) -> tuple:
    # ------------------------------------------------------
    # first solution : use 1x5 vector : trigonometry
    # ------------------------------------------------------
    # x1, y1, w, h, theta = Y
    # radian = degree_to_radian(inclination)
    # sin_theta, cos_theta = math.sin(radian), math.cos(radian)
    # x2, y2 = (x1 + w*cos_theta), (y1 + w*sin_theta)
    # x4, y4 = (x1 + h*sin_theta), (y1 + h*cos_theta)
    # x3, y3 = (x4 + w*cos_theta), (y4 + w*sin_theta)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # second solution : use 1x5 vector - trignometry changed
    # ------------------------------------------------------
    # x1, y1, w, h, theta = Y
    # angle_1 = degree_to_radian(theta)
    # angle_2 = angle_1 + math.pi*0.75

    # x2, y2 = (x1 + w * math.cos(angle_1)), (y1 + w * math.sin(angle_1))
    # x4, y4 = (x1 + h * math.cos(angle_2)), (y1 + w * math.sin(angle_2))
    # x3, y3 = (x2 + x4 - x1), (y2 + y4 - y1)
    #
    # ------------------------------------------------------

    # ------------------------------------------------------
    # third solution : use three points
    # ------------------------------------------------------
    x1, y1, x2, y2, x3, y3 = Y
    x4, y4 = (x1 + x3 - x2), (y1 + y3 - y2)
    # ------------------------------------------------------

    return ((x1, y1), (x2, y2), (x3, y3), (x4, y4))


def align_rectangle_point(x1, y1, x2, y2, x3, y3) -> tuple:
    '''Re-aligns the third point among three points to form a rectangle'''
    line1 = SG.Line(SG.Point(x1,y1), SG.Point(x2,y2))
    line2 = line1.perpendicular_line(SG.Point(x2,y2))
    line3 = line2.perpendicular_line(SG.Point(x3,y3))
    fixed_point = line2.intersection(line3)[0]
    return int(fixed_point.x), int(fixed_point.y)


def IOU(actual :tuple, predicted :tuple) -> float:
    '''Calculates overlap b/w two rectangles
    Parameters:
    actual and predicted are tuples of length 6
    (x1, y1, x2, y2, x3, y3)
    '''
    r_actual = get_rectangle_vertices(actual)
    r_predicted = get_rectangle_vertices(predicted)
    poly_actual = ShapelyPolygon(r_actual)
    poly_predicted = ShapelyPolygon(r_predicted)

    area_intersection = poly_predicted.intersection(poly_actual).area
    area_union = poly_predicted.union(poly_actual).area
    return area_intersection / area_union


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


def get_image_name(fullpath :str):
    filename = os.path.basename(fullpath)
    name, _ = os.path.splitext(filename)
    return name.replace('gray', '').replace('r', '')


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


def prepare_datapoints(data_raw_path='../../DataRaw', unique=False) -> list:
    gray_image_files = glob.glob(data_raw_path + '/*/*gray.png')
    rgb_image_files = glob.glob(data_raw_path + '/*/*r.png')

    cpos_files = glob.glob(data_raw_path + '/*/*cpos.txt')

    print('There are', len(rgb_image_files), 'image files and',
          len(cpos_files), 'cpos files')

    # convert relative paths to absolute paths
    rgb_image_files = map(os.path.abspath, rgb_image_files)
    gray_image_files = map(os.path.abspath, gray_image_files)
    cpos_files = map(os.path.abspath, cpos_files)

    datapoints = list()
    for rgb_img_file, gray_img_file, cpos_file in zip(rgb_image_files, gray_image_files, cpos_files):
        for rect in read_cpos_file(cpos_file):
            datapoints.append(DataPoint(rgb_img_file, gray_img_file, rect))
            if unique:
                break

    print('Prepared', len(datapoints), 'datapoints')

    return datapoints


def get_background_mappings(bg_mapping_file='../../DataRaw/backgroundMapping.txt'):
    bg_mapping = {}
    with open(bg_mapping_file) as f:
        content = f.readlines()
        data = map(lambda x: x.strip().split(), content)
        for img_f, bg_f in data:
            bg_mapping[get_image_name(img_f)] = bg_f

    return bg_mapping


def read_backgrounds(bg_dir='../../DataRaw/backgrounds'):
    files = glob.glob(bg_dir + '/*.png')
    files = map(os.path.abspath, files)
    backgrounds = {}
    for f in files:
        backgrounds[os.path.basename(f)] = cv2.imread(f, cv2.COLOR_BGR2GRAY)

    print('Total', len(backgrounds), 'background files')
    return backgrounds


def filter_unique_datapoints(datapoints :list, sort=True) -> list:
    '''Get only one entry for a given image'''
    if not sort:
        list(set(datapoints))
    return sorted(set(datapoints), key=lambda dp: dp.image_name)


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
