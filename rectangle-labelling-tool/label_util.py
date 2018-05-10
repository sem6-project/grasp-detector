import os
import sys
import json
import cv2
import argparse
import numpy as np
from enum import Enum

import utils


RECT_COLOR = [
    (255, 0, 0),
    (66, 110, 26),
    (0, 0, 255)
]
SHALL_QUIT = False


class Cartographer(object):
    def __init__(self, datapoint, image_name :str) -> None:
        self.datapoint = datapoint
        self.image_name = image_name
        self._image = None
        self.rectangles = []
        self.current_rectangle = []

    @property
    def image(self):
        '''To ensure that loading an image is not compulsory and
           image is read only once'''
        if self._image is None:
            self._image = self.datapoint.get_image(gray=False)
        return self._image

    def cleanup(self) -> None:
        del self._image
        self._image = None

    def track_events(self, event, x, y, flag, param) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return

        self.current_rectangle.append((x, y))
        intent = len(self.rectangles) + 1
        color = RECT_COLOR[intent % len(RECT_COLOR)]
        print('\rIntent', intent, self.current_rectangle, end='')

        cv2.circle(self.image, (x, y), 2, color, -1)

        if len(self.current_rectangle) == 3:
            x1, y1 = self.current_rectangle[0]
            x2, y2 = self.current_rectangle[1]
            x3, y3 = self.current_rectangle[2]
            # x3, y3 = utils.align_rectangle_point(
            #     x1, y1, x2, y2, x, y
            # )
            self.current_rectangle[2] = (x3, y3)
            x4, y4 = int(x1+x3-x2), int(y1+y3-y2)
            self.current_rectangle.append((x4, y4))

            cv2.polylines(self.image, [np.array(self.current_rectangle, np.int32)], True, color, 2)

            self.rectangles.append(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            )
            self.current_rectangle = []
            print('\nIntent', intent, self.rectangles[-1], '\n')

        sys.stdout.flush()


    def label_image(self, rectangles=[]) -> bool:
        cv2.namedWindow(self.image_name)
        cv2.setMouseCallback(self.image_name, self.track_events)
        shall_quit = 0
        print('::', self.image_name)

        if rectangles:
            self.rectangles = rectangles
            for i, rect in enumerate(self.rectangles):
                color = RECT_COLOR[i % len(RECT_COLOR)]
                cv2.polylines(self.image, [np.array(rect, np.int32)], True, color, 2)
            print(' > Rectangles already exist for this image. Press n to proceed or c to clear and redraw.')

        interrupted = False
        while True:
            cv2.imshow(self.image_name, self.image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('n'):
                break

            if k == 27:
                interrupted = True
                break

            if k == ord('q'):
                if shall_quit == 1:
                    global SHALL_QUIT
                    SHALL_QUIT = True
                    break
                print('Press q again to quit')
                shall_quit = 1

            if k == ord('c'):
                self.rectangles = []
                self.current_rectangle = []
                self.cleanup()

        cv2.destroyAllWindows()
        self.cleanup()
        return not interrupted


def parse_args():
    parser = argparse.ArgumentParser(description='Tool to label rectangles in image')
    parser.add_argument(
        '--data-dir',
        dest='data_raw',
        help='Path to DataRaw',
        type=str
    )
    parser.add_argument(
        '--output',
        dest='output_file',
        default='mappings.json',
        help='Output file to save rectangles at',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    datapoints = utils.prepare_datapoints(args.data_raw)
    unique_datapoints = utils.filter_unique_datapoints(datapoints)
    cartographers = [Cartographer(dp, dp.image_name)
                     for dp in unique_datapoints]

    output_file = args.output_file
    mapping = utils.read_mappings(output_file)
    print()

    for c in cartographers:
        if c.image_name in mapping:
            ok = c.label_image(mapping[c.image_name])
        else:
            ok = c.label_image()
        if ok:
            mapping[c.image_name] = c.rectangles
        if SHALL_QUIT:
            break


    utils.save_mappings(mapping, output_file)
    print('Saved mappings to', output_file)


if __name__ == '__main__':
    main()
