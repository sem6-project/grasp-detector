import os
import sys
from datetime import datetime
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
    def __init__(self, datapoint :utils.DataPoint, image_name :str) -> None:
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


    def label_image(self) -> bool:
        cv2.namedWindow(self.image_name)
        cv2.setMouseCallback(self.image_name, self.track_events)
        shall_quit = 0
        print('::', self.image_name)

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


def save_mappings(mapping :dict, output_file :str) -> None:
    content = {
        'timestamp': datetime.now().isoformat(),
        'mapping': mapping
    }
    with open(output_file, 'w') as f:
        f.write(json.dumps(content, indent=2))


def read_mappings(filepath :str) -> dict:
    with open(filepath) as f:
        content = json.loads(f.read())
    return content['mapping']


def merge_mappings(original_mapping :dict, new_mapping :dict) -> dict:
    mapping = {}
    for k, v in original_mapping.items():
        mapping[k] = v
    for k, v in new_mapping.items():
        mapping[k] = v
    return mapping


def main():
    args = parse_args()
    datapoints = utils.prepare_datapoints(args.data_raw)
    unique_datapoints = utils.filter_unique_datapoints(datapoints)
    cartographers = [Cartographer(dp, dp.image_name)
                     for dp in unique_datapoints]

    print()
    mapping = {}

    for c in cartographers:
        ok = c.label_image()
        if ok:
            mapping[c.image_name] = c.rectangles
        if SHALL_QUIT:
            break

    output_file = args.output_file
    if os.path.exists(args.output_file):
        print('File already exists', output_file)
        print('k : Quit')
        print('o : Overwrite')
        print('m : Merge')
        print('n : Save in new file (default)')
        whattodo = input('What to do? ').strip().lower()
        if whattodo == 'o':
            pass
        if whattodo == 'k':
            sys.exit(1)
        elif whattodo == 'm':
            original_mapping = read_mappings(output_file)
            mapping = merge_mappings(original_mapping, mapping)
        else:
            dirpath = os.path.dirname(output_file)
            filename = os.path.basename(output_file)
            filename, ext = os.path.splitext(filename)
            fileid = datetime.strftime(datetime.now(), '%d-%m-%H-%M-%S')
            output_file = os.path.join(dirpath, filename+'-'+fileid+ext)

    save_mappings(mapping, output_file)
    print('Saved mappings to', output_file)


if __name__ == '__main__':
    main()
