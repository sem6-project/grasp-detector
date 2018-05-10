# Test103

### Changes made from `Test102`
- [x] Prepare labelling tool to create sample intent dataset.
- [x] Train on custom dataset with intent.
- [x] Perform corrections on misaligned rectangles in intent dataset.



### Labelling tool
Asks you to draw rectangles on each and every image in *DataRaw* and saves the mapping in a `json** file.

* Asks 3 points for a given rectangle.
* Can draw at most 3 rectangles on a given image.

**Dependency**:
Python3, OpenCV, Numpy
```bash
pip3 install opencv-python numpy --user
```

**Usage**:
```bash
python3 label_util.py --data-dir ../../DataRaw --output mapping.json
```

**Commands allowed**:

These keystrokes can be pressed while working on any image.

| Keystroke | Action                                            |
|-----------|---------------------------------------------------|
| Esc       | Move to next image (discarding rectangles if any) |
| n         | Move to next image (rectangles are ok)            |
| c         | Clear rectangles on an image                      |
| q (twice) | Close GUI and save results                        |

**Note**: If you have already drawn rectangles for a given image, they will be shown if you view the image. You can either clear the image for new rectangles to be drawn or press 'n' to go to next image.
This is a change from previous functioning where the mappings were merged.
