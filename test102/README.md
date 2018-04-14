# This works

## Labelling tool
Asks you to draw rectangles on each and every image in *DataRaw* and saves the mapping in a `json** file.

* Asks 3 points for a given rectangle.
* Can draw at most 3 rectangles on a given image.

**Dependency**:
Python3, OpenCV, Numpy
```bash
pip3 install opencv-python numpy
```

**Usage**:
```python3
python3 label_util.py --data-dir ../../DataRaw --output mapping.json
```

**Commands allowed**:

These keystrokes can be pressed while working on any image.

| Keystroke | Action                                            |
|-----------|---------------------------------------------------|
| Esc       | Move to next image (discarding rectangles if any) |
| n         | Move to next image (rectangles are ok) |
| c         | Clear rectangles on an image                      |
| q (twice) | Close GUI and save results          |


**To handle duplicate files while saving results**:

| Keystroke | Action    | What happens ?                       |
|-----------|-----------|--------------------------------------|
| o         | Overwrite | Replace old mappings with new ones   |
| m         | Merge     | Merge both files (new over old ones) |
| n         | New       | Save to a new file (automatically)   |
| k         | Quit      | Quit without saving                  |
