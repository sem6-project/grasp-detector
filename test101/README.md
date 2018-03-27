# 2018 work - Not a SCAM

## How this will work?
* Detectron helps identify the object.
* Another neural network identifies the rectangles.

## Difference from previous work
Previous work had the rectangles already available. All they did was to train a neural network to tell if the rectangle is a good rectangle or bad rectangle.

Our works is based on finding the rectangle.

## Plan of action

### Test 101
Based on https://towardsdatascience.com/object-detection-with-neural-networks-a4e2c46b4491
and https://github.com/jrieke/shape-detection/blob/master/multiple-rectangles.ipynb

* Prepare a feature-file containing an array of objects. Each object has four attributes:
    - Image path    (string)              ---|
    - Object name   (string)              ---}- X
    - Intent name   (string)              ---|
    - Rectangle     (X, Y, Width, Height) ---}- Y

    This test will have image path and rectangle only.
    If there are mutiple rectangles for an image, they will be kept as different entries in this test.