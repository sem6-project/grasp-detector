# 2018 work 

## How this will work?
* Detectron helps identify the object.
* Another neural network identifies the rectangles.

## Difference from previous work
- Previous work had the rectangles already available. All they did was to train a neural network to tell if the rectangle is a good rectangle or bad rectangle.
- Our works is based on finding the rectangle.
- Batch the dataset rather than passing it completely at once.

## Input/Output description
**Input** > `Fixed size image in linear form`. Since, all images in `DataRaw` are of same dimension, this helps.
Later on, the images in `DataRaw` are to be replaced by output of `Detectron** for each image.

**Output**  > `1x4 vector` describing a rectangle `(length, breadth, width, height)`. (obtained from `*cpos.txt files`)

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
    
    
### What's to be done?
-[ ] Run this neural network as a proof of concept for this guessing game.
-[ ] Replace this neural network by a convolutional network (refer to [CIFAR Keras example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py) for reference)
-[ ] Change this to persist the model on hard drive
-[ ] Prepare another dataset based on Detectron's output (which will be masked object padded to make fixed size inputs)
-[ ] Tune and optimize this network
