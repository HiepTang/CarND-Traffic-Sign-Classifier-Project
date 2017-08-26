# Traffic Sign Classifier
## Overview
In this project, I used a convolutional neural network with Tensorflow in order to solve the traffic sign classifier with German traffic signs dataset.
Source code link here
## Dataset summary
I used the following code to get some basic information of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
```python
import numpy as np
SEED = 1501
np.random.seed(SEED)
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples=", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```
The dataset consists of 34799 32x32 RGB training images, 4410 validation images and 12630 testing images. There are 43 number of output classes.
## Exploratory visualization of the dataset
### Explore the distribution of the training and validation dataset
I define a function to show the histogram of the dataset.
```python
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

# Show the histogram of label frequency            
def show_histogram(y, n_classes):
    # histogram of label frequency
    hist, bins = np.histogram(y, bins=n_classes)
    center = (bins[:-1] + bins[1:]) / 2
    plt.title("Distribution of dataset")
    plt.xlabel("Class number")
    plt.ylabel("No image")
    plt.bar(center, hist)
    plt.show()
```
#### The distribution of training dataset
```python
show_histogram(y_train, n_classes)
```
The training distribution image HERE
The training dataset is very unbalanced, some classes have only about 180 images and some classes have over 1800 images.
#### The distribution of validation dataset
```python
show_histogram(y_valid, n_classes)
```
The validation distribution image HERE
The validation dataset is also very unbalanced. However, it maintains the ratio between training and validation dataset well on each class.
#### Show some sample training images
The show sample images function
```python
import random
from pandas.io.parsers import read_csv

random.seed(SEED)

signnames = read_csv("signnames.csv").values[:, 1]
col_width = max(len(name) for name in signnames)

# Function - Show 10 random sample images for each class
def show_sample_images(labels, images, cmap=None):
    sign_classes, class_indices, class_counts = np.unique(labels, return_index = True, return_counts = True)

    for c, c_index, c_count in zip(sign_classes, class_indices, class_counts):
        print("Class %i: %-*s  %s samples" % (c, col_width, signnames[c], str(c_count)))
        fig = plt.figure(figsize = (18, 3))
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
        random_indices = random.sample(range(c_index, c_index + c_count), 10)
        for i in range(10):
            axis = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
            if cmap == 'gray':
                axis.imshow(images[random_indices[i]].squeeze(), cmap='gray')
            else:
                axis.imshow(images[random_indices[i]])
        plt.show()
        print("--------------------------------------------------------------------------------------\n")
```
Let's show some sample images
```python
show_sample_images(y_train, X_train)
```
Please refer to the source code HERE to see the result of this function. The images are significant different in term of contrast and brightness.


