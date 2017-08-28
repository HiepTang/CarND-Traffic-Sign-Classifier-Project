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
## Data preprocessing and augmentation
### Convert to grayscale images
According the [Traffic Sign Classification article of Pierre Sermanet and Yann LeCun ](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), coverting the images to grayscale worked well and it also helps reduce the training time.
```python
# Convert images to grayscale
def grayscale_images(images):
    return np.sum(images/3, axis=3, keepdims=True)

# Convert to grayscale images
X_train_gray = grayscale_images(X_train)
X_valid_gray = grayscale_images(X_valid)
```
### Normalize images
Normalizing images helps the traing process for each feature to have a similar range so that out gradient descent don't go out of control. The normalized images don't change the output of the image on traffic sign classifier as some normalized images showing as below.
```python
# Normalized images
def normalized_images(images):
    return (images - 128)/128

# Normalized images
X_train_normalized = normalized_images(X_train_gray)
X_valid_normalized = normalized_images(X_valid_gray)
```
### Augment the data
#### Translate image
Translating image with a small pixels (3 pixels) in x,y directions doesn't change the meaning of image.
```python
import cv2

# Translate a random up to about 3 pixels in x, y directions
def translate_image(img, px = 3):
    rows,cols,_ = img.shape
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    
    return dst

test_img = X_train_normalized[1523]
test_dsts = []
test_dsts.append(test_img)
for i in range(9):
    test_dsts.append(translate_image(test_img))

show_images(test_dsts, cols = 10, cmap='gray')
```
#### Scale image
Scaling image randomly from -2 to 2 doesn't change the meaning of traffic sign output.
```python
# Scale image randomly from -2 to 2
def scale_image(img, lower_limit=-2, upper_limit=2):   
    rows,cols,_ = img.shape
    # transform limits
    px = np.random.randint(lower_limit,upper_limit)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

test_scale = []
test_scale.append(test_img)
for i in range(9):
    test_scale.append(scale_image(test_img, -3, 3))

show_images(test_scale)
```
#### Warp image
The traffic signs can be viewed from different viewpoints without changing the meaning, so we can warp traffic signs images in order to increate the dataset.
```python
def warp_image(img):
    
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.09   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.09

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

test_warp = []
test_warp.append(test_img)
for i in range(9):
    test_warp.append(warp_image(test_img))
    
show_images(test_warp)
```
#### Brightness image
Random changing the brightness with small number can increate the dataset size and the variance of data.
```python
def brightness_image(img, limit = 1.0):
    shifted = img + limit  
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - limit
    return dst

test_brightness = []
test_brightness.append(test_img)

for i in range(9):
    test_brightness.append(brightness_image(test_img))
    
show_images(test_brightness)
```
#### Putting it together - augment image
The translate, scale, warp and brightness image functions work fine. Now, it's time to put everything together in order to implement the augment image function. The show images testing function shows the augmented images are good and don't change the meaning of the original image.
```python
def augment_image(img):
    return translate_image(scale_image(warp_image(brightness_image(img))))

test_augs = []
test_augs.append(test_img)
for i in range(9):
    test_augs.append(augment_image(test_img))

show_images(test_augs)
```
I have tested the augment image function by randomly one image per class. The result shows that the augmented are some different from the original image and not change the meaning of traffic sign of the original image.
```python
# Test augment one image for each class
sign_classes, class_indices, class_counts = np.unique(y_train, return_index = True, return_counts = True)

for c, c_index, c_count in zip(sign_classes, class_indices, class_counts):
    print("Class %i: %-*s" % (c, col_width, signnames[c]))
    fig = plt.figure(figsize = (18, 3))
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
    random_index = random.sample(range(c_index, c_index + c_count), 1)
    random_image = X_train_normalized[random_index[0]]
    
    for i in range(10):
        axis = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
        axis.imshow(augment_image(random_image).squeeze(), cmap='gray')
    plt.show()
    print("--------------------------------------------------------------------------------------\n")
```
I have tried to augment the training dataset by 15 times and keep the same unbalanced distribution of dataset. It was a very difficult to decide between make the traing dataset become balanced but not much data and augment the traing dataset as much as possible. I finally choose the maximum data augmentation option after comparing two approaches.
```python
def augment_mass_train_images(images, labels, n_aug=15):
    sign_classes, class_indices, class_counts = np.unique(labels, return_index = True, return_counts = True)
    images_aug = []
    labels_aug = []
    n_aug = 15
    n_augs = []
    for c, c_index, c_count in zip(sign_classes, class_indices, class_counts):
        print("Processing Class %i: %-*s  %s samples" % (c, col_width, signnames[c], str(c_count)))
        n_augs.append(n_aug)
        for i in range(c_index, c_index + c_count):
            img = images[i]
            y = labels[i]
            images_aug.append(img)
            labels_aug.append(y)
            for j in range(n_aug):
                new_img = augment_image(img)
                images_aug.append(new_img)
                labels_aug.append(y)
            if i%10 == 0:
                print('-', end='')
            if i%50 == 0:
                print('|', end='')
        
        print()
        print("After augumentation size:", (c_count + 1)*n_aug)
        print("--------------------------------------------------------------------------------------\n")
    return images_aug, labels_aug, n_augs
    
    X_train_final, y_train_final, n_augs = augment_mass_train_images(X_train_normalized, y_train, n_aug=15)
```
The distribution of augmented traing dataset does not change. The size of traing dataset after augmentation is 556784 images. It's bigger than the original traing dataset 15 times.
```python
show_histogram(y_train_final, n_classes)
```
![FinalDistribution](TrainFinalDistribution.png)


