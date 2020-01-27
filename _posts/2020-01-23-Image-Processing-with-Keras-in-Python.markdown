---
layout: post
title: Image Processing With Neural Networks using Keras
date: 2020-01-23 00:00:00 +0300
description: In this blog, I will discuss Image Processing With Neural Networks using Keras. (optional)
img: Keras-Basic.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [ZUBI_ASH, Keras_Basic, DATACAMP, Keras, CNN, convolutional_neural_network] # add tag
---

In this blog, I will discuss Image Processing with Neural Networks using Keras.

##  Keras_Basic 

### Images as data: visualizations

To display image data, you will rely on Python's Matplotlib library, and specifically use matplotlib's pyplot sub-module, that contains many plotting commands. Some of these commands allow you to display the content of images stored in arrays.

{% highlight ruby %}
#=> Import matplotlib
import matplotlib.pyplot as plt

#=> Load the image
data = plt.imread('bricks.png')

#=> Display the image
plt.imshow(data)
plt.show()
{% endhighlight %}

{: .center}
![GOT]({{site.baseurl}}/assets/img/k_img_1.JPG)

### Images as data: changing images

To modify an image, you can modify the existing numbers in the array. In a color image, you can change the values in one of the color channels without affecting the other colors, by indexing on the last dimension of the array.


{% highlight ruby %}
#=> Set the red channel in this part of the image to 1
data[0:10, 0:10, 0] = 1

#=> Set the green channel in this part of the image to 0
data[0:10, 0:10, 2] = 0

#=> Set the blue channel in this part of the image to 0
data[0:10, 0:10, 1] = 0

#=> Visualize the result
plt.imshow(data)
plt.show()
{% endhighlight %}

{: .center}
![GOT]({{site.baseurl}}/assets/img/k_img_2.JPG)


### Create An One-hot Encoding

Neural networks expect the labels of classes in a dataset to be organized in a one-hot encoded manner: each row in the array contains zeros in all columns, except the column corresponding to a unique label, which is set to 1.

{% highlight ruby %}
#=> The number of image categories
n_categories = 3

#=> The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

#=> Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

#=> Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1
{% endhighlight %}


### Build a neural network
We will use the Keras library to create neural networks and to train these neural networks to classify images. These models will all be of the Sequential type, meaning that the outputs of one layer are provided as inputs only to the next layer.

Now, we will create a neural network with Dense layers, meaning that each unit in each layer is connected to all of the units in the previous layer. For example, each unit in the first layer is connected to all of the pixels in the input images. The Dense layer object receives as arguments the number of units in that layer, and the activation function for the units. For the first layer in the network, it also receives an input_shape keyword argument.


{% highlight ruby %}
#=> Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense

#=> Initializes a sequential model
model = Sequential()

#=> The first layer receives images as input, has 10 units and 'relu' activation
model.add(Dense(10, activation='relu', input_shape=(784,)))

#=> The second input layer has 10 units and 'relu' activation.
model.add(Dense(10, activation='relu'))

#=> The output layer has one unit for each category (3 categories) and 'softmax' activation.
model.add(Dense(3, activation='softmax'))

#=> Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

#=> Reshape the data to two-dimensional array
train_data = train_data.reshape((50, 784))

#=> Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

#=> Reshape the data to two-dimensional array
train_data = train_data.reshape((50, 784))

#=> Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

{% endhighlight %}


Train on 40 samples, validate on 10 samples

Epoch 1/3

`32/40 [==========>......] - ETA: 0s - loss: 1.0043 - acc: 0.5000`

`40/40 [==== ============] - 0s 7ms/step - loss: 1.0061 - acc: 0.5000 - val_loss: 0.9917 - val_acc: 0.4000`

Epoch 2/3

`32/40 [=============>...] - ETA: 0s - loss: 0.9580 - acc: 0.5625`

`40/40 [=================] - 0s 150us/step - loss: 0.9447 - acc: 0.6000 - val_loss: 0.9603 - val_acc: 0.4000`

Epoch 3/3

`32/40 [==============>..] - ETA: 0s - loss: 0.8779 - acc: 0.5625`

`40/40 [=================] - 0s 140us/step - loss: 0.8957 - acc: 0.5500 - val_loss: 0.9234 - val_acc: 0.4000`


Cross-validation for neural network evaluation

To evaluate the model, we use a separate test data-set. As in the train data, the images in the test data also need to be reshaped before they can be provided to the fully-connected network because the network expects one column per pixel in the input.


{% highlight ruby %}
#=> Reshape test data
test_data = test_data.reshape((10, 784))

#=> Evaluate the model
model.evaluate(test_data, test_labels)
{% endhighlight %}

`10/10 [===========================] - 0s 179us/step`

`Out[2]: [0.9184357523918152, 0.6000000238418579]`


### One dimensional convolutions

A convolution of an one-dimensional array with a kernel comprises of taking the kernel, sliding it along the array, multiplying it with the items in the array that overlap with the kernel in that location and summing this product.

{% highlight ruby %}
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#=> Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()

#=> Print conv
print(conv)
{% endhighlight %}

### Image convolutions

The convolution of an image with a kernel summarizes a part of the image as the sum of the multiplication of that part of the image with the kernel. Now, you will write the code that executes a convolution of an image with a kernel using Numpy. Given a black and white image that is stored in the variable im, write the operations inside the loop that would execute the convolution with the provided kernel.

{% highlight ruby %}
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)

#=> Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()

#=> Print result
print(result)
{% endhighlight %}

### Defining image convolution kernels

This code is now stored in a function called convolution() that takes two inputs: image and kernel and produces the convolved image. Now, you will be asked to define the kernel that finds a particular feature in the image.

For example, the following kernel finds a vertical line in images:

np.array([[-1, 1, -1], 
          [-1, 1, -1], 
          [-1, 1, -1]])


Define a kernel that finds a dark spot surrounded by bright pixels.(A horizontal line has dark pixels at the top, bright pixels in the middle, and then dark pixels in the bottom)


kernel = np.array([[-1, -1, -1], 
                   [1, 1, 1],
                   [-1, -1, -1]])          
 
 

Define a kernel that finds a light spot surrounded by dark pixels. (A light spot has a bright pixel (with larger values, e.g., 1) in the center, surrounded by pixels that are dark (lower values, e.g., -1))
 
 
kernel = np.array([[-1, -1, -1], 
                   [-1, 1, -1],
                   [-1, -1, -1]])        
                   
Define a kernel that finds a dark spot surrounded by bright pixels.(This is the exact opposite of a bright pixel surrounded by dark pixels.)                   


kernel = np.array([[1, 1, 1], 
                   [1, -1, 1],
                   [1, 1, 1]])    
