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

{% endhighlight %}
