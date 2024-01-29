# CNN for cifar10

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)

# normalize the data: 0-255 -> 0,1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# model...
model = keras.models.Sequential()

model.add(layers.Conv2D(32, (3,3), strides=(1,1) , padding="valid", activation='relu', input_shape=(32,32,3)))
# pooling layer
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(32, 3, activation='relu'))
# pooling layer
model.add(layers.MaxPool2D((2,2)))

# flatten
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())
# import sys; sys.exit()


# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# training
batch_size = 64
epochs  =5

model.fit(train_images, train_labels, epochs=epochs,
          batch_size = batch_size, verbose=2)

#evaluate
model.evaluate(test_images, test_labels, batch_size = batch_size, verbose=2)

# save the whole model
# two options. SavedModel, HDF5
model.save("conv_neural_net.keras")

# to load the model
new_model = keras.models.load_model("conv_neural_net.keras")
# do as u would from there
