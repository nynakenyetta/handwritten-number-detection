import os
import cv2 #for computer vision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #x is pixel data (image itself) y is the classification (number or digit)

#We want to normalize the pixels
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #we are flattening a certain input shape
model.add(tf.keras.layers.Dense(128, activation='relu')) #adding dense layer where each neuron is connected to another neuron layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #This represents the 10 digit neurons we'll have

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Now we need to fit the model (train it)
model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')