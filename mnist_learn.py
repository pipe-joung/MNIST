import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train /255
x_test = x_test /255

model = keras.models.Sequential([keras.layers.Flatten(input_shape =(28, 28)), 
keras.layers.Dense(256, activation='relu'), 
keras.layers.Dense(10, activation='softmax')])

# model = keras.layers.Dense(50, activation='relu')

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# print(model.summary())

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

model.save("e:\dd\work\MNIST\mnist.h5")


model_predict = model.predict(x_train[120].reshape(1, 28, 28))
# print(model_predict)
print(np.argmax(model_predict))
model.summary()



