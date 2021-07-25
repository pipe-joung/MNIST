import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Activation
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf_ver = tf.__version__
print("Tensroflow version: ", tf_ver)

from tensorflow import keras

kr_ver = keras.__version__
print("Keras version: ", kr_ver)


from keras.models import load_model
model = load_model('e:\dd\work\keras\mnist_test.h5')
print(model.summary())
model.summary()


