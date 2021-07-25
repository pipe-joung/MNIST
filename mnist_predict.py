# 저장된 MNIST모델에 이미지를 입력하여 판독

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax
import PIL.Image as pilimg


# 2. 모델 불러오기
from keras.models import load_model
model = load_model('e:\dd\work\MNIST\mnist.h5')

# 3. 모델 사용하기
num_img = pilimg.open('e:\dd\work\MNIST\s1.png')
num_img = num_img.resize((28, 28))
num_img = num_img.convert('L')
pix = np.array(num_img)

model_predict2 = model.predict(pix.reshape(1, 28, 28))
# print(model_predict2)
print(np.argmax(model_predict2))
