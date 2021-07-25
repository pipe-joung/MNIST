import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt

num_img = pilimg.open('e:\dd\work\MNIST\s3.png')
num_img = num_img.resize((28, 28))
num_img = num_img.convert('L')
pix = np.array(num_img)

plt.imshow(pix, cmap='gray')
plt.show()
