import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

img = cv2.imread('../image/test.jpg')[:, :, (2, 1, 0)]
plt.imshow(img)
plt.show()

