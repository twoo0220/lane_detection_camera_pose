import cv2
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
from Lane_detector import LaneDetector

#sys.path.append('../../code')

img = cv2.imread('../image/test.jpg')[:, :, (2, 1, 0)]
plt.imshow(img)
plt.show()

modelPath="../model/fastai_model.pth"
ld = LaneDetector(model_path = modelPath)

## 24.07.20
## todo : torch install, ModuleNotFoundError: No module named 'torch'