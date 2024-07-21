import cv2
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
from Lane_detector import LaneDetector

#sys.path.append('../../code')

img = cv2.imread('../image/sample.png')[:, :, (2, 1, 0)]

modelPath="../model/fastai_model.pth"
ld = LaneDetector(model_path = modelPath)

background_porb, left_prob, right_prob = ld.detect(img)
img_with_detection = copy.copy(img)
img_with_detection[left_prob > 0.5, :] = [0, 0, 255] #blue
img_with_detection[right_prob > 0.5, :] = [255, 0, 0] #red
plt.imshow(img_with_detection)

v_list, u_list = np.nonzero(left_prob > 0.5)
poly_left = np.poly1d(np.polyfit(u_list, v_list, deg=1))

v_list, u_list = np.nonzero(right_prob > 0.5)
poly_right = np.poly1d(np.polyfit(u_list, v_list, deg=1))

u = np.arange(0, 1024)
v_left = poly_left(u)
v_right = poly_right(u)
u_i, v_i = ld.get_intersection(poly_left, poly_right)
plt.plot(u, v_left, color="b")
plt.plot(u, v_right, color="r")
plt.xlim(0, 1024); plt.ylim(512, 0)
plt.xlabel('$u$'); plt.ylabel('$v$');
plt.scatter([u_i], [v_i], marker="o", s=100, color="y", zorder=10)

pitch, yaw = ld.get_py_from_vp(u_i, v_i, ld.cg.intrinsic_matrix)
print(f'yaw degree:   %.2f' % np.rad2deg(yaw))
print(f'pitch degree: %.2f' % np.rad2deg(pitch))
plt.show()
