import numpy as np
import cv2
import torch
from Camera_geometry import CameraGeometry
from fastseg import MobileV3Small

class LaneDetector():
    def __init__(self, model_path, cam_geom=CameraGeometry()):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = torch.load(model_path).to(self.device)
        else:
            self.model = torch.load(model_path, map_location = torch.device("cpu"))
            self.device = "cpu"
        self.model.eval()

    def _predict(self, img):
        with torch.no_grad():
            image_tensor = img.transpose(2, 0, 1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
            model_output = torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output

    def detect(self, img_array):
        model_output = self._predict(img_array)
        background, left, right = model_output[0, 0, :, :], model_output[0, 1, :, :], model_output[0, 2, :, :]
        return background, left, right
    
    def get_intersection(self, line1, line2):
        m1, c1 = line1
        m2, c2 = line2
        if m1 == m2:
            return None
        u_i = (c2 - c1) / (m1 - m2)
        v_i = (m1 * u_i) + c1
        return u_i, v_i
    
    def get_py_from_vp(self, u_i, v_i, K):
        p_infinity = np.array([u_i, v_i, 1])
        K_inv = np.linalg.inv(K)
        r3 = K_inv @ p_infinity
        r3 /=np.linalg.norm(r3)
        pitch = np.arcsin(r3[1])
        yaw = -np.arctan2(r3[0], r3[2])
        
        return pitch, yaw
