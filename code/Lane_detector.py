import numpy as np
import cv2
import torch
from Camera_geometry import CameraGeometry

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