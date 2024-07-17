# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/18 1:47
@File     : config.py
@Project  : YiJia_BE
@Introduce:
"""
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime

class ContainerNumberRegion:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = YOLO(model_path)
        self.warmup()
    def warmup(self):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(_dummy_image, verbose=False)
    def detect(self, img):
        res = self.model.predict(img, show=False, save=False, verbose=False)[0]
        conf_list = res.boxes.conf.tolist()
        # conf_list = res.obb.conf.tolist()
        if conf_list:
            max_number = max(conf_list)
            max_index = conf_list.index(max_number)
            number_region = res[max_index].boxes.xyxy.tolist()[0]
            # number_region = res[max_index].obb.xyxy.tolist()[0]
            number_region = [int(x) for x in number_region]
            x1, y1, x2, y2 = number_region
            number_region_img = img[y1:y2, x1:x2]
            return (x1, y1, x2, y2), number_region_img
        else:
            return None, None

