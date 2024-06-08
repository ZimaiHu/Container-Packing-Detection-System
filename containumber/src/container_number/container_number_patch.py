# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/6/7 0:05
@File     : container_number_patch.py
@Project  : YiJia
@Introduce:
"""
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
from ultralytics import YOLO
from datetime import datetime

class ContainerNumberPatch:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        patch_list = []
        res = self.model.predict(img, show=False, save=False, verbose=False,conf=0.5)[0]
        conf_list = res.boxes.conf.tolist()
        if conf_list:
            boxes = res.boxes.xyxy.tolist()
            sorted_boxes = sorted(boxes, key=lambda x: x[1])
            for box in sorted_boxes:
                box = [int(x) for x in box]
                x1, y1, x2, y2 = box
                patch = img[y1:y2, x1:x2]
                patch_list.append(patch)
            return patch_list
        else:
            return None
