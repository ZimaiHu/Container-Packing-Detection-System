# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/16 20:13
@File     : detector.py
@Project  : YiJia_BE
@Introduce:
"""
import os
from containumber.utils import image_to_numpy
from containumber.src.container_number.container_number_detector import ContainerNumberDetector
base_path = os.path.dirname(os.path.abspath(__file__))
class Manager:
    def __init__(self):
        self.detectors = {
            "container_number_detector": ContainerNumberDetector(),
            # 添加其他检测器
            # "another_detector": AnotherDetector(),
        }

        # 初始化检测器模型
        self._load_detector_models()

    def _load_detector_models(self):
        for _, detector in self.detectors.items():
            detector.load_model(base_path)

    # 检测器：货柜号
    def detect_container_number(self, img):
        img = image_to_numpy(img)
        txt = self.detectors["container_number_detector"].detect(img)
        return txt