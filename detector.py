# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/16 20:13
@File     : detector.py
@Project  : YiJia_BE
@Introduce:
"""

from src.detector_all import *
from weights.model_path_all import *
from utils import image_to_numpy

import os
# base_path = os.path.dirname(os.path.abspath(__file__))

class Detector:
    def __init__(self):
        self.detectors = {
            "container_number": (ContainerNumberDetector(), container_number_path),
            "cargo_label": (CargoLabelDetector(), cargo_label_path),
            "cargo_pallet_corner": (PalletDetector(), cargo_pallet_corner_path),
            "cargo_strap": (StrapDetector(), cargo_strap_path),
            "cargo_seal": (SealDetector(), cargo_seal_path),
            "cargo_chaituo": (ChaituoDetector(), cargo_chaituo_path),
            "cargo_qianhou": (QianhouDetector(), cargo_qianhou_path),
            "cargo_zhedang": (ZheDangDetector(), cargo_zhedang_path),
            "cargo_fods": (FodsDetector(), cargo_fods_path),
            # 添加其他检测器及其路径
            # "another_detector": (AnotherDetector(), another_path),
        }

        self.task_map = {
            "1": self.detect_cargo_label_det,
            "2": self.detect_pallet_corner_det,
            "3": self.detect_strap_det,
            "4": self.detect_seal_det,
            "5": self.detect_container_det,
            "6": self.detect_chaituo_det,
            "7": self.detect_fods_det,
            "8": self.detect_qianhou_det,
            "9": self.detect_zhedang_det
        }

        # 初始化检测器模型
        self._load_detector_models()

    def _load_detector_models(self):
        for detector, path in self.detectors.values():
            detector.load_model(path)

    def detect_all(self, img, task):
        if task in self.task_map:
            return self.task_map[task](img)
        else:
            raise ValueError("Invalid task number")
    def detect_cargo_label_det(self, img):
        return self.detectors["cargo_label"][0].detect_cargo_label(img)

    def detect_zhedang_det(self,img):
        return self.detectors["cargo_zhedang"][0].detect_zhedang_label(img)

    def detect_pallet_corner_det(self, img):
        return self.detectors["cargo_pallet_corner"][0].detect_pallet(img)

    def detect_strap_det(self, img):
        return self.detectors["cargo_strap"][0].detect_strap(img)

    def detect_seal_det(self, img):
        return self.detectors["cargo_seal"][0].detect_seal(img)

    def detect_container_det(self, img):
        img = image_to_numpy(img)
        return self.detectors["container_number"][0].detect(img)

    def detect_chaituo_det(self, img):
        return self.detectors["cargo_chaituo"][0].detect_chaituo(img)

    def detect_fods_det(self, img):
        return self.detectors["cargo_fods"][0].detect_fods(img)

    def detect_qianhou_det(self, img):
        return self.detectors["cargo_qianhou"][0].detect_qianhou(img)