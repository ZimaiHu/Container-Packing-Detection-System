# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 13:03
@File     : text_image_v2h.py
@Project  : LLM
@Introduce:
"""
import cv2
import numpy as np

from containumber.src.container_number.container_number_region import ContainerNumberRegion
from containumber.src.container_number.container_number_ocr import ContainerNumberOCR
from containumber.src.container_number.container_number_patch import ContainerNumberPatch
from containumber.src.config import patch_model_path,region_model_path,ocr_model_path
import os


class ContainerNumberDetector:
    def __init__(self):
        self.result_dict ={"coordinates": {"xmin": None, "ymin": None, "xmax": None, "ymax": None, "ocr_text": None}}
        self.region_detector = ContainerNumberRegion()  # 横向货柜号/纵向货柜号 区域检测
        self.patch_detector = ContainerNumberPatch()  # 单个货号检测
        self.ocr_detector = ContainerNumberOCR()  # ocr识别


    def load_model(self, base_path):
        self.region_detector.load_model(os.path.join(base_path, region_model_path))
        self.patch_detector.load_model( [os.path.join(base_path, patch_path) for patch_path in patch_model_path])
        self.ocr_detector.load_model(os.path.join(base_path, ocr_model_path))

    def detect(self, img):
        # step1:检测货柜号区域
        coordinates, number_region = self.region_detector.detect(img)
        if number_region is None:
            return {"coordinates": None,"ocr_text": None}
        self.record_coordinates(coordinates)

        # step2:调整图像方向（应对竖排）
        adjusted_img, adjusted_img_list = self.patch_detector.detect(number_region)
        if adjusted_img is None:
            return {"coordinates": None,"ocr_text": None}

        # step3:区域级别的ocr识别
        region_ocr_result_src = self._region_level_ocr(adjusted_img)
        # 标准校验ocr结果
        if region_ocr_result_src:
            region_ocr_result_valid = self.ocr_detector.validate_container_id(region_ocr_result_src)
            if region_ocr_result_valid:
                self.record_ocr(region_ocr_result_valid)
                return self.result_dict

        # step4:patch级别的ocr识别
        patch_ocr_result_src = self._patch_level_ocr(adjusted_img_list)
        if patch_ocr_result_src:
            patch_ocr_result_valid = self.ocr_detector.validate_container_id(patch_ocr_result_src)
            if patch_ocr_result_valid:
                self.record_ocr(patch_ocr_result_valid)
                return self.result_dict

        # step5:替换转换ocr结果
        if region_ocr_result_src and patch_ocr_result_src and len(region_ocr_result_src) > 7 and len(patch_ocr_result_src) > 7:
      # if region_ocr_result_src and patch_ocr_result_src and len(region_ocr_result_src) == 11 and len(patch_ocr_result_src) == 11:
            transfer_result_better = self.ocr_detector.compare_and_generate(region_ocr_result_src, patch_ocr_result_src)
            transfer_result_valid = self.ocr_detector.generate_correct_code(transfer_result_better)
            if transfer_result_valid:
                self.record_ocr(transfer_result_valid)
                return self.result_dict
            else:
                self.record_ocr(" ")
                return self.result_dict

        self.record_ocr(" ")
        return self.result_dict

    def _region_level_ocr(self, region_img):
        text = self.ocr_detector.detect(region_img)
        return text

    def _patch_level_ocr(self, patch_img_list):
        text_list = []
        for patch in patch_img_list:
            text_patch = self.ocr_detector.detect(patch)

            text_list.append(text_patch)
        text = ''.join(text_list)
        return text

    def record_coordinates(self, coordinates):
        # 记录坐标
        self.result_dict["coordinates"]["xmin"] = coordinates[0]
        self.result_dict["coordinates"]["ymin"] = coordinates[1]
        self.result_dict["coordinates"]["xmax"] = coordinates[2]
        self.result_dict["coordinates"]["ymax"] = coordinates[3]
    def record_ocr(self, ocr_result):
        # 记录货柜号
        self.result_dict["coordinates"]["ocr_text"] = ocr_result

