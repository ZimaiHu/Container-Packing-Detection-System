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

from src.container.container_number_region import ContainerNumberRegion
from src.container.container_number_ocr import ContainerNumberOCR
from src.container.container_number_patch import ContainerNumberPatch
from src.container.container_number_cls_ocr import ContainerNumberCLSOCR
import os
import matplotlib.pyplot as plt
from typing import List, Dict


class ContainerNumberDetector:
    def __init__(self):
        self.result_dict = {"coordinates": {"xmin": None, "ymin": None, "xmax": None, "ymax": None, "ocr_text": None}}
        self.region_detector = ContainerNumberRegion()  # 横向货柜号/纵向货柜号 区域检测
        self.patch_detector = ContainerNumberPatch()  # 单个货号检测
        self.ocr_detector = ContainerNumberOCR()  # ocr识别
        self.cls_detector = ContainerNumberCLSOCR()  # 基于yolo-cls ocr识别

    def load_model(self, model_path):
        self.region_detector.load_model(model_path[0])
        self.patch_detector.load_model(model_path[1])
        self.ocr_detector.load_model(model_path[2])
        self.cls_detector.load_model(model_path[3])

    def detect(self, img):
        # step1:检测货柜号区域
        coordinates, number_region = self.region_detector.detect(img)
        if number_region is None:
            return {"coordinates": None, "ocr_text": None}
        self.record_coordinates(coordinates)

        # step2:调整图像方向（应对竖排）
        adjusted_img, adjusted_img_list = self.patch_detector.detect(number_region)
        if adjusted_img is None:
            return {"coordinates": None, "ocr_text": None}

        # step3:区域级别的ocr识别
        region_ocr_result_src = self._region_level_ocr(adjusted_img)
        # 标准校验ocr结果
        if region_ocr_result_src:
            # print("region_ocr_result_src: ", region_ocr_result_src)
            region_ocr_result_valid = self.ocr_detector.validate_container_id(region_ocr_result_src)
            if region_ocr_result_valid:
                # print("region_ocr_result_valid: ", region_ocr_result_valid)
                self.record_ocr(region_ocr_result_valid)
                return self.result_dict

        # step4:patch级别的ocr识别
        patch_ocr_result_src = self._patch_level_ocr(adjusted_img_list)
        if patch_ocr_result_src:
            # print("patch_ocr_result_src: ", patch_ocr_result_src)
            patch_ocr_result_valid_list = self.cls_detector.convert_level1(patch_ocr_result_src)
            for index in patch_ocr_result_valid_list:
                patch_ocr_result_valid = self.ocr_detector.validate_container_id(index)
                if patch_ocr_result_valid:
                    # print("patch_ocr_result_valid: ", patch_ocr_result_valid)
                    self.record_ocr(patch_ocr_result_valid)
                    return self.result_dict

        # step5:替换转换ocr结果
        if region_ocr_result_src and patch_ocr_result_src and len(region_ocr_result_src) > 7 and len(
                patch_ocr_result_src) > 7:
            # print("region_ocr_result_src: ", region_ocr_result_src)
            # if region_ocr_result_src and patch_ocr_result_src and len(region_ocr_result_src) == 11 and len(patch_ocr_result_src) == 11:
            transfer_result_better = self.ocr_detector.compare_and_generate(region_ocr_result_src, patch_ocr_result_src)
            transfer_result_valid = self.ocr_detector.generate_correct_code(transfer_result_better)
            if transfer_result_valid:
                # print("transfer_result_valid1: ", transfer_result_valid)
                self.record_ocr(transfer_result_valid)
                return self.result_dict
            else:
                # print("transfer_result_valid2: ", transfer_result_valid)
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
            # text_patch = self.ocr_detector.detect(patch)
            # text_list.append(text_patch)
            text_patch_top3 = self.cls_detector.detect(patch)
            text_patch_top1 = text_patch_top3[0]
            text_list.append(text_patch_top1)
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

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for detection in detections:
            if 'coordinates' in detection:
                # Draw box
                xmin = detection['coordinates']['xmin']
                ymin = detection['coordinates']['ymin']
                xmax = detection['coordinates']['xmax']
                ymax = detection['coordinates']['ymax']
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Draw ocr_text as label
                ocr_text = detection['coordinates']['ocr_text']
                cv2.putText(image, ocr_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return image

if __name__ == '__main__':
    detector = ContainerNumberDetector()
    detector.load_model([
        "../../weights/contain_number/container_number_region_5_7_3.pt",
        [
            "../../weights/contain_number/container_number_patch_shu_7_3.pt",
            "../../weights/contain_number/container_number_patch_heng_7_3.pt"
        ],
        "../../weights/ocr/cv_crnn_ocr-recognition-general_damo_finetuning",
        "../../weights/ocr/yolo_cls_ocr/container_number_cls_ocr.pt"
    ])

    image = cv2.imread('../../ceshitu/fault/huoguihaof2.jpg')  # 请根据实际情况填写图像路径
    result = detector.detect(image)
    print("result:", result)

    # 将 result 包装在列表中传递给 draw_detections 方法
    drawn_image = detector.draw_detections(image, [result])
    cv2.imwrite('high_quality_output.jpg', drawn_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()