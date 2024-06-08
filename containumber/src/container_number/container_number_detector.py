# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 13:03
@File     : text_image_v2h.py
@Project  : LLM
@Introduce:
"""
import warnings
from datetime import datetime
import cv2
from containumber.src.container_number.container_number_region import ContainerNumberRegion
from containumber.src.container_number.container_number_v2h import ContainerNumberV2h
from containumber.src.container_number.container_number_ocr import ContainerNumberOCR
from containumber.src.container_number.container_number_patch import ContainerNumberPatch
from containumber.src.config import v2h_model_path, ocr_model_path, region_model_path, patch_model_path
import os


class ContainerNumberDetector:
    def __init__(self):
        self.container_number = None
        self.region_detector = ContainerNumberRegion()  # 横向货柜号/纵向货柜号
        self.patch_detector = ContainerNumberPatch()  # 货柜号单个号裁剪
        self.v2h_detector = ContainerNumberV2h()  # 竖排转横排
        self.ocr_detector = ContainerNumberOCR()  # ocr识别
        # self.processed_count = 0
    def load_model(self, base_path):
        self.region_detector.load_model(os.path.join(base_path, region_model_path))
        self.v2h_detector.load_model(os.path.join(base_path, v2h_model_path))  # 竖排转横排
        self.ocr_detector.load_model(os.path.join(base_path, ocr_model_path))
        self.patch_detector.load_model(os.path.join(base_path, patch_model_path))

    def compare_and_generate(self,result1, result2):
        # 对比前4位，选择字母多的
        def select_letters(part1, part2):
            count1 = sum(1 for c in part1 if c.isalpha())
            count2 = sum(1 for c in part2 if c.isalpha())
            if count1 > count2:
                return part1
            elif count2 > count1:
                return part2
            else:
                return part1  # 如果字母数量相等，选择第一个

        # 对比后6位，选择数字优先的
        def select_digits(part1, part2):
            result = []
            for i in range(len(part1)):
                if part1[i].isdigit() and not part2[i].isdigit():
                    result.append(part1[i])
                elif part2[i].isdigit() and not part1[i].isdigit():
                    result.append(part2[i])
                else:
                    result.append(part1[i])  # 优先选择第一个的字符
            return ''.join(result)

        # 分割前4位和第五位及之后
        front1, back1 = result1[:4], result1[4:]
        front2, back2 = result2[:4], result2[4:]

        # 选择前4位和第五位及之后
        selected_front = select_letters(front1, front2)
        selected_back = select_digits(back1, back2)

        # 合并结果
        final_result = selected_front + selected_back
        return final_result
    def detect(self, img):
        # step1:检测货柜号区域
        coordinates, number_region = self.region_detector.detect(img)
        if number_region is None:
            return {"0": []}
        else:
            # step2:调整图像方向（应对竖排）
            adjusted_img = self._adjust_image_orientation(number_region)
            # step3:区域级别的ocr识别
            region_ocr_result = self._region_level_ocr(adjusted_img)
            print(region_ocr_result)
            region_ocr_result1 = self.ocr_detector.validate_container_id(region_ocr_result)
            print(region_ocr_result1)
            if region_ocr_result1 is None:
                # step4:块级别的ocr识别
                patch_img_list = self.patch_detector.detect(number_region)
                patch_ocr_result = self._patch_level_ocr(patch_img_list)
                print(patch_ocr_result)
                patch_ocr_result1 = self.ocr_detector.validate_container_id(patch_ocr_result)
                print(patch_ocr_result1)
                if patch_ocr_result1 is None:
                    # 检查输入结果是否都是11位
                    if len(region_ocr_result) != 11 or len(patch_ocr_result) != 11:
                        return {"0": [
                            {"xmin": coordinates[0], "ymin": coordinates[1], "xmax": coordinates[2],
                             "ymax": coordinates[3],
                             "OCR_result": ' '}]}
                    else:
                        #强制转换校验
                        better=self.compare_and_generate(region_ocr_result,patch_ocr_result)
                        print(better)
                        final=self.ocr_detector.generate_correct_code(better)
                        if final is None:
                            return {"0": [
                            {"xmin": coordinates[0], "ymin": coordinates[1], "xmax": coordinates[2], "ymax": coordinates[3],
                             "OCR_result": ' '}]}
                        else:
                            return {"0": [
                                {"xmin": coordinates[0], "ymin": coordinates[1], "xmax": coordinates[2], "ymax": coordinates[3],
                                 "OCR_result": final}]}
                else:
                    return {"0": [
                        {"xmin": coordinates[0], "ymin": coordinates[1], "xmax": coordinates[2], "ymax": coordinates[3],
                         "OCR_result": patch_ocr_result1}]}
            else:
                return {"0": [
                    {"xmin": coordinates[0], "ymin": coordinates[1], "xmax": coordinates[2], "ymax": coordinates[3],
                     "OCR_result": region_ocr_result1}]}
    def _adjust_image_orientation(self, img):
        height, width = img.shape[:2]
        if height > width:  # vertical image
            _, processed_img = self.v2h_detector.detect(img)
            return processed_img
        else:
            return img

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
