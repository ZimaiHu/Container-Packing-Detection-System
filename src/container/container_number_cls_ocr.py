# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/7/9 10:26
@File     : container_number_cls_ocr.py
@Project  : YiJia
@Introduce:
"""
import re
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# 分类索引
cls_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
            13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
            25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

# 替换表
digit_to_letter = {'0': ['U','O','D'], '1': ['T','I'], '2': ['S']}
letter_to_digit = {'B': ['8'], 'D': ['0'],'I': ['1'],'O':['0'],'T':['1','7'],'L':['1','7'],'U':'0'}
letter_to_letter = {'D': ['U'],'U':['D']}
digit_to_digit = {'1':'7'}


class ContainerNumberCLSOCR:
    def __init__(self):
        self.cls_ocr_recognition = None
        self.model = None
        self.alternative_option_level1 = []
        self.alternative_option_level2 = []

    def load_model(self, model_path):
        self.model = YOLO(model_path)  # 横向小块分割

        self.warmup()

    def warmup(self):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(_dummy_image, verbose=False)

    def detect(self, image):
        txt_list = []
        res = self.model.predict(source=image, show=False, save=False, verbose=False)[0]
        top3 = res.probs.top5[0:3]
        for index in top3:
            txt_list.append(cls_dict[index])

        return txt_list

    #
    def convert_level1(self, text):
        # 判断text是否为11位
        if len(text) != 11:
            return [text]  # 如果不是11位，返回原文本
        # 判断前4位字母后7位数字
        pattern = r'^[A-Z]{4}\d{7}$'
        if re.match(pattern, text):
            return [text]
        # 取前4位作为字母区，后7位作为数字区
        letter_part = text[:4]
        number_part = text[4:]
        alternative_option = []
        # 递归生成所有可能的替换
        def generate_replacements(current, index, letter_part, number_part):

            if index == len(letter_part) + len(number_part):
                alternative_option.append(current)
                return

            if index < len(letter_part):
                char = letter_part[index]
                replacements = digit_to_letter.get(char, [char])  # 只使用 digit_to_letter 的替换规则
            else:
                char = number_part[index - len(letter_part)]
                replacements = letter_to_digit.get(char, [char])  # 只使用 letter_to_digit 的替换规则

            for replacement in replacements:
                generate_replacements(current + replacement, index + 1, letter_part, number_part)

        # 开始递归生成替换结果
        generate_replacements("", 0, letter_part, number_part)
        return alternative_option

