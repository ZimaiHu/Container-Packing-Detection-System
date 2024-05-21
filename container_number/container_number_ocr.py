# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 13:39
@File     : container_number_ocr.py
@Project  : LLM
@Introduce:
"""
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
class OCRDetector:
    def __init__(self):
        self.ocr_recognition = pipeline(Tasks.ocr_recognition,
                                        model='container_number/model/cv_crnn_ocr-recognition-general_damo')

    def detect(self, img):
        # step1:识别
        number = self.ocr_recognition(img)
        # step2:校验
        # check_number = self._verify(number)
        number=number['text'][0]
        number=number.replace(" ","")
        number = self.error_check(number)
        return number
    # 校验
    def characterConversion(self,str):
        # 转换成大写字母
        str = str.upper()
        basic_character = "0123456789A?BCDEFGHIJK?LMNOPQRSTU?VWXYZ"
        for i in range(0, len(basic_character)):
            if str == basic_character[i]:
                return i
    def compute_check_digit(self,containerId):
        sum = 0
        # 计算前10位参数的和
        for i in range(0, 10):
            # print(f"当前位置为{i},字母为{containerId[i]},当前参数对应的的数值为{characterConversion(containerId[i])}")
            # print(f"当前计算公式为[{containerId[i]}* 2的{i}次方]，当前的值为{characterConversion(containerId[i]) * (2 ** i)}")
            sum += self.characterConversion(containerId[i]) * (2 ** i)

        # 将累加的和对11进行模运算:sum % 11,当校验位等于10时要继续模运算,sum % 11 % 10,保证最终结果为0~9直接的数
        check_digit = sum % 11 % 10
        return check_digit

    def error_check(self,formatted_code):
        original_length = len(formatted_code)
        if original_length < 11:
            # Add default letters to the first part if necessary
            if original_length < 4:
                formatted_code = formatted_code.ljust(4, 'A')
            # Add default numbers to the middle part if necessary
            if original_length < 10:
                formatted_code = formatted_code.ljust(10, '0')

        elif len(formatted_code) > 11:
            formatted_code = formatted_code[:10]  # Trimming to first 10 characters for check digit calculation
        # Replace common misread characters
        replacements = {'1': 'I', '4': 'A', '6': 'G', '8': 'B', '0': 'O'}
        reverse_replacements = {v: k for k, v in replacements.items()}

        formatted_code = list(formatted_code)
        # Fix any mistaken characters
        for i in range(4):
            if formatted_code[i] in replacements:
                formatted_code[i] = replacements[formatted_code[i]]
        for i in range(4, 10):
            if formatted_code[i] in reverse_replacements:
                formatted_code[i] = reverse_replacements[formatted_code[i]]

        # Calculate the correct check digit
        correct_check_digit = self.compute_check_digit(formatted_code)
        formatted_code = formatted_code[:10]  # Ensure we only have the first 10 characters
        formatted_code.append(str(correct_check_digit))  # Add the calculated check digit

        return "".join(formatted_code)