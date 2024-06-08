# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 13:39
@File     : container_number_ocr.py
@Project  : LLM
@Introduce:
"""
from modelscope import pipeline, Tasks
import math
from itertools import product
class ContainerNumberOCR:
    def __init__(self):
        self.ocr_recognition = None

    def load_model(self, model_path):

        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model=model_path)

    def detect(self, img):
        # step1:识别
        number = self.ocr_recognition(img)
        # step2:校验
        # check_number = self._verify(number)
        number = number['text'][0]
        number = number.replace(" ", "")
        # number = self.error_check(number)
        return number
    # 校验
    @staticmethod
    def character_conversion(val):
        # 转换成大写字母
        val = val.upper()
        basic_character = "0123456789A?BCDEFGHIJK?LMNOPQRSTU?VWXYZ"
        for i in range(len(basic_character)):
            if val == basic_character[i]:
                return i
        return -1  # 如果字符不在合法字符集内，返回 -1 以表示错误

    def check_carton_no(self, container_id):
        sum = 0
        container_id_list = list(container_id)

        # 计算前10位参数的和
        for i in range(10):
            char_value = self.character_conversion(container_id[i])
            if char_value == -1:
                # 处理无效字符
                return None
            sum += char_value * (2 ** i)

        # 将累加的和对11进行模运算，再对10取模，保证结果在0~9之间
        check_digit = sum % 11 % 10
        if container_id[-1].isdigit():
            if int(check_digit) == int(container_id_list[-1]):
                return container_id
            else:
                return None
        else:
            return None

    def validate_container_id(self, container_id):
        if len(container_id) == 11:
            letter_list = container_id[:4]
            number_list = container_id[4:]

            if letter_list.isalpha():
                if number_list[:-1].isdigit():
                    return self.check_carton_no(container_id)
                else:
                    return None
            else:
                return None
        else:
            return None
    def generate_correct_code(self, container_id):
        # 首先检查输入的集装箱号是否已经正确
        if self.validate_container_id(container_id) is not None:
            return container_id

        # 定义相似字符的映射规则
        letter_map = {
            '0': ['0','O','C'], '1': ['1', 'I', 'L'], '5': ['5', 'S'], '6': ['6', 'G'], '7': ['7', 'T'], '8': ['8', 'B'],
            'B': ['B','8'], 'G': ['G', '6'], 'I': ['I', '1', 'L','T'], 'O': ['O', '0', 'D','U'], 'S': ['S', '5'],
            'T': ['T','7','I'],
            'U': ['U', 'V','O'], 'V': ['V', 'U'], 'L': ['L', 'I', '1'], 'Z': ['Z', '2'], '2': ['2', 'Z'], 'C': ['C', 'G','0'],
            'D': ['D', 'O']
        }
        # 生成前4位的所有可能组合
        possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id[:4]])
        for combo in possible_combinations:
            corrected_id = ''.join(combo) + container_id[4:]
            if self.validate_container_id(corrected_id) is not None:
                return corrected_id
        return None  # 如果所有组合都不能通过校验