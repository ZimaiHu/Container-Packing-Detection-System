# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 13:39
@File     : container_number_ocr.py
@Project  : LLM
@Introduce:
"""
import os

from modelscope import pipeline, Tasks
from itertools import product


class ContainerNumberOCR:
    def __init__(self):
        self.ocr_recognition = None

    def load_model(self, model_path):

        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model=model_path)


    def detect(self, img):
        number = self.ocr_recognition(img)
        number = number['text'][0]
        number = number.replace(" ", "")
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

    def compare_and_generate(self, result1, result2):

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
            min_length = min(len(part1), len(part2))
            for i in range(min_length):
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

    def generate_correct_code(self, container_id):
        # 首先检查输入的集装箱号是否已经正确
        if self.validate_container_id(container_id) is not None:
            return container_id
        else:
            letter_map = {
                '0': ['0', 'O', 'C', 'U'], '1': ['1', 'I', 'L', 'T'], '5': ['5', 'S'], '6': ['6', 'G'], '7': ['7', 'T'],
                '8': ['8', 'B'], '9': ['9', 'P'],
                'G': ['G', 'C'], 'I': ['I', 'L', 'T'], 'O': ['O', 'D', 'U'],
                'T': ['T', 'I'],
                'U': ['U', 'V', 'O', 'D'], 'V': ['V', 'U'], 'L': ['L', 'I'], '2': ['2', 'Z'],
                'C': ['C', 'G', 'O'], 'D': ['D', 'O', 'U']
            }
            digit_map = {
                'B': '8', 'C': '0', 'D': '0', 'G': '6', 'I': '1',
                'L': '1', 'O': '0', 'P': '9', 'Q': '0',
                'S': '9', 'T': '7', 'U': '0', 'V': '0', 'Z': '2'
            }
            if len(container_id) == 11:
                letter_list = container_id[:4]
                number_list = container_id[4:]
                # 检查后7位是否有字母
                if not number_list[:-1].isdigit():
                    # 将后7位的字母映射为数字
                    mapped_number_list = ''.join(digit_map.get(c, c) for c in number_list[:-1]) + number_list[-1]
                    container_id = letter_list + mapped_number_list
                    if self.validate_container_id(container_id) is not None:
                        return container_id
                    else:
                        possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id[:4]])
                        max_same_chars = 0
                        best_corrected_id = None
                        for combo in possible_combinations:
                            corrected_id = ''.join(combo) + container_id[4:]
                            # print(corrected_id)
                            if self.validate_container_id(corrected_id) is not None:
                                same_chars = sum(1 for x, y in zip(corrected_id, container_id) if x == y)
                                if same_chars > max_same_chars:
                                    # print(same_chars)
                                    max_same_chars = same_chars
                                    best_corrected_id = corrected_id
                        return best_corrected_id if best_corrected_id is not None else None
                else:
                    possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id[:4]])
                    max_same_chars = 0
                    best_corrected_id = None
                    for combo in possible_combinations:
                        corrected_id = ''.join(combo) + container_id[4:]
                        # print(corrected_id)
                        if self.validate_container_id(corrected_id) is not None:
                            same_chars = sum(1 for x, y in zip(corrected_id, container_id) if x == y)
                            if same_chars > max_same_chars:
                                # print(same_chars)
                                max_same_chars = same_chars
                                best_corrected_id = corrected_id
                    return best_corrected_id if best_corrected_id is not None else None
            else:
                # 处理长度大于11位的情况
                start_index = 0
                while start_index < len(container_id) and container_id[start_index].isdigit():
                    start_index += 1

                end_index = len(container_id) - 1
                while end_index >= 0 and container_id[end_index].isdigit():
                    end_index -= 1
                max_same_chars = 0
                best_corrected_id = None
                for i in range(start_index, end_index - 4 + 2):
                    container_id2 = container_id[i:i + 4] + container_id[len(container_id) - 7:]
                    # print(container_id2)
                    letter_list = container_id2[:4]
                    number_list = container_id2[4:]
                    # 检查后7位是否有字母
                    if not number_list[:-1].isdigit():
                        # 将后7位的字母映射为数字
                        mapped_number_list = ''.join(digit_map.get(c, c) for c in number_list[:-1]) + number_list[-1]
                        container_id2 = letter_list + mapped_number_list
                        if self.validate_container_id(container_id2) is not None:
                            return container_id2
                        else:
                            possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id2[:4]])
                            for combo in possible_combinations:
                                corrected_id = ''.join(combo) + container_id2[4:]
                                # print(corrected_id)
                                if self.validate_container_id(corrected_id) is not None:
                                    same_chars = sum(1 for x, y in zip(corrected_id, container_id2) if x == y)
                                    if same_chars > max_same_chars:
                                        # print(same_chars)
                                        max_same_chars = same_chars
                                        best_corrected_id = corrected_id
                    else:
                        possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id2[:4]])
                        for combo in possible_combinations:
                            corrected_id = ''.join(combo) + container_id2[4:]
                            # print(corrected_id)
                            if self.validate_container_id(corrected_id) is not None:
                                same_chars = sum(1 for x, y in zip(corrected_id, container_id2) if x == y)
                                if same_chars > max_same_chars:
                                    # print(same_chars)
                                    max_same_chars = same_chars
                                    best_corrected_id = corrected_id
                if best_corrected_id is not None:
                    return best_corrected_id
                else:
                    return None
