# from itertools import product
#
# class ContainerIDValidator:
#     def __init__(self):
#         self.basic_character = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#
#     @staticmethod
#     def character_conversion(val):
#         # 转换成大写字母
#         val = val.upper()
#         basic_character = "0123456789A?BCDEFGHIJK?LMNOPQRSTU?VWXYZ"
#         for i in range(len(basic_character)):
#             if val == basic_character[i]:
#                 return i
#         return -1  # 如果字符不在合法字符集内，返回 -1 以表示错误
#
#     def check_carton_no(self, container_id):
#         sum = 0
#         container_id_list = list(container_id)
#
#         # 计算前10位参数的和
#         for i in range(10):
#             char_value = self.character_conversion(container_id[i])
#             if char_value == -1:
#                 # 处理无效字符
#                 return None
#             sum += char_value * (2 ** i)
#
#         # 将累加的和对11进行模运算，再对10取模，保证结果在0~9之间
#         check_digit = sum % 11 % 10
#         if container_id[-1].isdigit():
#             if int(check_digit) == int(container_id_list[-1]):
#                 return container_id
#             else:
#                 return None
#         else:
#             return None
#
#     def validate_container_id(self, container_id):
#         if len(container_id) == 11:
#             letter_list = container_id[:4]
#             number_list = container_id[4:]
#
#             if letter_list.isalpha():
#                 if number_list[:-1].isdigit():
#                     return self.check_carton_no(container_id)
#                 else:
#                     return None
#             else:
#                 return None
#         else:
#             return None
#
#     def generate_correct_code(self, container_id):
#         # 首先检查输入的集装箱号是否已经正确
#         if self.validate_container_id(container_id) is not None:
#             return container_id
#         else:
#             letter_map = {
#                 '0': ['0', 'O', 'C','U'], '1': ['1', 'I', 'L', 'T'], '5': ['5', 'S'], '6': ['6', 'G'], '7': ['7', 'T'],
#                 '8': ['8', 'B'], '9': ['9', 'P'],
#                 'G': ['G', 'C'], 'I': ['I', 'L', 'T'], 'O': ['O', 'D', 'U'],
#                 'T': ['T', 'I'],
#                 'U': ['U', 'V', 'O', 'D'], 'V': ['V', 'U'], 'L': ['L', 'I'], '2': ['2', 'Z'],
#                 'C': ['C', 'G', 'O'], 'D': ['D', 'O', 'U']
#             }
#             digit_map = {
#                 'B': '8', 'C': '0', 'D': '0', 'G': '6', 'I': '1',
#                 'L': '1', 'O': '0', 'P': '9', 'Q': '0',
#                 'S': '9', 'T': '7', 'U': '0', 'V': '0','Z': '2'
#             }
#             if len(container_id) == 11:
#                 letter_list = container_id[:4]
#                 number_list = container_id[4:]
#                 # 检查后7位是否有字母
#                 if not number_list[:-1].isdigit():
#                     # 将后7位的字母映射为数字
#                     mapped_number_list = ''.join(digit_map.get(c, c) for c in number_list[:-1]) + number_list[-1]
#                     container_id = letter_list + mapped_number_list
#                     if self.validate_container_id(container_id) is not None:
#                         return container_id
#                     else:
#                         possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id[:4]])
#                         max_same_chars = 0
#                         best_corrected_id = None
#                         for combo in possible_combinations:
#                             corrected_id = ''.join(combo) + container_id[4:]
#                             print(corrected_id)
#                             if self.validate_container_id(corrected_id) is not None:
#                                 same_chars = sum(1 for x, y in zip(corrected_id, container_id) if x == y)
#                                 if same_chars > max_same_chars:
#                                     print(same_chars)
#                                     max_same_chars = same_chars
#                                     best_corrected_id = corrected_id
#                         return best_corrected_id if best_corrected_id is not None else None
#                 else:
#                     possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id[:4]])
#                     max_same_chars = 0
#                     best_corrected_id = None
#                     for combo in possible_combinations:
#                         corrected_id = ''.join(combo) + container_id[4:]
#                         print(corrected_id)
#                         if self.validate_container_id(corrected_id) is not None:
#                             same_chars = sum(1 for x, y in zip(corrected_id, container_id) if x == y)
#                             if same_chars > max_same_chars:
#                                 print(same_chars)
#                                 max_same_chars = same_chars
#                                 best_corrected_id = corrected_id
#                     return best_corrected_id if best_corrected_id is not None else None
#             else:
#                 # 处理长度大于11位的情况
#                 start_index = 0
#                 while start_index < len(container_id) and container_id[start_index].isdigit():
#                     start_index += 1
#
#                 end_index = len(container_id) - 1
#                 while end_index >= 0 and container_id[end_index].isdigit():
#                     end_index -= 1
#                 max_same_chars = 0
#                 best_corrected_id = None
#                 for i in range(start_index, end_index - 4 + 2):
#                     container_id2 = container_id[i:i + 4] + container_id[len(container_id) -7:]
#                     print(container_id2)
#                     letter_list = container_id2[:4]
#                     number_list = container_id2[4:]
#                     # 检查后7位是否有字母
#                     if not number_list[:-1].isdigit():
#                         # 将后7位的字母映射为数字
#                         mapped_number_list = ''.join(digit_map.get(c, c) for c in number_list[:-1]) + number_list[-1]
#                         container_id2 = letter_list + mapped_number_list
#                         if self.validate_container_id(container_id2) is not None:
#                             return container_id2
#                         else:
#                             possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id2[:4]])
#                             for combo in possible_combinations:
#                                 corrected_id = ''.join(combo) + container_id2[4:]
#                                 print(corrected_id)
#                                 if self.validate_container_id(corrected_id) is not None:
#                                     same_chars = sum(1 for x, y in zip(corrected_id, container_id2) if x == y)
#                                     if same_chars > max_same_chars:
#                                         print(same_chars)
#                                         max_same_chars = same_chars
#                                         best_corrected_id = corrected_id
#                     else:
#                         possible_combinations = product(*[letter_map.get(c, [c]) for c in container_id2[:4]])
#                         for combo in possible_combinations:
#                             corrected_id = ''.join(combo) + container_id2[4:]
#                             print(corrected_id)
#                             if self.validate_container_id(corrected_id) is not None:
#                                 same_chars = sum(1 for x, y in zip(corrected_id, container_id2) if x == y)
#                                 if same_chars > max_same_chars:
#                                     print(same_chars)
#                                     max_same_chars = same_chars
#                                     best_corrected_id = corrected_id
#                 if best_corrected_id is not None:
#                     return best_corrected_id
#                 else:
#                     return None
# # 使用示例
# validator = ContainerIDValidator()
# container_id = 'BM0U6622791'
# correct_code = validator.generate_correct_code(container_id)
# if correct_code:
#     print("Corrected Container ID:", correct_code)
# else:
#     print("No valid correction found.")
# import paddleocr
# print(paddleocr.__version__)