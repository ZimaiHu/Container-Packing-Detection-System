from paddleocr import PaddleOCR
import os
import re
import logging
import cv2
logging.getLogger('ppocr').setLevel(logging.WARNING)
# 设置环境变量以避免某些库的潜在冲突
class ChaituoDetector:
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr = None
    def load_model(self):
        pass
        # self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=True, use_mkldnn=False)
    # 主探测函数
    def detect_chaituo(self, img_path):
        label_text = self.recognize_text_paddleocr(img_path)
        formatted_number = self.format_extracted_number(label_text)
        formatted_number = formatted_number.replace(".", "")
        return formatted_number
    # OCR识别
    def recognize_text_paddleocr(self, img_path):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=True, use_mkldnn=False)
        """ 使用PaddleOCR从图像中识别文本，专注于垂直对齐的文本 """

        # 对图像进行OCR识别
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"图像加载失败: {img_path}")

        #  #修改，对图片进行裁剪
        # height, width, _ = img.shape
        #
        # # 计算裁剪区域
        # start_row = height // 3
        # end_row = 2 * height // 3
        # start_col = width // 3
        # end_col = 2 * width // 3
        #
        # img = img[start_row:end_row, start_col:end_col]

        if self.ocr is None:
            self.load_model()

        result = self.ocr.ocr(img, cls=True)
        print(result)
        for t in result:
            if t is None:
                return ""
            else:
                # 初始化一个空列表来存储所有识别到的文本
                all_texts = []

                # 遍历每个结果并提取文本
                for res in result:
                    print(res)
                    for line in res:
                        # 追加每行文本（line[1][0]包含文本）
                        all_texts.append(line[1][0])

                # 将所有提取的文本片段连接成一个字符串
                combined_text = ' '.join(all_texts)
                return combined_text

    # 正则变换
    def format_extracted_number(self, text):
        parts = text.split()
        valid_numbers = []
        dot_part = None
        if parts:
            for part in parts:
                number = ''.join(re.findall(r'\d', part))
                if len(number) == 8:
                    valid_numbers.append(part)
                    if '.' in part:
                        dot_part = part

            if len(valid_numbers) == 2 and dot_part:
                return dot_part
            return ' '.join([''.join(re.findall(r'\d', part)) for part in valid_numbers])
        return ""

 