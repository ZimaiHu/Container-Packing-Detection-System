from paddleocr import PaddleOCR
import os
import re
import logging
import cv2
logging.getLogger('ppocr').setLevel(logging.WARNING)
# 设置环境变量以避免某些库的潜在冲突
class ChaituoDetector:
    def __init__(self):
        pass

    def load_model(self, model_paths):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, use_mkldnn=False,
                             det_model_dir=model_paths[0])
    # 主探测函数
    def detect_chaituo(self, img_path):
        label_text = self.recognize_text_paddleocr(img_path)
        formatted_number = self.format_extracted_number(label_text)
        formatted_number = formatted_number.replace(".", "")
        return formatted_number
    # OCR识别
    def recognize_text_paddleocr(self, img_path):

        """ 使用PaddleOCR从图像中识别文本，专注于垂直对齐的文本 """

        # 对图像进行OCR识别
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"图像加载失败: {img_path}")

         #修改，对图片进行裁剪
        height, width, _ = img.shape

        # 计算裁剪区域
        start_row = height // 3
        end_row = 2 * height // 3
        start_col = width // 3
        end_col = 2 * width // 3

        img = img[start_row:end_row, start_col:end_col]

        if self.ocr is None:
            self.load_model()

        result = self.ocr.ocr(img, cls=True)
        for t in result:
            if t is None:
                return ""
            else:
                # 初始化一个空列表来存储所有识别到的文本
                all_texts = []

                # 遍历每个结果并提取文本
                for res in result:
                    for line in res:
                        # 追加每行文本（line[1][0]包含文本）
                        all_texts.append(line[1][0])

                # 将所有提取的文本片段连接成一个字符串
                combined_text = ' '.join(all_texts)
                return combined_text

    # 正则变换
    def format_extracted_number(self, text):
        """ 从文本中提取并格式化一个带有两个小数点的数字序列 """
        matches = re.findall(r'(?<!\d)\d+\.\d+\.\d+(?!\d)', text)
        if matches:
            match = matches[0]
            parts = match.split('.')
            if len(parts) == 3:
                # 格式化提取的数字为: 三位数.三位数.两位数
                formatted_number = f"{parts[0][-3:]}.{parts[1]}.{parts[2][:2]}"
                return formatted_number
        return ""  # 如果没有匹配项，则返回空字符串

