from paddleocr import PaddleOCR
import os
import re
import logging
import cv2
logging.getLogger('ppocr').setLevel(logging.WARNING)
# Set environment variable to avoid potential conflicts with some libraries
class ChaituoDetector:
    def __init__(self, lang='en'):
        pass
    def load_model(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_mem=8000, gpu_id=3)
#主探测函数
    def detect_chaituo(self, img_path):
        label_text = self.recognize_text_paddleocr(img_path)
        formatted_number = self.format_extracted_number(label_text)
        formatted_number = formatted_number.replace(".", "")
        return formatted_number
#OCR识别
    def recognize_text_paddleocr(self, img_path):
        """ Recognize text from an image using PaddleOCR, focusing on vertically aligned text """
        # Perform OCR on the image
        img = cv2.imread(img_path)
        result = self.ocr.ocr(img, cls=True)
        # Initialize an empty list to store all recognized texts
        all_texts = []

        # Loop through each result and extract text
        for res in result:
            for line in res:
                # Append each line of text (line[1][0] contains the text)
                all_texts.append(line[1][0])

        # Join all extracted text pieces into a single string
        combined_text = ' '.join(all_texts)
        return combined_text
#正则变换
    def format_extracted_number(self, text):
        """ Extract and format a numeric sequence with two decimal points from the text """
        matches = re.findall(r'(?<!\d)\d+\.\d+\.\d+(?!\d)', text)
        if matches:
            match = matches[0]
            parts = match.split('.')
            if len(parts) == 3:
                # Format extracted number as: three digits.three digits.two digits
                formatted_number = f"{parts[0][-3:]}.{parts[1]}.{parts[2][:2]}"
                return formatted_number
        return ""  # Return an empty string if no match is found