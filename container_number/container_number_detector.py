# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 13:03
@File     : text_image_v2h.py
@Project  : LLM
@Introduce:
"""
# step 2: text_image_v2h.py
# How to use: python3 text_image_v2h.py
import cv2
from container_number.container_number_v2h import V2HCharDetector  # character detection, vertical to horizontal
from container_number.container_number_ocr import OCRDetector
class ContainerNumberDetector:
    def __init__(self):

        self.v2h_char_detector = V2HCharDetector()  # 竖排转横排
        self.ocr_detector = OCRDetector()  # ocr识别
        # self.processed_count = 0

    def detect(self, img):
        # step1:调整图像方向
        adjusted_img = self.adjust_image_orientation(img)
        cv2.imwrite('output_images/adjusted_img.png', adjusted_img)
        text = self.recognize_text(adjusted_img)
        return text

    def adjust_image_orientation(self, img):
        if img is not None:
            height, width = img.shape[:2]
            if height > width:  # vertical image
                _, processed_img = self.v2h_char_detector.detect(img)
                return processed_img

            else:
                return img
    def recognize_text(self,img):
        text = self.ocr_detector.detect(img)
        return text
if __name__ == '__main__':
    detector = ContainerNumberDetector()
    test_image = cv2.imread(r"D:\final\ceshitu\be9.png")
    txt = detector.detect(test_image)
    print(txt)
