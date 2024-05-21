import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import logging
from paddleocr import PaddleOCR
import cv2
import re
from collections import Counter
from ultralytics import YOLO  # 使用 YOLOv8

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('ppocr').setLevel(logging.WARNING)


class SealDetector:
    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        logging.getLogger('ppocr').setLevel(logging.WARNING)

    def load_model(self, model_path):
        # 加载 YOLOv8 模型
        self.model = YOLO(model_path[0])  # 修改为直接加载 YOLOv8 模型
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_mem=8000, gpu_id=3)

    # 主探测函数
    def detect_seal(self, img_path):
        detected_objects = self.detect_objects(img_path)
        label_0_regions = self.draw_boxes_on_image(img_path, detected_objects)

        img_cv2 = cv2.imread(img_path)
        seal_results = []
        label_id = 1
        for region in label_0_regions:
            rotated_region = self.correct_skew(img_cv2, region)
            rotated_region_90 = cv2.rotate(rotated_region, cv2.ROTATE_90_CLOCKWISE)
            rotated_region_901 = cv2.rotate(rotated_region, cv2.ROTATE_90_COUNTERCLOCKWISE)
            result1 = self.recognize_text_paddleocr(rotated_region)
            result190 = self.recognize_text_paddleocr(rotated_region_90)
            result1901 = self.recognize_text_paddleocr(rotated_region_901)
            ocr_results = [result1, result190, result1901]
            max_length_text = self.extract_useful_text(ocr_results)
            xmin, ymin, xmax, ymax = region
            seal_result = {
                "label_id": label_id,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "OCR_result": max_length_text
            }
            seal_results.append(seal_result)
            label_id += 1
        return {'fengtiao': seal_results}

    # 探测封条
    def detect_objects(self, img_path):
        logger.info(f"Loading image: {img_path}")
        results = self.model(img_path)
        return results

    # 找位置
    def draw_boxes_on_image(self, img_path, detected_objects):
        img = Image.open(img_path)
        label_0_regions = []
        for result in detected_objects:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框和置信度
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
                confidence = box.conf.cpu().numpy()
                class_id = box.cls.cpu().numpy()

                if confidence >= self.min_confidence and class_id == 0:  # 假设手势类别为 0
                    label_0_regions.append((xmin, ymin, xmax, ymax))
        return label_0_regions

    # 正变换
    def correct_skew(self, img_cv2, region):
        xmin, ymin, xmax, ymax = region
        region_img = img_cv2[ymin:ymax, xmin:xmax]
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None:
            return region_img
        rotate_angle = 0
        max_len = 0
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                if x1 == x2:
                    continue
                line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_len > max_len:
                    max_len = line_len
                    t = float(y2 - y1) / (x2 - x1)
                    rotate_angle = math.degrees(math.atan(t))
                    if rotate_angle > 45:
                        rotate_angle = -90 + rotate_angle
                    elif rotate_angle < -45:
                        rotate_angle = 90 + rotate_angle
        (h, w) = region_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        return cv2.warpAffine(region_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 识别文字
    def recognize_text_paddleocr(self, image):
        result = self.ocr.ocr(image, cls=True)
        if not result or result[0] is None:
            return ""
        all_texts = []
        for res in result:
            if res:
                for line in res:
                    text = line[1][0]
                    filtered_text = re.sub(r'[^A-Za-z0-9]', '', text)
                    if filtered_text:
                        all_texts.append(filtered_text)
        return ' '.join(all_texts)

    # 挑选文字
    def extract_useful_text(self, ocr_results):
        pattern = r'\b(?:[A-Z][A-Z0-9]{6,}\d|\d{8,})\b'
        matched_texts = []
        for text in ocr_results:
            matches = re.findall(pattern, text)
            matched_texts.extend(matches)
        if not matched_texts:
            return ""
        counter = Counter(matched_texts)
        return counter.most_common(1)[0][0]