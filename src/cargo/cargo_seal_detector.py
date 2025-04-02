import numpy as np
from PIL import Image
import math
import logging
import cv2
import re
from collections import Counter
from ultralytics import YOLO  # 使用 YOLOv8
import matplotlib.pyplot as plt

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('ppocr').setLevel(logging.WARNING)


class SealDetector:
    def __init__(self, min_confidence=0.8):
        self.ocr = None
        self.model = None
        self.min_confidence = min_confidence

    def load_model(self, model_path):
        # 加载 YOLOv8 模型
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model = YOLO(model_path[0])  # 修改为直接加载 YOLOv8 模型
        self.model.predict(_dummy_image, verbose=False)

    def load_ocr_model(self, model_obj):
        self.ocr = model_obj

    def detect_seal(self, img_path):
        # 检测对象
        results = self.model(img_path)

        # 找到置信度最高的封条区域
        high_confidence_region = None
        max_confidence = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf.cpu().numpy())
                class_id = int(box.cls.cpu().numpy())

                if class_id == 0 and confidence > max_confidence:  # 假设封条类别为 0
                    high_confidence_region = (xmin, ymin, xmax, ymax)
                    max_confidence = confidence

        if not high_confidence_region:
            return {'fengtiao': []}

        # 读取图像并处理区域
        img_cv2 = cv2.imread(img_path)
        xmin, ymin, xmax, ymax = high_confidence_region
        region = img_cv2[ymin:ymax, xmin:xmax]

        # 矫正倾斜并进行OCR
        rotated_region = self.correct_skew(img_cv2, high_confidence_region)
        rotated_region_90 = cv2.rotate(rotated_region, cv2.ROTATE_90_CLOCKWISE)
        rotated_region_901 = cv2.rotate(rotated_region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ocr_results = [
            self.recognize_text_paddleocr(rotated_region),
            self.recognize_text_paddleocr(rotated_region_90),
            self.recognize_text_paddleocr(rotated_region_901)
        ]

        # 提取有用的文本
        max_length_text = self.extract_useful_text(ocr_results)

        # 构建结果
        seal_result = {
            "label_id": 1,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "OCR_result": max_length_text
        }

        return {'fengtiao': [seal_result]}

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
                    # 修改这里的文本处理逻辑
                    if '-' in text:
                        text = text.split('-')[-1]  # 取 '-' 后面的部分
                    filtered_text = re.sub(r'[^A-Za-z0-9]', '', text)
                    if filtered_text:
                        all_texts.append(filtered_text)
        return ' '.join(all_texts)

    # 挑选文字
    # def extract_useful_text(self, ocr_results):
    #     pattern = r'\b(?:[A-Z][A-Z0-9]{6,}\d|\d{8,})\b'
    #     matched_texts = []
    #     for text in ocr_results:
    #         matches = re.findall(pattern, text)
    #         matched_texts.extend(matches)
    #     if not matched_texts:
    #         return ""
    #     counter = Counter(matched_texts)
    #     return counter.most_common(1)[0][0]

    def extract_useful_text(self, ocr_results):
        # 原始的匹配模式
        pattern = r'\b(?:[A-Z][A-Z0-9]{6,}\d|\d{8,})\b'
        matched_texts = []
        # 遍历 OCR 结果，寻找匹配的文本
        for text in ocr_results:
            matches = re.findall(pattern, text)
            matched_texts.extend(matches)
        # 如果找到了符合原始模式的文本，返回出现次数最多的那个
        if matched_texts:
            counter = Counter(matched_texts)
            return counter.most_common(1)[0][0]
        # 如果没有符合原始模式的文本，寻找包含字母和数字的文本
        alpha_num_pattern = r'\b\w*\d\w*\b'
        alpha_num_texts = []
        for text in ocr_results:
            matches = re.findall(alpha_num_pattern, text)
            alpha_num_texts.extend(matches)
        # 如果找到了包含字母和数字的文本，返回出现次数最多的那个
        if alpha_num_texts:
            counter = Counter(alpha_num_texts)
            return counter.most_common(1)[0][0]
        # 如果没有任何符合的文本，返回空字符串
        return ""

    def draw_detections(self, image, results):
        """
        在图像上绘制检测到的封条和识别的文字。

        :param image: 原始图像 (numpy array, OpenCV 格式)
        :param results: 检测结果字典
        :return: 绘制了检测结果的图像
        """
        drawn_image = image.copy()

        for seal in results['fengtiao']:
            # 绘制边界框
            cv2.rectangle(drawn_image,
                          (seal['xmin'], seal['ymin']),
                          (seal['xmax'], seal['ymax']),
                          (0, 255, 0),  # 绿色
                          2)

            # 绘制文本
            label = f"{seal['label_id']}: {seal['OCR_result']}"
            cv2.putText(drawn_image,
                        label,
                        (seal['xmin'], seal['ymin'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # 白色
                        1,
                        cv2.LINE_AA)

        return drawn_image


if __name__ == '__main__':
    from paddleocr import PaddleOCR
    from modelscope import pipeline, Tasks

    detector = SealDetector()
    detector.load_model(["../../weights/cargo/fengtiao.pt"])

    handwritten_recognition_model = pipeline(Tasks.ocr_recognition,
                                             model="../../weights/ocr/cv_convnextTiny_ocr-recognition-handwritten_damo")
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, use_mkldnn=False,
                           det_model_dir="../../weights/ocr/ch_PP-OCRv4_det_infer")


    detector.load_ocr_model(paddle_ocr)

    result = detector.detect_seal('../../ceshitu/ceshi/fengtiao.jpg')
    print("result:", result)

    image = cv2.imread('../../ceshitu/ceshi/fengtiao.jpg')
    drawn_image = detector.draw_detections(image, result)
    cv2.imwrite('high_quality_output.jpg', drawn_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
