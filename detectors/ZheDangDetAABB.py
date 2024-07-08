from PIL import ImageFont
import os
import re
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.getLogger('ppocr').setLevel(logging.WARNING)
from detect.AlgorithmManager import AlgorithmManager
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import pandas as pd
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ZheDangDetector:
    def __init__(self):
        # 初始化模型
        self.core = AlgorithmManager()

    def load_model(self, model_path):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        #加载遮挡模型
        self.model_zhedang = YOLO(model_path[0])        #改动
        self.model_zhedang.predict(_dummy_image, verbose=False)

        #加载手写模型
        self.model_shouxie = YOLO(model_path[1])
        self.model_shouxie.predict(_dummy_image, verbose=False)

        #加载ocr识别模型
        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model=model_path[2])
        start_time = time.time()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_mem=8000,
                             det_model_dir='weights/ch_PP-OCRv4_det_infer')

        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"OC took {elapsed_time:.4f} seconds")

    # 主探测函数
    def detect_zhedang_label(self, img_path, min_confidence=0.65):

        image = cv2.imread(img_path)

        start_time = time.time()
        results = self.model_zhedang.predict(source=image, show=False, device=0, save=False, verbose=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"model_zhedangpredict took {elapsed_time:.4f} seconds")

        classes = results[0].boxes.cls.tolist()
        coordinates = results[0].boxes.xyxy.tolist()
        confidences = results[0].boxes.conf.tolist()

        data = []
        for i in range(len(classes)):
            data.append(coordinates[i] + [confidences[i], int(classes[i])])

        df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
        detected_goods_and_labels = df
        img = cv2.imread(img_path)

        start_time = time.time()
        results = self.process_detections(detected_goods_and_labels, img, min_confidence)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"process_detections took {elapsed_time:.4f} seconds")
        for item in results:
            labelingood = item['labelingood']
            if len(labelingood) > 1:
                valid_labels = [label for label in labelingood if len(label['ocr_result']) > 3]
                if len(valid_labels) > 1:
                    labelingood = [max(valid_labels, key=lambda x: len(x['ocr_result']))]
            item['labelingood'] = labelingood

        return results

    # 手写识别
    def crop_and_ocr(self, image):
        # 使用手写识别模型进行预测
        start_time = time.time()
        results = self.model_shouxie.predict(source=image, show=False, save=False, verbose=False)[0]
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"model_shouxie.predict took {elapsed_time:.4f} seconds")

        coord_list = results.boxes.xyxy.tolist()
        if coord_list:  # 确保coord_list不为空
            x1, y1, x2, y2 = map(int, coord_list[0])
            cropped_image = image[y1:y2, x1:x2]
            figure_img = cropped_image
            result = self.ocr_recognition(figure_img)
            return result['text'][0]
        else:
            return ""

    # 综合步骤
    '''
    @方法介绍：用于处理检测到的物体，并返回包含物体和标签信息的列表。
    @方法参数：
        detected_objects：一个包含检测到的物体数据的数据框（DataFrame）。
        img：输入的图像。
        min_confidence：最小置信度阈值，用于筛选检测结果。
    '''

    def process_detections(self, detected_objects, img, min_confidence):
        counter = 1
        overlapping_objects = []
        for index, row in detected_objects.iterrows():
            if row['confidence'] > min_confidence and row['class'] == 0:
                cimage = img[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]
                r = self.core.start(img=cimage, target="XiangTi")
                overlapping_info = {
                    'goods_id': counter,
                    'xmin': row['xmin'],
                    'ymin': row['ymin'],
                    'xmax': row['xmax'],
                    'ymax': row['ymax'],
                    'state': r,
                    'labelingood': []
                }
                counter += 1
                normal_labels = []
                dismantle_labels = []
                for _, label_row in detected_objects.iterrows():
                    if label_row['class'] == 1 and label_row['confidence'] >= min_confidence:
                        if (row['xmin'] < label_row['xmax'] and row['xmax'] > label_row['xmin'] and
                                row['ymin'] < label_row['ymax'] and row['ymax'] > label_row['ymin']):
                            cropped_image = img[int(label_row['ymin']):int(label_row['ymax']),
                                            int(label_row['xmin']):int(label_row['xmax'])]
                            label_text = self.recognize_text_paddleocr(cropped_image)
                            k = label_text.replace(" ", "")
                            if not label_text or len(k) < 5:
                                label_text = self.crop_and_ocr(cropped_image)
                                label_type = '0'  # 拆托标签
                            else:
                                label_text = self.recognize_text_paddleocr(cropped_image)
                                label_text = self.format_extracted_number(label_text)
                                label_text = label_text.replace(".", "")
                                label_type = '1'  # 正常标签
                            label_info = {
                                'label_id': counter,
                                'xmin': label_row['xmin'],
                                'ymin': label_row['ymin'],
                                'xmax': label_row['xmax'],
                                'ymax': label_row['ymax'],
                                'ocr_result': label_text if label_text else "",
                            }
                            if label_type == '1':
                                normal_labels.append(label_info)
                            else:
                                dismantle_labels.append(label_info)
                            counter += 1
                # 过滤标签信息，确保每个货物最多保留一个正常货物标签和一个拆托标签
                filtered_labels = []
                if normal_labels:
                    normal_labels.sort(key=lambda x: len(x['ocr_result']), reverse=True)
                    filtered_labels.append(normal_labels[0])  # 保留最长的正常货物标签
                if dismantle_labels:
                    filtered_labels.append(dismantle_labels[0])  # 保留一个拆托标签
                overlapping_info['labelingood'] = filtered_labels
                overlapping_objects.append(overlapping_info)
        return overlapping_objects

    '''
    @方法作用：用于格式化提取的数字。
    @方法参数：
        text：一个包含文本的字符串。
    '''
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

    '''
    @方法作用：使用PaddleOCR对图像中的文本进行识别并提取
        调用PaddleOCR的OCR方法对图像进行文本识别。
        如果没有识别出任何文本，返回空字符串。
        提取识别结果中的所有文本行，并将其存储在列表中。
        将提取出的所有文本行合并成一个字符串，并返回。
    @方法参数：
        image：一个图像对象，用于进行OCR（光学字符识别）。
    '''
    # paddle识别
    def recognize_text_paddleocr(self, image):
        result = self.ocr.ocr(image, cls=True)
        if result[0] is None:
            return ""
        all_texts = []
        for res in result:
            if res is not None:
                for line in res:
                    all_texts.append(line[1][0])
        combined_text = ' '.join(all_texts)
        return combined_text

    def draw_detections(self, image, detections):
        for detection in detections:
            xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(
                detection['ymax'])
            label = str(detection['state'])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            for label_info in detection['labelingood']:
                lxmin, lymin, lxmax, lymax = int(label_info['xmin']), int(label_info['ymin']), int(
                    label_info['xmax']), int(label_info['ymax'])
                ocr_result = str(label_info['ocr_result'])
                cv2.rectangle(image, (lxmin, lymin), (lxmax, lymax), (255, 0, 0), 2)
                cv2.putText(image, ocr_result, (lxmin, lymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return image


if __name__ == '__main__':
    detector = ZheDangDetector()
    detector.load_model(["../weights/zhedangold.pt",
                         "../weights/shouxie.pt",
                         "../detectors/CargoLabel/cv_convnextTiny_ocr-recognition-handwritten_damo"])
    result = detector.detect_zhedang_label('../zhedang12/11.jpg')

    image = cv2.imread('../zhedang12/11.jpg')
    drawn_image = detector.draw_detections(image, result)

    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()