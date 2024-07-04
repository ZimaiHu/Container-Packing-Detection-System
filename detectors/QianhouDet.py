from PIL import ImageFont
import os
import re
import logging
import numpy as np
logging.getLogger('ppocr').setLevel(logging.WARNING)
from detect.AlgorithmManager import AlgorithmManager
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class QianhouDetector:
    def __init__(self):
        # 初始化模型
        self.core = AlgorithmManager()
    def load_model(self, model_path):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model_cargo = YOLO(model_path[0])
        self.model_cargo.predict(_dummy_image, verbose=False)
        self.model_shouxie = YOLO(model_path[1])
        self.model_shouxie.predict(_dummy_image, verbose=False)
        self.model_guanjianzi = YOLO("weights/guanjianzi.pt")
        self.model_guanjianzi.predict(_dummy_image, verbose=False)
        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model=model_path[2])
#主探测函数
    def detect_qianhou(self, img_path, min_confidence=0.65):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_mem=8000,det_model_dir='weights/ch_PP-OCRv4_det_infer')
        image = cv2.imread(img_path)
        results = self.model_cargo.predict(source=image, show=False, save=False, verbose=False)
        classes = results[0].boxes.cls.tolist()
        coordinates = results[0].boxes.xyxy.tolist()
        confidences = results[0].boxes.conf.tolist()
        data = []
        for i in range(len(classes)):
            data.append(coordinates[i] + [confidences[i], int(classes[i])])

        df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
        detected_goods_and_labels = df
        img = cv2.imread(img_path)

        results = self.process_detections(detected_goods_and_labels, img, min_confidence)
        for item in results:
            labelingood = item['labelingood']
            if len(labelingood) > 1:
                valid_labels = [label for label in labelingood if len(label['ocr_result']) > 3]
                if len(valid_labels) > 1:
                    labelingood = [max(valid_labels, key=lambda x: len(x['ocr_result']))]
            item['labelingood'] = labelingood
        return results
#手写识别
    def crop_and_ocr(self, image):
        # 使用手写识别模型进行预测
        results = self.model_shouxie.predict(source=image, show=False, save=False, verbose=False)[0]
        coord_list = results.boxes.xyxy.tolist()
        if coord_list:  # 确保coord_list不为空
            x1, y1, x2, y2 = map(int, coord_list[0])
            cropped_image = image[y1:y2, x1:x2]
            figure_img = cropped_image
            result = self.ocr_recognition(figure_img)
            return result['text'][0]
        else:
            return ""
#综合步骤
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
                                label_type = '0'#拆托标签
                            else:
                                label_text = self.recognize_text_paddleocr(cropped_image)
                                label_text = self.format_extracted_number(label_text)
                                label_text = label_text.replace(".", "")
                                label_type = '1'#正常标签
                                if len(label_text) == 0:
                                    print("进入框选")
                                    results = \
                                    self.model_guanjianzi.predict(source=cropped_image, show=False, save=False,
                                                                  verbose=False)[0]
                                    coord_list = results.boxes.xyxy.tolist()
                                    if coord_list:  # 确保coord_list不为空
                                        x1, y1, x2, y2 = map(int, coord_list[0])
                                        guanjianzi_img = cropped_image[y1:y2, x1:x2]
                                        # cv2.imwrite(f'guanjianzi_img{counter}.jpg', guanjianzi_img)
                                        label_text = self.ocr_recognition(guanjianzi_img)['text'][0]
                                        # label_text=self.recognize_text_paddleocr(guanjianzi_img)
                                        label_text = label_text.replace(".", "")
                                    else:
                                        label_text = ""
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
#正则变换
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
#paddle识别
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