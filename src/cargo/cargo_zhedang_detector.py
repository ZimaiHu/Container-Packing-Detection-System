import cv2
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import List, Dict, Tuple
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from paddleocr import PaddleOCR
from ultralytics import YOLO

# 设置日志级别
logging.getLogger("ppocr").setLevel(logging.ERROR)

# 设置环境变量
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ZheDangDetector:
    def __init__(self):
        pass

    def load_model(self, model_paths: List[str]):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # 使用循环加载YOLO模型
        self.models = {
            'zhedang': YOLO(model_paths[0]),
            'shouxie': YOLO(model_paths[1]),
            'guanjianzi': YOLO(model_paths[2]),
            'xiangTi': YOLO(model_paths[3])
        }
        for model in self.models.values():
            model.predict(_dummy_image, verbose=False)

        # 加载OCR模型
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_mem=8000,
                             det_model_dir=model_paths[4])
        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model=model_paths[5])

    def detect_zhedang_label(self, img_path: str, min_confidence: float = 0.65) -> List[Dict]:
        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        results = self.models['zhedang'].predict(source=image, show=False, device=0, save=False, verbose=False)
        detected_objects = self._process_yolo_results(results[0])

        box_positions, label_positions = self._extract_object_info(detected_objects, min_confidence)

        overlapping_objects = self.detect_overlap_ocr(box_positions, label_positions, image)
        results = self._convert_format(overlapping_objects)

        # 对每个物品进行状态检测
        for item in results:
            if 'goods_id' in item:
                ymin, ymax = max(0, int(item['ymin'])), min(height, max(0, int(item['ymax'])))
                xmin, xmax = max(0, int(item['xmin'])), min(width, max(0, int(item['xmax'])))
                crop_img = image[ymin:ymax, xmin:xmax]
                states = self.models['xiangTi'].predict(source=crop_img, show=False, save=False)[0]
                xiang_list = states.boxes.cls.tolist()  # 托盘好坏
                if len(xiang_list) == 0:
                    item['state'] = 0
                elif int(xiang_list[0]) == 1:
                    item['state'] = 0
                else:
                    item['state'] = 1
        return results

    def _process_yolo_results(self, result) -> pd.DataFrame:
        coordinates = result.obb.xyxyxyxy.tolist()
        confidences = result.obb.conf.tolist()
        classes = result.obb.cls.tolist()
        data = [
            coord[0] + coord[1] + coord[2] + coord[3] + [confidences[i], int(classes[i])]
            for i, coord in enumerate(coordinates)
        ]
        return pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'confidence', 'class'])

    def _extract_object_info(self, detected_objects: pd.DataFrame, min_confidence: float) -> Tuple[
        List[Dict], List[Dict]]:
        box_positions = []
        label_positions = []

        for index, row in detected_objects.iterrows():
            if row['confidence'] > min_confidence:
                object_info = {
                    'coordinates': tuple(row[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']]),
                    'confidence': row['confidence']
                }

                if row['class'] == 0:
                    object_info['goods_id'] = index
                    box_positions.append(object_info)
                elif row['class'] == 1:
                    object_info['label_id'] = index
                    label_positions.append(object_info)

        return box_positions, label_positions

    def detect_overlap_ocr(self, boxes: List[Dict], labels: List[Dict], img: np.ndarray) -> List[Dict]:
        overlapping_objects = []
        id_counter = 1

        for box in boxes:
            box_polygon = Polygon(self._coordinates_to_points(box['coordinates']))
            overlapping_info = {
                'goods_id': id_counter,
                'box_coordinates': box['coordinates'],
                'labels': []
            }
            id_counter += 1

            for label in labels:
                label_polygon = Polygon(self._coordinates_to_points(label['coordinates']))
                if box_polygon.intersects(label_polygon):
                    bounds = label_polygon.bounds
                    label_image = img[int(bounds[1]):int(bounds[3]), int(bounds[0]):int(bounds[2])]
                    label_text, label_type = self._process_label(label_image)
                    label_info = {
                        'label_id': id_counter,
                        'label_coordinates': label['coordinates'],
                        'ocr_result': label_text if label_text else "",
                        'label_type': label_type
                    }
                    overlapping_info['labels'].append(label_info)
                    id_counter += 1

            overlapping_objects.append(overlapping_info)

        return overlapping_objects

    def _coordinates_to_points(self, coordinates: Tuple) -> List[Tuple[float, float]]:
        return [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]

    def _process_label(self, cropped_image: np.ndarray) -> Tuple[str, str]:
        label_text = self.recognize_text_paddleocr(cropped_image)
        k = label_text.replace(" ", "")
        if not label_text or len(k) < 5:
            label_text = self._crop_and_ocr(cropped_image)
            label_type = '0'  # 拆托标签
        else:
            label_text = self._format_extracted_number(label_text).replace(".", "")
            label_type = '1'  # 正常标签
            if not label_text:
                label_text = self._process_guanjianzi(cropped_image)
        label_text = ''.join(re.findall(r'\d+', label_text))
        return label_text, label_type

    def _crop_and_ocr(self, image: np.ndarray) -> str:
        results = self.models['shouxie'].predict(source=image, show=False, save=False, verbose=False)[0]
        coord_list = results.boxes.xyxy.tolist()
        if coord_list:
            x1, y1, x2, y2 = map(int, coord_list[0])
            cropped_image = image[y1:y2, x1:x2]
            result = self.ocr_recognition(cropped_image)
            return result['text'][0] if 'text' in result and result['text'] else ""
        return ""

    def _format_extracted_number(self, text: str) -> str:
        parts = text.split()
        valid_numbers = []
        dot_part = None
        for part in parts:
            number = ''.join(filter(str.isdigit, part))
            if len(number) == 8:
                valid_numbers.append(part)
                if '.' in part:
                    dot_part = part
        if len(valid_numbers) == 2 and dot_part:
            return dot_part
        return ' '.join([''.join(filter(str.isdigit, part)) for part in valid_numbers])

    def _process_guanjianzi(self, image: np.ndarray) -> str:
        results = self.models['guanjianzi'].predict(source=image, show=False, save=False, verbose=False)[0]
        coord_list = results.boxes.xyxy.tolist()
        if coord_list:
            x1, y1, x2, y2 = map(int, coord_list[0])
            guanjianzi_img = image[y1:y2, x1:x2]
            return self.ocr_recognition(guanjianzi_img)['text'][0].replace(".", "")
        return ""

    def recognize_text_paddleocr(self, image: np.ndarray) -> str:
        result = self.ocr.ocr(image, cls=True)
        if not result[0]:
            return ""
        return ' '.join(line[1][0] for res in result if res for line in res)

    def _convert_format(self, original_data: List[Dict]) -> List[Dict]:
        return [
            {
                'goods_id': item['goods_id'],
                'xmin': min(item['box_coordinates'][0::2]),
                'ymin': min(item['box_coordinates'][1::2]),
                'xmax': max(item['box_coordinates'][0::2]),
                'ymax': max(item['box_coordinates'][1::2]),
                'state': 1,
                'labelingood': [
                    {
                        'label_id': label['label_id'],
                        'xmin': min(label['label_coordinates'][0::2]),
                        'ymin': min(label['label_coordinates'][1::2]),
                        'xmax': max(label['label_coordinates'][0::2]),
                        'ymax': max(label['label_coordinates'][1::2]),
                        'ocr_result': label['ocr_result']
                    } for label in item['labels']
                ]
            } for item in original_data
        ]

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for detection in detections:
            # Draw box
            xmin, ymin, xmax, ymax = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw labels
            for label in detection['labelingood']:
                label_xmin, label_ymin, label_xmax, label_ymax = map(int, [label['xmin'], label['ymin'], label['xmax'], label['ymax']])
                cv2.rectangle(image, (label_xmin, label_ymin), (label_xmax, label_ymax), (255, 0, 0), 2)
                cv2.putText(image, str(label['ocr_result']), (label_xmin, label_ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw goods_id
            cv2.putText(image, f"ID: {detection['goods_id']}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image

if __name__ == '__main__':
    detector = ZheDangDetector()
    detector.load_model([
        "../weights/zhedang.pt",
        "../weights/shouxie.pt",
        "../weights/guanjianzi.pt",
        "../detectors/CargoLabel/cv_convnextTiny_ocr-recognition-handwritten_damo"
    ])

    result = detector.detect_zhedang_label('../ceshitu/zhedang3.jpg')

    image = cv2.imread('../ceshitu/zhedang3.jpg')
    drawn_image = detector.draw_detections(image, result)
    cv2.imwrite('high_quality_output.jpg', drawn_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()