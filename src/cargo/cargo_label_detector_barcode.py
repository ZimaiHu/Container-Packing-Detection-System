import os
import re
import cv2
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import List, Dict, Tuple
from ultralytics import YOLO
from pyzbar.pyzbar import decode  # 引入 pyzbar 库，用于条形码解码

# Set logging level
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CargoLabelDetector:
    def __init__(self):
        self.ocr_recognition = None
        self.ocr = None
        self.models = None

    def load_model(self, model_paths: List[str]):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Load YOLO models
        self.models = {
            'cargo': YOLO(model_paths[0]),
            'shouxie': YOLO(model_paths[1]),
            'guanjianzi': YOLO(model_paths[2]),
            'xiangTi': YOLO(model_paths[3])
        }
        for model in self.models.values():
            model.predict(_dummy_image, verbose=False)

    def load_ocr_model(self, model_obj):
        self.ocr = model_obj[0]
        self.ocr_recognition = model_obj[1]

    def detect_cargo_label(self, img_path: str, min_confidence: float = 0.8) -> List[Dict]:
        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        results = self.models['cargo'].predict(source=image, show=False, save=False, verbose=False)
        detected_objects = self._process_yolo_results(results[0])

        box_positions, label_positions = self._extract_object_info(detected_objects, min_confidence)

        overlapping_objects = self.detect_overlap_ocr(box_positions, label_positions, image)
        results = self._convert_format(overlapping_objects)

        # Detect state for each item
        for item in results:
            if 'goods_id' in item:
                ymin, ymax = max(0, int(item['ymin'])), min(height, max(0, int(item['ymax'])))
                xmin, xmax = max(0, int(item['xmin'])), min(width, max(0, int(item['xmax'])))
                crop_img = image[ymin:ymax, xmin:xmax]
                states = self.models['xiangTi'].predict(source=crop_img, show=False, save=False)[0]
                xiang_list = states.boxes.cls.tolist()
                if len(xiang_list) == 0:
                    item['state'] = 0
                elif int(xiang_list[0]) == 1:
                    item['state'] = 0
                else:
                    item['state'] = 1
        return results

    def _process_yolo_results(self, result) -> pd.DataFrame:
        return pd.DataFrame(
            [coord + [conf, int(cls)] for coord, conf, cls in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist()
            )],
            columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
        )

    def _extract_object_info(self, detected_objects: pd.DataFrame, min_confidence: float) -> Tuple[
        List[Dict], List[Dict]]:
        box_positions = []
        label_positions = []

        for index, row in detected_objects.iterrows():
            if row['confidence'] > min_confidence:
                object_info = {
                    'coordinates': (row['xmin'], row['ymin'], row['xmax'], row['ymax']),
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
        overlap_threshold = 0.8
        overlapping_objects = []
        assigned_labels = set()
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
                label_id = tuple(label['coordinates'])
                if label_id in assigned_labels:
                    continue

                label_polygon = Polygon(self._coordinates_to_points(label['coordinates']))
                if box_polygon.intersects(label_polygon):
                    overlap_ratio = box_polygon.intersection(label_polygon).area / label_polygon.area

                    if overlap_ratio >= overlap_threshold:
                        bounds = label_polygon.bounds
                        label_image = img[int(bounds[1]):int(bounds[3]), int(bounds[0]):int(bounds[2])]
                        label_text, label_type, barcode_text = self._process_label(label_image)

                        if label_type == '2':
                            continue

                        label_info = {
                            'label_id': id_counter,
                            'label_coordinates': label['coordinates'],
                            'ocr_result': label_text if label_text else "",
                            'label_type': label_type,
                            'barcode_result': barcode_text  # 新增的条形码内容
                        }
                        overlapping_info['labels'].append(label_info)

                        assigned_labels.add(label_id)
                        id_counter += 1

            if overlapping_info['labels']:
                overlapping_objects.append(overlapping_info)

        return overlapping_objects

    def _coordinates_to_points(self, coordinates: Tuple) -> List[Tuple[float, float]]:
        return [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[1]),
                (coordinates[2], coordinates[3]), (coordinates[0], coordinates[3])]

    def _process_label(self, cropped_image: np.ndarray) -> Tuple[str, str, str]:
        label_text = self.recognize_text_paddleocr(cropped_image)
        barcode_text = self._detect_barcode(cropped_image)  # 新增条形码检测
        k = label_text.replace(" ", "")
        label_type = '1' if label_text else '0'

        if not label_text or len(k) < 5:
            if len(k) == 1 and re.match(r'[A-Za-z]', k):
                label_type = '2'
                label_text = ''
            else:
                label_type = '0'
                label_text = self._crop_and_ocr(cropped_image)
        else:
            label_text = self._format_extracted_number(label_text).replace(".", "")
            if not label_text or len(label_text) > 8:
                label_text = self._process_guanjianzi(cropped_image)

        label_text = ''.join(re.findall(r'\d+', label_text))
        return label_text, label_type, barcode_text

    def _detect_barcode(self, image: np.ndarray) -> str:
        barcodes = decode(image)
        if barcodes:
            return barcodes[0].data.decode("utf-8")
        return ""

    def _crop_and_ocr(self, image: np.ndarray) -> str:
        try:
            results = self.models['shouxie'].predict(source=image, show=False, save=False, verbose=False)[0]
            coord_list = results.boxes.xyxy.tolist()
            if not coord_list:
                return ""
            x1, y1, x2, y2 = map(int, coord_list[0])
            cropped_image = image[y1:y2, x1:x2]
            result = self.ocr_recognition(cropped_image)
            return result['text'][0] if 'text' in result and result['text'] else ""

        except BaseException:
            return ""

    def _format_extracted_number(self, text: str) -> str:
        parts = text.split()
        valid_numbers = []
        dot_part = None
        for part in parts:
            number = ''.join(filter(str.isdigit, part))
            if len(number) == 8:
                if '.' in part:
                    dot_part = part
                    valid_numbers = [part]
                    break
                elif not dot_part:
                    valid_numbers.append(part)

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
        try:
            result = self.ocr.ocr(image, cls=True)
            if not result[0]:
                return ""
            return ' '.join(line[1][0] for res in result if res for line in res)
        except BaseException:
            return ""

    def _convert_format(self, original_data: List[Dict]) -> List[Dict]:
        return [
            {
                'goods_id': item['goods_id'],
                'xmin': item['box_coordinates'][0],
                'ymin': item['box_coordinates'][1],
                'xmax': item['box_coordinates'][2],
                'ymax': item['box_coordinates'][3],
                'state': 1,
                'labelingood': [
                    {
                        'label_id': label['label_id'],
                        'xmin': label['label_coordinates'][0],
                        'ymin': label['label_coordinates'][1],
                        'xmax': label['label_coordinates'][2],
                        'ymax': label['label_coordinates'][3],
                        'ocr_result': label['ocr_result'],
                        'barcode_result': label['barcode_result']  # 包含条形码内容
                    } for label in item['labels']
                ]
            } for item in original_data
        ]

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for detection in detections:
            xmin, ymin, xmax, ymax = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            for label in detection['labelingood']:
                label_xmin, label_ymin, label_xmax, label_ymax = map(int, [label['xmin'], label['ymin'], label['xmax'], label['ymax']])
                cv2.rectangle(image, (label_xmin, label_ymin), (label_xmax, label_ymax), (255, 0, 0), 2)
                cv2.putText(image, f"{label['ocr_result']} / {label['barcode_result']}", (label_xmin, label_ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(image, f"ID: {detection['goods_id']}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image


if __name__ == '__main__':
    from paddleocr import PaddleOCR
    from modelscope import pipeline, Tasks

    detector = CargoLabelDetector()
    detector.load_model([
        "../../weights/cargo/cargolabel.pt",
        "../../weights/cargo/shouxie.pt",
        "../../weights/cargo/guanjianzi.pt",
        "../../weights/cargo/huowuposun.pt"
    ])

    handwritten_recognition_model = pipeline(Tasks.ocr_recognition,
                                             model="../../weights/ocr/cv_convnextTiny_ocr-recognition-handwritten_damo")
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, use_mkldnn=False,
                           det_model_dir="../../weights/ocr/ch_PP-OCRv4_det_infer")

    detector.load_ocr_model([paddle_ocr, handwritten_recognition_model])

    result = detector.detect_cargo_label('../../ceshitu/ceshi/barcode.jpg')
    print("result", result)

    image = cv2.imread('../../ceshitu/ceshi/barcode.jpg')
    drawn_image = detector.draw_detections(image, result)
    cv2.imwrite('high_quality_output.jpg', drawn_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
