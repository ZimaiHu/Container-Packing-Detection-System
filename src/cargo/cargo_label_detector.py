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
        # print("box_positions", box_positions)
        # print("label_positions", label_positions)

        overlapping_objects = self.detect_overlap_ocr(box_positions, label_positions, image)
        results = self._convert_format(overlapping_objects)

        # Detect state for each item
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
        # 现在逻辑是：检测箱体，然后挨个看标签属于哪个箱体，一个标签只能属于一个箱体，如果检测过的标签，
        # 则下个箱体时就不检测了。然后如果一个箱体无标签，则不框了，说明算法有错误，误判了框体。
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
                # if box_polygon.contains(label_polygon):  # 只检查完全包含
                if box_polygon.intersects(label_polygon):  # 判断是否相交
                    overlap_ratio = box_polygon.intersection(label_polygon).area / label_polygon.area  # 计算重合度

                    if overlap_ratio >= overlap_threshold:
                        bounds = label_polygon.bounds
                        label_image = img[int(bounds[1]):int(bounds[3]), int(bounds[0]):int(bounds[2])]
                        label_text, label_type = self._process_label(label_image)

                        # 跳过 label_type 为 '2' 的标签
                        if label_type == '2':
                            continue  # 如果是字母标签，不添加这个框

                        label_info = {
                            'label_id': id_counter,
                            'label_coordinates': label['coordinates'],
                            'ocr_result': label_text if label_text else "",
                            'label_type': label_type
                        }
                        overlapping_info['labels'].append(label_info)

                        assigned_labels.add(label_id)
                        id_counter += 1


            if overlapping_info['labels']:
                # 如果检测到有标签，直接添加到结果列表
                overlapping_objects.append(overlapping_info)
            else:
                # 如果没有检测到标签，检查框体是否与任何标签重叠
                box_polygon = Polygon(self._coordinates_to_points(box['coordinates']))

                has_overlap = any(
                    box_polygon.intersects(Polygon(self._coordinates_to_points(label['coordinates'])))
                    for label in labels
                )

                if not has_overlap:
                    # 如果没有重叠，说明标签没检测到，那就单独把框框出来吧。
                    overlapping_objects.append(overlapping_info)

        return overlapping_objects

    # def detect_overlap_ocr(self, boxes, labels, img):
    #     overlapping_objects = []
    #     id_counter = 1  # 用于给 goods 和 labels 分配连续的 ID
    #
    #     for box in boxes:
    #         box_polygon = Polygon(
    #             [(box['coordinates'][i], box['coordinates'][i + 1]) for i in range(0, len(box['coordinates']), 2)]
    #         )
    #         overlapping_info = {
    #             'goods_id': id_counter,
    #             'box_coordinates': box['coordinates'],
    #             'labels': []
    #         }
    #         id_counter += 1  # 增加计数器为下一个 ID 做准备
    #         normal_labels = []
    #         dismantle_labels = []
    #         for label in labels:
    #             # 创建标签多边形
    #             label_polygon = Polygon(
    #                 [(label['coordinates'][i], label['coordinates'][i + 1]) for i in
    #                  range(0, len(label['coordinates']), 2)]
    #             )
    #             # 检查重叠
    #             if box_polygon.intersects(label_polygon):
    #                 # 截取标签所在图像区域进行OCR
    #                 bounds = label_polygon.bounds
    #                 label_image = img[int(bounds[1]):int(bounds[3]), int(bounds[0]):int(bounds[2])]
    #                 label_text, label_type = self.process_label(label_image)
    #                 label_info = {
    #                     'label_id': id_counter,
    #                     'label_coordinates': label['coordinates'],
    #                     'ocr_result': label_text if label_text else "",
    #                     'label_type': label_type
    #                 }
    #                 if label_type == '1':
    #                     normal_labels.append(label_info)
    #                 else:
    #                     dismantle_labels.append(label_info)
    #                 id_counter += 1  # 每添加一个标签，增加计数器
    #         filtered_labels = []
    #         if normal_labels:
    #             normal_labels.sort(key=lambda x: len(x['ocr_result']), reverse=True)
    #             filtered_labels.append(normal_labels[0])  # 保留最长的正常货物标签
    #         if dismantle_labels:
    #             filtered_labels.append(dismantle_labels[0])  # 保留一个拆托标签
    #         overlapping_info['labels'] = filtered_labels
    #         overlapping_objects.append(overlapping_info)
    #     return overlapping_objects

    def _coordinates_to_points(self, coordinates: Tuple) -> List[Tuple[float, float]]:
        return [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[1]),
                (coordinates[2], coordinates[3]), (coordinates[0], coordinates[3])]

    def _process_label(self, cropped_image: np.ndarray) -> Tuple[str, str]:
        label_text = self.recognize_text_paddleocr(cropped_image)
        k = label_text.replace(" ", "")
        if not label_text or len(k) < 5:
            # Perform a wine glass label check before setting to 拆托标签
            if len(k) == 1 and re.match(r'[A-Za-z]', k):
                label_type = '2'  # 字母标签
                label_text = ''  # No need for label text, as it's identified by the symbol
            else:
                label_type = '0'  # 拆托标签
                label_text = self._crop_and_ocr(cropped_image)
            # label_type = '0'  # 拆托标签
            # label_text = self._crop_and_ocr(cropped_image)
        else:
            label_type = '1'  # 正常标签
            label_text = self._format_extracted_number(label_text).replace(".", "")
            if not label_text or len(label_text) > 8: # 标识有多个字符
                label_text = self._process_guanjianzi(cropped_image)
        label_text = ''.join(re.findall(r'\d+', label_text))
        return label_text, label_type

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

        except BaseException :
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
                    valid_numbers = [part]  # 重置 valid_numbers 只包含带点的部分
                    break  # 找到带点的部分后立即退出循环
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
        except BaseException :
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
                        'ocr_result': label['ocr_result']
                    } for label in item['labels']
                ]
            } for item in original_data
        ]

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for detection in detections:
            # Draw box
            xmin, ymin, xmax, ymax = map(int,
                                         [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw labels
            for label in detection['labelingood']:
                label_xmin, label_ymin, label_xmax, label_ymax = map(int, [label['xmin'], label['ymin'], label['xmax'],
                                                                           label['ymax']])
                cv2.rectangle(image, (label_xmin, label_ymin), (label_xmax, label_ymax), (255, 0, 0), 2)
                cv2.putText(image, str(label['ocr_result']), (label_xmin, label_ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw goods_id
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

    # 将 PaddleOCR 对象和手写识别模型传递给 load_ocr_model
    detector.load_ocr_model([paddle_ocr, handwritten_recognition_model])

    result = detector.detect_cargo_label('../../ceshitu/fault/zhedang1.jpg')
    print("result", result)

    image = cv2.imread('../../ceshitu/fault/zhedang1.jpg')
    drawn_image = detector.draw_detections(image, result)
    cv2.imwrite('high_quality_output.jpg', drawn_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()