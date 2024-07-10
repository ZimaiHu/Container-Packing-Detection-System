from PIL import ImageFont
import os
import re
import cv2
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

logging.getLogger("ppocr").setLevel(logging.ERROR)

from detect.AlgorithmManager import AlgorithmManager
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from paddleocr import PaddleOCR
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ZheDangDetector:
    def __init__(self):
        # 初始化模型
        self.core = AlgorithmManager()

    def load_model(self, model_path):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # 加载遮挡模型
        self.model_zhedang = YOLO(model_path[0])        #改动
        self.model_zhedang.predict(_dummy_image, verbose=False)

        # 加载手写模型
        self.model_shouxie = YOLO(model_path[1])
        self.model_shouxie.predict(_dummy_image, verbose=False)

        # 加载关键字模型
        self.model_guanjianzi = YOLO(model_path[2])
        self.model_guanjianzi.predict(_dummy_image, verbose=False)

        # 加载ocr识别模型
        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model=model_path[3])
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_mem=8000,
                             det_model_dir='weights/ch_PP-OCRv4_det_infer')
    # 主探测函数
    def detect_zhedang_label(self, img_path, min_confidence=0.65):
        image = cv2.imread(img_path)
        height, width = image.shape[:2]  # 获取图像的高度和宽度

        results = self.model_zhedang.predict(source=image, show=False, device=0, save=False, verbose=False)
        coordinates = results[0].obb.xyxyxyxy.tolist()
        confidences = results[0].obb.conf.tolist()
        classes = results[0].obb.cls.tolist()
        data = [
            coord[0] + coord[1] + coord[2] + coord[3] + [confidences[i], int(classes[i])]
            for i, coord in enumerate(coordinates)
        ]
        detected_goods_and_labels = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4',
                                                                'confidence', 'class'])

        box_positions = self.extract_box_info(detected_goods_and_labels, min_confidence)
        label_positions = self.extract_label_info(detected_goods_and_labels, min_confidence)

        overlapping_objects = self.detect_overlap_ocr(box_positions, label_positions, image)

        results = self.convert_format(overlapping_objects)

        # for item in results:
        #     if 'goods_id' in item:
        # # 从整个图像中裁剪出该物品的区域
        # crop_img = image[int(item['ymin']):int(item['ymax']), int(item['xmin']):int(item['xmax'])]
        # item['state'] = self.core.start(img=crop_img, target="XiangTi")

        # 对每个物品进行状态检测
        for item in results:
            if 'goods_id' in item:
                # 确保坐标不为负值，且不超出图像边界
                ymin = max(0, int(item['ymin']))
                ymax = min(height, max(0, int(item['ymax'])))
                xmin = max(0, int(item['xmin']))
                xmax = min(width, max(0, int(item['xmax'])))

                # 从整个图像中裁剪出该物品的区域
                crop_img = image[ymin:ymax, xmin:xmax]
                item['state'] = self.core.start(img=crop_img, target="XiangTi")
        return results

    def extract_box_info(self, detected_objects, min_confidence):
        box = []
        for index, row in detected_objects.iterrows():
            if row['confidence'] > min_confidence and row['class'] == 0:
                box_info = {
                    'goods_id': index,
                    'coordinates': (
                    row['x1'], row['y1'], row['x2'], row['y2'], row['x3'], row['y3'], row['x4'], row['y4']),
                    'confidence': row['confidence']
                }
                box.append(box_info)
        return box

    def extract_label_info(self, detected_objects, min_confidence):
        label = []
        for index, row in detected_objects.iterrows():
            if row['confidence'] >= min_confidence and row['class'] == 1:
                label_info = {
                    'label_id': index,
                    'coordinates': (
                    row['x1'], row['y1'], row['x2'], row['y2'], row['x3'], row['y3'], row['x4'], row['y4']),
                    'confidence': row['confidence']
                }
                label.append(label_info)
        return label

    def detect_overlap_ocr(self, boxes, labels, img):
        overlapping_objects = []
        id_counter = 1  # 用于给 goods 和 labels 分配连续的 ID

        for box in boxes:
            # 创建箱体多边形
            box_polygon = Polygon(
                [(box['coordinates'][i], box['coordinates'][i + 1]) for i in range(0, len(box['coordinates']), 2)]
            )
            overlapping_info = {
                'goods_id': id_counter,
                'box_coordinates': box['coordinates'],
                'labels': []
            }
            id_counter += 1  # 增加计数器为下一个 ID 做准备

            for label in labels:
                # 创建标签多边形
                label_polygon = Polygon(
                    [(label['coordinates'][i], label['coordinates'][i + 1]) for i in
                     range(0, len(label['coordinates']), 2)]
                )
                # 检查重叠
                if box_polygon.intersects(label_polygon):
                    # 截取标签所在图像区域进行OCR
                    bounds = label_polygon.bounds
                    label_image = img[int(bounds[1]):int(bounds[3]), int(bounds[0]):int(bounds[2])]
                    label_text, label_type = self.process_label(label_image)
                    label_info = {
                        'label_id': id_counter,
                        'label_coordinates': label['coordinates'],
                        'ocr_result': label_text if label_text else "",
                        'label_type': label_type
                    }
                    overlapping_info['labels'].append(label_info)
                    id_counter += 1  # 每添加一个标签，增加计数器

            overlapping_objects.append(overlapping_info)

        return overlapping_objects

    def process_label(self, cropped_image):
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
            if len(label_text) == 0:
                results = self.model_guanjianzi.predict(source=cropped_image, show=False, save=False, verbose=False)[0]
                coord_list = results.boxes.xyxy.tolist()
                if coord_list:  # 确保coord_list不为空
                    x1, y1, x2, y2 = map(int, coord_list[0])
                    guanjianzi_img = cropped_image[y1:y2, x1:x2]
                    label_text = self.ocr_recognition(guanjianzi_img)['text'][0]
                    label_text = label_text.replace(".", "")
                else:
                    label_text = ""
        return label_text, label_type

    # 手写识别
    def crop_and_ocr(self, image):
        # 使用手写识别模型进行预测
        results = self.model_shouxie.predict(source=image, show=False, save=False, verbose=False)[0]

        coord_list = results.boxes.xyxy.tolist()
        if coord_list:  # 确保coord_list不为空
            x1, y1, x2, y2 = map(int, coord_list[0])
            cropped_image = image[y1:y2, x1:x2]
            figure_img = cropped_image
            result = self.ocr_recognition(figure_img)
            return result['text'][0] if 'text' in result and result['text'] else ""
        else:
            return ""

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

    def convert_format(self, original_data):
        converted_data = []

        for item in original_data:
            # 提取商品框坐标
            box_coords = item['box_coordinates']
            xmin = min(box_coords[0::2])  # 偶数索引为x坐标
            ymin = min(box_coords[1::2])  # 奇数索引为y坐标
            xmax = max(box_coords[0::2])
            ymax = max(box_coords[1::2])

            # 创建新的商品字典
            new_item = {
                'goods_id': item['goods_id'],
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'state': 1,  # 假设所有商品状态为1
                'labelingood': []
            }

            # 处理标签
            for label in item['labels']:
                label_coords = label['label_coordinates']
                new_label = {
                    'label_id': label['label_id'],
                    'xmin': min(label_coords[0::2]),
                    'ymin': min(label_coords[1::2]),
                    'xmax': max(label_coords[0::2]),
                    'ymax': max(label_coords[1::2]),
                    'ocr_result': label['ocr_result']
                }
                new_item['labelingood'].append(new_label)

            converted_data.append(new_item)

        return converted_data

    def draw_detections(self, image, detections):
        for detection in detections:
            # Draw box
            xmin, ymin, xmax, ymax = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Draw labels
            for label in detection['labelingood']:
                label_xmin, label_ymin, label_xmax, label_ymax = label['xmin'], label['ymin'], label['xmax'], label[
                    'ymax']
                cv2.rectangle(image, (int(label_xmin), int(label_ymin)), (int(label_xmax), int(label_ymax)),
                              (255, 0, 0), 2)

                ocr_result = str(label['ocr_result'])
                cv2.putText(image, ocr_result, (int(label_xmin), int(label_ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw goods_id
            goods_id = str(detection['goods_id'])
            cv2.putText(image, f"ID: {goods_id}", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image


    # def draw_detections_old(self, image, detections):
    #     for detection in detections:
    #         # Draw box
    #         box_coords = detection['box_coordinates']
    #         pts = np.array([(box_coords[i], box_coords[i + 1]) for i in range(0, len(box_coords), 2)], np.int32)
    #         pts = pts.reshape((-1, 1, 2))
    #         cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    #
    #         # Draw labels
    #         for label in detection['labels']:
    #             label_coords = label['label_coordinates']
    #             label_pts = np.array([(label_coords[i], label_coords[i + 1]) for i in range(0, len(label_coords), 2)],
    #                                  np.int32)
    #             label_pts = label_pts.reshape((-1, 1, 2))
    #             cv2.polylines(image, [label_pts], isClosed=True, color=(255, 0, 0), thickness=2)
    #
    #             ocr_result = str(label['ocr_result'])
    #             cv2.putText(image, ocr_result, (int(label_coords[0]), int(label_coords[1]) - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #
    #     return image

if __name__ == '__main__':

    logging.getLogger("ppocr").setLevel(logging.ERROR)
    detector = ZheDangDetector()
    detector.load_model(["../weights/zhedang.pt",
                         "../weights/shouxie.pt",
                         "../weights/guanjianzi.pt",
                         "../detectors/CargoLabel/cv_convnextTiny_ocr-recognition-handwritten_damo"])
    result = detector.detect_zhedang_label('../zhedang12/3.jpg')

    image = cv2.imread('../zhedang12/3.jpg')
    drawn_image = detector.draw_detections(image, result)
    cv2.imwrite('high_quality_output.jpg', drawn_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()