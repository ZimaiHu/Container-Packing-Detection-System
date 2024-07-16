import os
import cv2
import logging
import numpy as np
from detect.AlgorithmManager import AlgorithmManager
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from paddleocr import PaddleOCR
from ultralytics import YOLO

logging.getLogger("ppocr").setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ZheDangDetector:
    def __init__(self):
        self.core = AlgorithmManager()

    def load_model(self, model_paths):
        self.model_zhedang = self._load_yolo_model(model_paths[0])
        self.model_shouxie = self._load_yolo_model(model_paths[1])
        self.model_guanjianzi = self._load_yolo_model(model_paths[2])
        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model=model_paths[3])
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_mem=8000,
                             det_model_dir='weights/ch_PP-OCRv4_det_infer')

    def _load_yolo_model(self, model_path):
        model = YOLO(model_path)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy_image, verbose=False)
        return model

    def detect_zhedang_label(self, img_path, min_confidence=0.5):
        image = cv2.imread(img_path)
        results = self.model_zhedang.predict(source=image, show=False, device=0, save=False, verbose=False)

        detected_objects = self._process_yolo_results(results[0])

        box_positions = self._extract_object_info(detected_objects, min_confidence, class_id=0)
        label_positions = self._extract_object_info(detected_objects, min_confidence, class_id=1)

        overlapping_objects = self.detect_overlap_ocr(box_positions, label_positions, image)
        results = self.convert_format(overlapping_objects)

        for item in results:
            if 'goods_id' in item:
                crop_img = image[int(item['ymin']):int(item['ymax']), int(item['xmin']):int(item['xmax'])]
                item['state'] = self.core.start(img=crop_img, target="XiangTi")

        print("results", results)
        return results

    def _process_yolo_results(self, result):
        return [
            coord + [conf, int(cls)]
            for coord, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ]

    def _extract_object_info(self, detected_objects, min_confidence, class_id):
        return [
            {
                f'{"goods" if class_id == 0 else "label"}_id': i,
                'coordinates': tuple(obj[:4]),
                'confidence': obj[4]
            }
            for i, obj in enumerate(detected_objects)
            if obj[4] > min_confidence and int(obj[5]) == class_id
        ]

    def detect_overlap_ocr(self, boxes, labels, img):
        overlapping_objects = []
        id_counter = 1

        for box in boxes:
            box_coords = box['coordinates']
            overlapping_info = {
                'goods_id': id_counter,
                'box_coordinates': box_coords,
                'labels': []
            }
            id_counter += 1

            for label in labels:
                label_coords = label['coordinates']
                if self._is_overlapping(box_coords, label_coords):
                    label_image = self._crop_image(img, label_coords)
                    label_text, label_type = self.process_label(label_image)
                    label_info = {
                        'label_id': id_counter,
                        'label_coordinates': label_coords,
                        'ocr_result': label_text if label_text else "",
                        'label_type': label_type
                    }
                    overlapping_info['labels'].append(label_info)
                    id_counter += 1

            overlapping_objects.append(overlapping_info)

        return overlapping_objects

    def _is_overlapping(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

    def _crop_image(self, img, coords):
        x1, y1, x2, y2 = map(int, coords)
        return img[y1:y2, x1:x2]

    def process_label(self, cropped_image):
        label_text = self.recognize_text_paddleocr(cropped_image)
        k = label_text.replace(" ", "")
        if not label_text or len(k) < 5:
            label_text = self.crop_and_ocr(cropped_image)
            label_type = '0'  # 拆托标签
        else:
            label_text = self.format_extracted_number(label_text)
            label_text = label_text.replace(".", "")
            label_type = '1'  # 正常标签
            if not label_text:
                label_text = self._process_guanjianzi(cropped_image)
        return label_text, label_type

    def _process_guanjianzi(self, image):
        results = self.model_guanjianzi.predict(source=image, show=False, save=False, verbose=False)[0]
        coord_list = results.boxes.xyxy.tolist()
        if coord_list:
            x1, y1, x2, y2 = map(int, coord_list[0])
            guanjianzi_img = image[y1:y2, x1:x2]
            return self.ocr_recognition(guanjianzi_img)['text'][0].replace(".", "")
        return ""

    def crop_and_ocr(self, image):
        results = self.model_shouxie.predict(source=image, show=False, save=False, verbose=False)[0]
        coord_list = results.boxes.xyxy.tolist()
        if coord_list:
            x1, y1, x2, y2 = map(int, coord_list[0])
            cropped_image = image[y1:y2, x1:x2]
            return self.ocr_recognition(cropped_image)['text'][0]
        return ""

    def format_extracted_number(self, text):
        import re
        parts = text.split()
        valid_numbers = [part for part in parts if len(''.join(re.findall(r'\d', part))) == 8]
        dot_part = next((part for part in valid_numbers if '.' in part), None)

        if len(valid_numbers) == 2 and dot_part:
            return dot_part
        return ' '.join([''.join(re.findall(r'\d', part)) for part in valid_numbers])

    def recognize_text_paddleocr(self, image):
        result = self.ocr.ocr(image, cls=True)
        if not result[0]:
            return ""
        return ' '.join([line[1][0] for res in result if res for line in res])

    def convert_format(self, original_data):
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

    def draw_detections(self, image, detections):
        for detection in detections:
            self._draw_box(image, detection, (0, 255, 0), f"ID: {detection['goods_id']}")
            for label in detection['labelingood']:
                self._draw_box(image, label, (255, 0, 0), str(label['ocr_result']))
        return image

    def _draw_box(self, image, obj, color, text):
        cv2.rectangle(image, (int(obj['xmin']), int(obj['ymin'])), (int(obj['xmax']), int(obj['ymax'])), color, 2)
        cv2.putText(image, text, (int(obj['xmin']), int(obj['ymin']) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

if __name__ == '__main__':
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    detector = ZheDangDetector()
    detector.load_model([
        "../weights/zhedang1400.pt",
        "../weights/shouxie.pt",
        "../weights/guanjianzi.pt",
        "../detectors/CargoLabel/cv_convnextTiny_ocr-recognition-handwritten_damo"
    ])
    result = detector.detect_zhedang_label('../ceshitu/zhedang3.jpg')

    image = cv2.imread('../ceshitu/zhedang3.jpg')
    drawn_image = detector.draw_detections(image, result)
    cv2.imwrite('high_quality_output.jpg', drawn_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()