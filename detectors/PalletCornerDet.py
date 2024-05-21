from PIL import Image
from ultralytics import YOLO  # 使用 YOLOv8
from detect.AlgorithmManager import AlgorithmManager

class PalletDetector:
    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence

    def load_model(self, model_path):
        # 加载 YOLOv8 模型
        self.model = YOLO(model_path[0])  # 修改为直接加载 YOLOv8 模型
        self.core = AlgorithmManager()

    # 主探测函数
    def detect_pallet(self, img_path):
        tray_results = self.model(img_path)
        img = Image.open(img_path)
        return self.process_detections(tray_results, img)

    # 处理过程
    def process_detections(self, tray_results, img):
        tray_info = []
        tray_counter = 20
        # 遍历每一个检测结果
        for result in tray_results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框和置信度
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                confidence = box.conf.cpu().numpy()
                class_id = box.cls.cpu().numpy()

                if confidence >= self.min_confidence and class_id == 0:  # 假设手势类别为 0
                    cropped_image = img.crop((xmin - 100, ymin - 100, xmax + 500, ymax + 100))
                    r = self.core.start(img=cropped_image, target="TuoPan")
                    tray_info.append({
                        'tray_id': tray_counter,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                        'state': r
                    })
                    tray_counter += 1
        return tray_info