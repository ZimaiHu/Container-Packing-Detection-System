from PIL import Image
from ultralytics import YOLO  # 使用 YOLOv8
class PalletDetector:
    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence

    def load_model(self, model_path):
        # 加载 YOLOv8 模型
        self.model = YOLO(model_path[0])  # 修改为直接加载 YOLOv8 模型
        self.tuoPan_model = YOLO(model_path[1])
    # 主探测函数
    def detect_pallet(self, img_path):
        tray_results = self.model(img_path)
        img = Image.open(img_path)
        return self.process_detections(tray_results, img)

    # 处理过程
    def process_detections(self, tray_results, img):
        tray_info = []
        tray_counter = 40
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
                    #托盘破损检测
                    results = self.tuoPan_model.predict(source=cropped_image, show=False, save=False, verbose=False)[0]
                    # 分析结果
                    tuoPan_cls_list = results.boxes.cls.tolist()  # 托盘好坏
                    if len(tuoPan_cls_list) == 0:
                        r=0
                    elif int(tuoPan_cls_list[0]):
                        r=0
                    else:
                        r=1
                    tray_info.append({
                        'tray_id': tray_counter,
                        'xmin': float(xmin),
                        'ymin': float(ymin),
                        'xmax': float(xmax),
                        'ymax': float(ymax),
                        'state': r
                    })
                    tray_counter += 1
        return tray_info