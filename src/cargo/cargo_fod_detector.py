from ultralytics import YOLO


class FodsDetector:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = YOLO(model_path[0])

    def detect_fods(self, image_path):
        results = self.model.predict(source=image_path, show=False, save=False)
        fod_cls_list = results[0].boxes.cls.tolist()
        overlapping_info = []

        has_fod = 0 if not fod_cls_list or int(fod_cls_list[0]) else 1
        if has_fod:
            for counter, (cls, coord) in enumerate(zip(results[0].boxes.cls.tolist(), results[0].boxes.xyxy.tolist()),
                                                   start=1):
                xmin, ymin, xmax, ymax = coord
                object_info = {
                    'tray_id': counter,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                }
                overlapping_info.append(object_info)

        return overlapping_info
