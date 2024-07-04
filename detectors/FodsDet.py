from ultralytics import YOLO
class FodsDetector:
    def __init__(self):
        pass
    def load_model(self, model_path):
        self.model = YOLO(model_path[0])
    def detect_fods(self, image_path):
        results = self.model.predict(source=image_path, show=False, save=False)
        fod_cls_list = results[0].boxes.cls.tolist()
        overlapping_info = []
        if len(fod_cls_list) == 0:
            has_fod=0
            object_info = {
                # 'tray_id': has_fod,
            }
            # overlapping_info.append(object_info)
        elif int(fod_cls_list[0]):
            has_fod=0
            object_info = {
                # 'tray_id': has_fod,
            }
            # overlapping_info.append(object_info)
        else:
            has_fod=1
            counter = 1
            for cls, coord in zip(results[0].boxes.cls.tolist(), results[0].boxes.xyxy.tolist()):
                xmin, ymin, xmax, ymax = coord
                object_info = {
                    # 'has_fod': has_fod,
                    'tray_id': counter,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                }
                overlapping_info.append(object_info)
                counter += 1
        return overlapping_info
