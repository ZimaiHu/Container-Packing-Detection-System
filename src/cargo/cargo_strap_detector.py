import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString

class StrapDetector:
    '''
    从模型中获取绑带的 OBB 坐标。
    提取每个箱子的 AABB 坐标。
    检查每个箱子是否与任何绑带相交。
    '''

    def __init__(self):
        pass

    def load_model(self, model_paths):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.models = {
            'cargo': YOLO(model_paths[0]),
            'strap': YOLO(model_paths[1])
        }
        for model in self.models.values():
            model.predict(_dummy_image, verbose=False)

    def detect_strap(self, img_path, draw=False):
        strap_positions = self.extract_strap_info(img_path)

        strap_bboxes = [
            {
                'xmin': int(strap['x1']),
                'ymin': int(strap['y1']),
                'xmax': int(strap['x2']),
                'ymax': int(strap['y2'])
            }
            for strap in strap_positions
        ]

        box_positions = self.extract_box_info(img_path)

        any_box_without_strap, box_intersections = self.check_strap_intersects_box(box_positions, strap_positions)

        if draw:
            self.draw_boxes_and_straps(img_path, box_positions, strap_positions)

        return any_box_without_strap, strap_bboxes

    def extract_strap_info(self, img_path):
        image = cv2.imread(img_path)
        results = self.models['strap'].predict(source=image, show=False, device=0, save=False, verbose=False)
        strap_coordinates = results[0].obb.xyxyxyxy.tolist()

        strap_results = []
        for index, coords in enumerate(strap_coordinates):
            avg_x1 = (coords[0][0] + coords[1][0]) / 2
            avg_y1 = (coords[0][1] + coords[1][1]) / 2
            avg_x2 = (coords[2][0] + coords[3][0]) / 2
            avg_y2 = (coords[2][1] + coords[3][1]) / 2

            strap_info = {
                'strap_id': index + 1,
                'x1': avg_x1,
                'y1': avg_y1,
                'x2': avg_x2,
                'y2': avg_y2
            }
            strap_results.append(strap_info)

        return strap_results

    def extract_box_info(self, img_path, min_confidence=0.65):
        image = cv2.imread(img_path)
        results = self.models['cargo'].predict(source=image, show=False, device=0, save=False, verbose=False)
        classes = results[0].boxes.cls.tolist()
        coordinates = results[0].boxes.xyxy.tolist()
        confidences = results[0].boxes.conf.tolist()

        data = []
        for i in range(len(classes)):
            if classes[i] == 0 and confidences[i] >= min_confidence:
                data.append(coordinates[i] + [confidences[i], int(classes[i])])

        df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

        results = []
        for index, row in df.iterrows():
            box_info = {
                'box_id': index + 1,
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax']
            }
            results.append(box_info)
        return results

    def check_strap_intersects_box(self, box_positions, strap_positions):
        box_intersections = {box['box_id']: False for box in box_positions}

        for strap in strap_positions:
            strap_line = LineString([(strap['x1'], strap['y1']), (strap['x2'], strap['y2'])])

            for box in box_positions:
                box_id = box['box_id']
                box_polygon = Polygon([(box['xmin'], box['ymin']), (box['xmax'], box['ymin']),
                                       (box['xmax'], box['ymax']), (box['xmin'], box['ymax'])])

                if strap_line.intersects(box_polygon):
                    box_intersections[box_id] = True

        all_boxes_with_strap = all(box_intersections.values())
        return all_boxes_with_strap, box_intersections

    def draw_boxes_and_straps(self, img_path, box_positions, strap_positions):
        image = cv2.imread(img_path)
        for box in box_positions:
            cv2.rectangle(image, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])),
                          (255, 0, 0), 2)
            cv2.putText(image, f"Box {box['box_id']}", (int(box['xmin']), int(box['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        for strap in strap_positions:
            avg_x1 = int(strap['x1'])
            avg_y1 = int(strap['y1'])
            avg_x2 = int(strap['x2'])
            avg_y2 = int(strap['y2'])

            cv2.line(image, (avg_x1, avg_y1), (avg_x2, avg_y2), (0, 255, 0), 2)
            cv2.putText(image, f"Strap {strap['strap_id']}", (avg_x1, avg_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_path = "output_with_boxes_and_straps.jpg"
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

if __name__ == '__main__':
    detector = StrapDetector()
    detector.load_model(['../../weights/cargo/cargolabel.pt',
                         '../../weights/cargo/bangdai.pt'
                         ])
    result = detector.detect_strap('../../ceshitu/ceshi/bangdai.jpeg', draw=True)