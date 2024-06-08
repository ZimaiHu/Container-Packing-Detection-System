import cv2
import numpy as np

class StrapDetector:
    def __init__(self, min_line_length=100, max_line_gap=10, threshold=50, min_angle_deg=30, max_angle_deg=60):
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.threshold = threshold
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg

    def detect_strap(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Image could not be loaded"}

        edges = self.img_process(img)
        straps_present, straps_positions = self.detect_and_mark_diagonal_lines(img.copy(), edges)

        result = {
            # "Has Straps": "yes" if straps_present else "no",
            "Strap Positions":
                straps_positions if straps_present else []
        }
        return result

    def img_process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def detect_and_mark_diagonal_lines(self, original_image, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, self.threshold,
                                minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        straps_present = False
        straps_positions = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if self.min_angle_deg <= np.abs(angle) <= self.max_angle_deg or \
                       (180 - self.max_angle_deg) <= np.abs(angle) <= (180 - self.min_angle_deg):
                        straps_present = True
                        # Determine the bounding box coordinates
                        xmin = int(min(x1, x2))
                        ymin = int(min(y1, y2))
                        xmax = int(max(x1, x2))
                        ymax = int(max(y1, y2))
                        strap_info = {
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax
                        }
                        straps_positions.append(strap_info)
        return straps_present, straps_positions
