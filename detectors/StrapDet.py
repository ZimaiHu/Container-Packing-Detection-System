import cv2
import numpy as np
class StrapDetector:
    def __init__(self, min_line_length=100, max_line_gap=10, threshold=50, min_angle_deg=20, max_angle_deg=70):
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.threshold = threshold
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
#主探测函数
    def detect_strap(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Image could not be loaded"}

        edges = self.img_process(img)
        straps_present, straps_positions = self.detect_and_mark_diagonal_lines(img.copy(), edges)

        result = {
            "Has Straps": "yes" if straps_present else "no",
            "Strap Positions": straps_positions if straps_present else []
        }
        return result
#图像处理
    def img_process(self, image):
        # 将图像转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 应用高斯模糊减少噪声，改善边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        return edges
#找线
    def detect_and_mark_diagonal_lines(self, original_image, edges):
        # 使用Hough变换找到线条
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
                        straps_positions.append((x1, y1, x2, y2))
        return straps_present, straps_positions