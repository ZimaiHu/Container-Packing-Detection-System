import cv2
from detect.AlgorithmManager import AlgorithmManager
from container_number.container_number_detector import ContainerNumberDetector
class ContainerDetector:
    def __init__(self, algorithm_target="shu"):
        self.algorithm_target = algorithm_target
    def load_model(self):
        self.algorithm_manager = AlgorithmManager()
        self.detector = ContainerNumberDetector()
#主探测函数
    def detect_container(self, img_path):
        # 开始检测过程
        result = self.algorithm_manager.start(img=img_path, target=self.algorithm_target)

        # 检查结果是否为空
        if result is None or len(result) == 0:
            return {'coordinates': None, 'ocr_text': None}

        # 解包结果中的坐标
        xmin, ymin, xmax, ymax = result[0]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # 读取图像并裁剪到检测区域
        img = cv2.imread(img_path)
        if img is None:
            print("Error loading image.")
            return {'coordinates': None, 'ocr_text': None}

        roi_img = img[ymin:ymax, xmin:xmax]
        cv2.imwrite('shuocr.png', roi_img)

        # 对裁剪后的图像执行OCR
        ocr_text = self.detector.detect(roi_img)

        return {
            'coordinates': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
            'ocr_text': ocr_text
        }
