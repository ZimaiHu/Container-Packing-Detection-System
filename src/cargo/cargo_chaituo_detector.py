import os
import re
import logging
import cv2

logging.getLogger('ppocr').setLevel(logging.WARNING)

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置环境变量以避免某些库的潜在冲突
class ChaituoDetector:
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr = None

    def load_model(self, model_path):
        pass
    def load_ocr_model(self, model_obj):
        self.ocr = model_obj

    # 主探测函数
    def detect_chaituo(self, img_path):
        label_text = self.recognize_text_paddleocr(img_path)
        formatted_number = self.format_extracted_number(label_text)
        formatted_number = formatted_number.replace(".", "")
        label_text = ''.join(re.findall(r'\d+', formatted_number))
        return label_text

    # OCR识别
    def recognize_text_paddleocr(self, img_path):
        """ 使用PaddleOCR从图像中识别文本，专注于垂直对齐的文本 """

        # 对图像进行OCR识别
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"图像加载失败: {img_path}")

        # 修改，对图片进行裁剪
        height, width, _ = img.shape

        # 计算裁剪区域
        start_row = height // 3
        end_row = 2 * height // 3
        start_col = width // 3
        end_col = 2 * width // 3

        img = img[start_row:end_row, start_col:end_col]

        if self.ocr is None:
            self.load_model()

        result = self.ocr.ocr(img, cls=True)

        for t in result:
            if t is None:
                return ""
            else:
                # 初始化一个空列表来存储所有识别到的文本
                all_texts = []

                # 遍历每个结果并提取文本
                for res in result:
                    for line in res:
                        # 追加每行文本（line[1][0]包含文本）
                        all_texts.append(line[1][0])

                # 将所有提取的文本片段连接成一个字符串
                combined_text = ' '.join(all_texts)
                return combined_text

    # 正则变换
    def format_extracted_number(self, text):
        """
        从文本中提取并格式化一个带有两个小数点的数字序列，优先匹配 'kg' 前面的数字
        """
        # 匹配所有符合格式的数字序列
        matches = re.findall(r'(?<!\d)\d+\.\d+\.\d+(?!\d)', text)
        if not matches:
            return ""  # 如果没有匹配项，直接返回空字符串

        # 查找 "kg" 的位置
        kg_index = text.find("kg")
        if kg_index != -1:
            # 如果找到 "kg"，优先选择 "kg" 前最近的数字
            closest_match = None
            closest_distance = float('inf')
            for match in matches:
                match_index = text.find(match)
                if match_index != -1 and match_index < kg_index:
                    # 计算距离
                    distance = kg_index - match_index
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_match = match

            # 如果找到符合条件的 "kg" 前数字，格式化并返回
            if closest_match:
                parts = closest_match.split('.')
                if len(parts) == 3:
                    return f"{parts[0][-3:]}.{parts[1]}.{parts[2][:2]}"

        # 如果没有找到符合条件的数字或没有 "kg"，返回第一个匹配
        match = matches[0]
        parts = match.split('.')
        if len(parts) == 3:
            return f"{parts[0][-3:]}.{parts[1]}.{parts[2][:2]}"

        return ""  # 如果没有符合条件的，返回空字符串


if __name__ == '__main__':
    import cv2
    from paddleocr import PaddleOCR

    # 创建 ChaituoDetector 实例
    detector = ChaituoDetector()

    # 初始化 PaddleOCR 模型
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, use_mkldnn=False,
                           det_model_dir="../../weights/ocr/ch_PP-OCRv4_det_infer")

    # 将 PaddleOCR 对象传递给 load_ocr_model
    detector.load_ocr_model(paddle_ocr)

    # 测试图片路径
    img_path = '../../ceshitu/fault/bupaiceshi.jpg'  # 请根据需要修改为实际的图片路径

    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"无法加载图片: {img_path}")

    # 获取图像尺寸
    height, width, _ = image.shape

    # 计算裁剪区域
    start_row = height // 3
    end_row = 2 * height // 3
    start_col = width // 3
    end_col = 2 * width // 3

    # 绘制裁剪区域的矩形框
    cv2.rectangle(image, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)  # 绿色框，线宽为2

    # 缩放图像以适应窗口大小
    scale_percent = 50  # 缩放比例，调整为原来的 50%
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # 进行探测并获取结果
    result = detector.detect_chaituo(img_path)
    print("Extracted Label Text:", result)

    # 显示缩放后的图像
    cv2.imshow('Image with Cropping Area', resized_image)
    cv2.waitKey(0)  # 等待键盘输入以关闭显示窗口
    cv2.destroyAllWindows()
