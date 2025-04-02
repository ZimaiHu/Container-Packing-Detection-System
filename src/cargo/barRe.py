import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

def display_with_barcodes(im, decodedObjects):
    # 遍历所有检测到的条形码
    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # 如果条形码检测区域的点数大于 4，则将其转换为凸包
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))  # 将凸包转换为适合 OpenCV 的坐标格式
        else:
            hull = points

        # 获取边界点数
        n = len(hull)

        # 画出条形码的边界框
        for j in range(0, n):
            pt1 = (int(hull[j][0]), int(hull[j][1]))
            pt2 = (int(hull[(j + 1) % n][0]), int(hull[(j + 1) % n][1]))
            cv2.line(im, pt1, pt2, (255, 0, 0), 3)

        # 在条形码上方添加条形码数据
        barcode_data = decodedObject.data.decode("utf-8").replace("\x1d", "")
        x, y = int(hull[0][0]), int(hull[0][1])
        cv2.putText(im, barcode_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow("Barcode Detection", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filter_20_digits_barcodes(decoded_objects):
    """
    筛选出位数为20位的条码数据，并去除前两位

    Args:
        decoded_objects (list): pyzbar解码后的对象列表

    Returns:
        list: 处理后的数据列表（例如：0037... → 3716...）
    """
    filtered_data = []
    for obj in decoded_objects:
        # 获取原始数据并移除分隔符
        raw_data = obj.data.decode("utf-8").replace("\x1d", "")

        # 筛选20位数据
        if len(raw_data) == 20:
            # 去除前两位并保存
            filtered_data.append(raw_data[2:])

    return filtered_data


def detect_and_display_barcode(image_path):
    # 打开图像
    img = Image.open(image_path)
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 使用 pyzbar 解码条形码
    decoded_objects = decode(img)
    for obj in decoded_objects:
        # 去除数据中的分隔符 \x1d
        barcode_data = obj.data.decode("utf-8").replace("\x1d", "")
        print("Type : ", obj.type)
        print("Data : ", barcode_data)

    # 筛选出位数为20位的条码数据，并去除前两位
    filtered_results = filter_20_digits_barcodes(decoded_objects)
    print("\n处理后的20位条码数据:")
    for data in filtered_results:
        print(f"{data}")

    # 在图片上标记条形码(测试时打开，部署时关闭)
    # display_with_barcodes(img_cv2, decoded_objects)

    return filtered_results[0]

if __name__ == '__main__':

    # 调用函数，传入图像路径
    # detect_and_display_barcode("../../ceshitu/yijia_test/test3.png")  # 请将路径替换为实际图像路径
    detect_and_display_barcode("../../ceshitu/ceshi/barcode.jpg")  # 请将路径替换为实际图像路径
