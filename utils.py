# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/16 2:11
@File     : utils.py
@Project  : YiJia_BE
@Introduce:
"""
import numpy as np
import cv2
from PIL import Image
import requests
import base64

def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64, np.intc)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

def image_to_numpy(image):
    # 如果输入是 URL 类型的图像
    if isinstance(image, str) and image.startswith('http'):
        response = requests.get(image)
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
    # 如果输入是文件路径类型的图像
    elif isinstance(image, str):
        image = cv2.imread(image)
    # 如果输入是 numpy 数组
    elif isinstance(image, np.ndarray):
        image = image
    # 如果输入是 PIL 图像
    elif isinstance(image, Image.Image):
        image = np.array(image)
        # 如果图像是 RGB 模式，需要转换为 BGR 模式
        if image.shape[2] == 3:  # 检查是否是RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 如果输入是 Base64 编码的图像
    elif isinstance(image, str) and image.startswith('data:image'):
        header, encoded = image.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = np.asarray(bytearray(image_data), dtype=np.uint8)
        image = cv2.imdecode(image, -1)
    else:
        raise ValueError("Unsupported input type")

    return image

