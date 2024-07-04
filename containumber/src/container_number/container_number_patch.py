# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 12:59
@File     : container_number_v2h.py
@Project  : LLM
@Introduce:
"""
import time
import numpy as np
import cv2
from ultralytics import YOLO

# model_path = "container_number/model/best.pt"
confidence_threshold = 0.1


class ContainerNumberPatch:
    def __init__(self):
        self.s_model = None
        self.h_model = None

    def load_model(self, model_path):
        self.s_model = YOLO(model_path[0])  # 横向小块分割
        self.h_model = YOLO(model_path[1])  # 竖向小块分割
        self.warmup()

    def warmup(self):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.s_model.predict(_dummy_image, verbose=False)
        self.h_model.predict(_dummy_image, verbose=False)

    def detect(self, image):
        height, width, _ = image.shape
        # 竖向图片
        if height > width:
            res = self.s_model.predict(image, show=False, save=False, verbose=False)[0]
            boxes = res.boxes.data.tolist()
            if boxes:
                sorted_boxes = sorted(boxes, key=lambda x: x[1])
                new_image = self.reassemble_characters(image, True, sorted_boxes)
                boxes_img_list = self.get_boxes_image_list(sorted_boxes, image)
                return new_image, boxes_img_list
            else:
                return None, None
        # 横向图片
        else:
            res = self.h_model.predict(image, show=False, save=False, verbose=False)[0]
            boxes = res.boxes.data.tolist()
            if boxes:
                sorted_boxes = sorted(boxes, key=lambda x: x[0])
                boxes_img_list = self.get_boxes_image_list(sorted_boxes, image)
                return image, boxes_img_list
            else:
                return None, None

    @staticmethod
    def reassemble_characters(image, is_vertical, boxes):
        # define the expansion size of character region
        if is_vertical:
            expand_x, expand_y = 10, 2
        else:
            expand_x, expand_y = 2, 5

        # store the cropped and adjusted character regions
        cropped_images = []

        # iterate over each bounding box, crop and expand the character region
        for box in boxes:
            x1, y1, x2, y2 = box[:4]

            # calculate the expanded coordinates, make sure not to exceed the image boundary
            x1_expanded = max(x1 - expand_x, 0)
            y1_expanded = max(y1 - expand_y, 0)
            x2_expanded = min(x2 + expand_x, image.shape[1])
            y2_expanded = min(y2 + expand_y, image.shape[0])

            # crop the expanded region
            cropped = image[int(y1_expanded):int(y2_expanded), int(x1_expanded):int(x2_expanded)]
            cropped_images.append(cropped)

        # calculate the max height of all cropped regions
        max_height = max([img.shape[0] for img in cropped_images])

        # add black padding to adjust the height of each region
        adjusted_images = []
        for img in cropped_images:
            # calculate the padding size
            padding_top = (max_height - img.shape[0]) // 2
            padding_bottom = max_height - img.shape[0] - padding_top

            # add padding
            padded = cv2.copyMakeBorder(img, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            adjusted_images.append(padded)

        # horizontal concatenation of character regions
        new_image = np.concatenate(adjusted_images, axis=1)

        # add extra black padding to the top and bottom of the new image
        padding_height = 3
        new_image_padded = cv2.copyMakeBorder(new_image, padding_height, padding_height, 0, 0, cv2.BORDER_CONSTANT,
                                              value=[0, 0, 0])

        return new_image_padded

    @staticmethod
    def get_boxes_image_list(boxes, img):
        boxes_image_list = []
        for box in boxes:
            box = [int(x) for x in box[:4]]
            x1, y1, x2, y2 = box
            patch = img[y1:y2, x1:x2]
            # cv2.imwrite(f"patch_{datetime.now()}.jpg", patch)
            boxes_image_list.append(patch)
        return boxes_image_list
    # def reorder_boxes(self, is_vertical, boxes):
    #     if is_vertical:
    #         # sort by y1 from top to bottom
    #         boxes = sorted(boxes, key=lambda box: box[1])
    #     else:
    #         # sort by x1 from left to right
    #         boxes = sorted(boxes, key=lambda box: box[0])
    #
    #     return boxes
