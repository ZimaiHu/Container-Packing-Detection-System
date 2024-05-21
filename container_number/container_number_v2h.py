# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/14 12:59
@File     : V2HCharDetector.py
@Project  : LLM
@Introduce:
"""
import os
import time
from ultralytics import YOLO
import numpy as np
import cv2
model_path = "container_number/model/best.pt"
confidence_threshold = 0.1
class V2HCharDetector:
    def __init__(self):
        self.model = YOLO(model_path)
        self.warmup()

    def warmup(self):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(_dummy_image, verbose=False)

    def detect(self, image):

        height, width, _ = image.shape
        is_vertical = False
        if height > width:

            is_vertical = True
        else:
            pass

        st = time.time()
        res = self.model(image, verbose=False)
        boxes = res[0].boxes.data.cpu().numpy()
        # boxes: [[x1, y1, x2, y2, conf, cls],[...box2....],[...box3...],..., [...boxN...]]]
        num_boxes = len(boxes)
        # print(f"{num_boxes} char boxes detected. Took {time.time() - st:.3f} seconds.")
        # print('All chars det confidence:', [box[4] for box in boxes])
        new_boxes = [box for box in boxes if box[4] >= confidence_threshold]

        if num_boxes - len(new_boxes) > 0:
            pass
            # print(f"{num_boxes - len(new_boxes)} char boxes removed, confidence below {confidence_threshold}.")
        else:
            # print(f"All char boxes kept, confidence above {confidence_threshold}.")
            pass
        """
        debug_image = image.copy()
        for box in new_boxes:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.imwrite("temp_images/chars_debug.jpg", debug_image)
        """

        is_res_ok = True
        if len(new_boxes) >= 11 and is_vertical:
            new_boxes = self.reorder_boxes(is_vertical, new_boxes)
            image = self.reassemble_characters(image, is_vertical, new_boxes)
            # print(f"Reassembled {len(new_boxes)} char boxes into a new image.")
            is_res_ok = True
        elif len(new_boxes) == 11 and not is_vertical:
            new_boxes = self.reorder_boxes(is_vertical, new_boxes)
            image = self.reassemble_characters(image, is_vertical, new_boxes)
            # print(f"Reassembled {len(new_boxes)} char boxes into a new image.")
            is_res_ok = True
        elif len(new_boxes) > 11 and not is_vertical:
            # print(f"More than 11 char boxes detected, but horizontal, return original cropped image")
            is_res_ok = False
        else:
            is_res_ok = False
            # print(f"Manual op required. Only {len(new_boxes)} char boxes left, return original cropped image")

        # cv2.imwrite("temp_images/chars.jpg", image)
        return is_res_ok, image

    def reassemble_characters(self, image, is_vertical, boxes):
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

    def reorder_boxes(self, is_vertical, boxes):
        if is_vertical:
            # sort by y1 from top to bottom
            boxes = sorted(boxes, key=lambda box: box[1])
        else:
            # sort by x1 from left to right
            boxes = sorted(boxes, key=lambda box: box[0])

        return boxes

# detector = V2HCharDetector()
# test_image = cv2.imread("test_images/crop_2.jpg")
# result = detector.detect(test_image)
