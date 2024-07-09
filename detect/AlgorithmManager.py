# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/4/27 14:56
@File     : AlgorithmCore.py
@Project  : YiJia
@Introduce:
"""
from ultralytics import YOLO
import os
base_path = os.path.abspath(os.path.dirname(__file__))
TuoPan_model_path = os.path.join(base_path, "model/TuoPan.pt")
shu_model=os.path.join(base_path, "model/best.pt")
XiangTi_model=os.path.join(base_path, "model/xiang.pt")
TuoPan_CN_names = {0: '1', 1: '0'}
class AlgorithmManager:
    def __init__(self):
        self.target = None
        self.tuoPan_model = YOLO(TuoPan_model_path)
        self.shu_model =YOLO(shu_model)
        self.xiangTi_model = YOLO(XiangTi_model)
    def start(self, img, target):
        each_CN_class_list = []
        if target == "TuoPan":
            # 得到结果
            results = self.tuoPan_model.predict(source=img, show=False, save=False, verbose=False)[0]
            # 分析结果
            tuoPan_cls_list = results.boxes.cls.tolist()  # 托盘好坏
            if len(tuoPan_cls_list) == 0:
                return 0
            elif int(tuoPan_cls_list[0]):
                return 0
            else:
                return 1
        elif target == "shu":
            results = self.shu_model.predict(source=img, show=False, save=False, verbose=False,conf=0.6)[0]
            coord_list = results.boxes.xyxy.tolist() # 坐标
            return coord_list
        elif target == "XiangTi":
            results = self.xiangTi_model.predict(source=img, show=False, save=False)[0]
            xiang_list = results.boxes.cls.tolist()  # 托盘好坏
            if len(xiang_list) == 0:
                return 0
            elif int(xiang_list[0]) == 1:
                return 0
            else:
                return 1
        if each_CN_class_list:
            return each_CN_class_list