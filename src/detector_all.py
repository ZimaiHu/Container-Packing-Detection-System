# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/7/17 12:53
@File     : detector.py
@Project  : ff
@Introduce:
"""
from src.cargo.cargo_fod_detector import FodsDetector
from src.cargo.cargo_label_detector import CargoLabelDetector
from src.cargo.cargo_pallet_corner_detector import PalletDetector
from src.cargo.cargo_strap_detector import StrapDetector
from src.cargo.cargo_seal_detector import SealDetector
from src.cargo.cargo_chaituo_detector import ChaituoDetector
from src.cargo.cargo_qianhou_detector import QianhouDetector
from src.cargo.cargo_zhedang_detector import ZheDangDetector
from src.container.container_number_detector import ContainerNumberDetector

__all__ = [
    "FodsDetector",
    "CargoLabelDetector",
    "PalletDetector",
    "StrapDetector",
    "SealDetector",
    "ChaituoDetector",
    "QianhouDetector",
    "ZheDangDetector",
    "ContainerNumberDetector"
]
