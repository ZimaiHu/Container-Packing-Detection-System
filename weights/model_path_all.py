# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/18 1:47
@File     : config.py
@Project  : YiJia_BE
@Introduce: Configuration paths for various models
"""

container_number_path = [
    "weights/contain_number/container_number_region_5_7_3.pt",
    [
        "weights/contain_number/container_number_patch_shu_7_3.pt",
        "weights/contain_number/container_number_patch_heng_7_3.pt"
    ],
    "weights/ocr/cv_crnn_ocr-recognition-general_damo_finetuning",
    "weights/ocr/yolo_cls_ocr/container_number_cls_ocr.pt"
]

cargo_label_path = [
    "weights/cargo/cargolabel.pt",
    "weights/cargo/shouxie.pt",
    "weights/cargo/guanjianzi.pt",
    "weights/cargo/huowuposun.pt",
    "weights/ocr/ch_PP-OCRv4_det_infer",
    "weights/ocr/cv_convnextTiny_ocr-recognition-handwritten_damo"
]

cargo_pallet_corner_path = [
    "weights/cargo/tuojiao.pt",
    "weights/cargo/TuoPanposun.pt"
]

cargo_strap_path = [
    "weights/cargo/cargolabel.pt",
    "weights/cargo/bangdai.pt"
]

cargo_seal_path = [
    "weights/cargo/fengtiao.pt",
    "weights/ocr/ch_PP-OCRv4_det_infer"
]

cargo_chaituo_path = [
    "weights/ocr/ch_PP-OCRv4_det_infer"
]

cargo_qianhou_path = [
    "weights/cargo/qianhou.pt",
    "weights/cargo/shouxie.pt",
    "weights/cargo/guanjianzi.pt",
    "weights/cargo/huowuposun.pt",
    "weights/ocr/ch_PP-OCRv4_det_infer",
    "weights/ocr/cv_convnextTiny_ocr-recognition-handwritten_damo"
]

cargo_zhedang_path = [
    "weights/cargo/zhedang.pt",
    "weights/cargo/shouxie.pt",
    "weights/cargo/guanjianzi.pt",
    "weights/cargo/huowuposun.pt",
    "weights/ocr/ch_PP-OCRv4_det_infer",
    "weights/ocr/cv_convnextTiny_ocr-recognition-handwritten_damo"
]

cargo_fods_path = [
    "weights/cargo/fod.pt"
]

__all__ = [
    "container_number_path", "cargo_label_path", "cargo_pallet_corner_path",
    "cargo_strap_path", "cargo_seal_path", "cargo_chaituo_path", "cargo_qianhou_path",
    "cargo_zhedang_path", "cargo_fods_path"
]
