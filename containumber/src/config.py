# -*- coding:utf-8 -*-
"""
@Author   : NEU_DAO WangShuo
@Time     : 2024/5/18 1:47
@File     : config.py
@Project  : YiJia_BE
@Introduce:
"""
# 模型路径
# region_model_path = "models/container_number/container_number_region_4.pt"       # v0 优秀
# region_model_path_obb = "models/container_number/container_number_region_obb_7_2.pt"   # v1_obb
# region_model_path = "models/container_number/container_number_region_4.pt"
# patch_model_path = "models/container_number/container_number_patch.pt"       # v0
# patch_model_path = "models/container_number/container_number_patch_7_2.pt"   # v1 优秀

# ocr_model_path = "models/ocr/cv_crnn_ocr-recognition-general_damo"           # v0
# ocr_model_path = "/home/daoes/wangshuo/MyProject/YiJia/models/ocr/ocr_funeting"   # v1 优秀

# 使用版
region_model_path = "models/container_number/container_number_region_5_7_3.pt"
patch_model_path = ["models/container_number/container_number_patch_shu_7_3.pt",
                    "models/container_number/container_number_patch_heng_7_3.pt"]
ocr_model_path = "models/ocr/ocr_funeting"
