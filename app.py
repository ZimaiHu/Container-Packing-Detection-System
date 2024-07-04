import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detector import Detector
from PIL import Image
import time  # 导入 time 模块
if __name__ == "__main__":
    # detector=Detector()
    # def draw_text_on_image(image_path, text):
    #     # 打开图片
    #     img = Image.open(image_path)
    #     draw = ImageDraw.Draw(img)
    #     # 选择一个合适的字体和大小。这里使用 20pt 的大小
    #     # 如果你有 .ttf 格式的字体文件，可以使用下面的代码加载：
    #     # font = ImageFont.truetype('path_to_your_font.ttf', 20)
    #     # 由于可能没有指定字体，这里使用 PIL 的默认字体
    #     font = ImageFont.load_default()
    #     # 设置文本位置和颜色
    #     text_position = (10, 10)  # 图片左上角开始
    #     text_color = (255, 255, 0)  # 明亮的黄色
    #     # 在图片上添加文本
    #     draw.text(text_position, text, font=font, fill=text_color)
    #     img.save(image_path)
    # def detect_images_in_folder(folder_path)
    #     error_folder = os.path.join(folder_path, 'error')
    #     # 创建错误文件夹如果不存在
    #     if not os.path.exists(error_folder):
    #         os.makedirs(error_folder)
    #     # 遍历文件夹中的所有文件
    #     for filename in os.listdir(folder_path):
    #         # 构建完整的文件路径
    #         file_path = os.path.join(folder_path, filename)
    #         # 检查是否是文件
    #         if os.path.isfile(file_path):
    #             try:
    #                 t1 = time.time()
    #                 txt = detector.detect_all(file_path, '5')  # 确保这个函数返回一个字典
    #                 t2 = time.time()
    #
    #                 # 安全地获取 OCR 结果
    #                 detections = txt.get('0', [])
    #                 if detections:  # 确保列表非空
    #                     ocr_result = detections[0].get('OCR_result', '')
    #                 else:
    #                     ocr_result = 'No detections'  # 没有检测到内容
    #
    #                 if ocr_result.strip() == '':
    #                     shutil.copy(file_path, os.path.join(error_folder, filename))
    #                     draw_text_on_image(file_path, 'ERROR: OCR result empty')
    #                 else:
    #                     draw_text_on_image(file_path,ocr_result)
    #
    #                 print(f"Processed {filename} in {t2 - t1} seconds. Detection Result: {ocr_result}")
    #             except Exception as e:
    #                 print(f"Error processing {file_path}: {e}")
    # # 用例使用
    # folder_path = r"C:\Users\86195\Desktop\白色柜体-编号实例（2）"
    # detect_images_in_folder(folder_path)
    img_path1=r"ceshitu/7501.jpg"
    img_path2 = r"ceshitu/333.jpeg"
    img_path3=r"ceshitu/9832.jpeg"
    img_path4 = r"ceshitu/0333.jpeg"
    img_path5 = r"ceshitu/80e.png"
    img_path6 = r"ceshitu/3323.png"
    img_path7 = r"ceshitu/333.jpeg"
    img_path8=r"ceshitu/4338.jpg"
    #(1：货物标签，2：托盘角，3：绑带，4：封条，5：货柜，6：拆托，7：异物检测，8遮挡货物)
    # 创建 Detector 实例
    detector = Detector()
    t1 = time.time()
    txt = detector.detect_all(img_path1, "1")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    goods = txt
    # 打开图像
    image = Image.open(img_path1)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)

    # 添加货物矩形和标签
    for good in goods:
        # 画出货物的矩形
        rect = patches.Rectangle((good['xmin'], good['ymin']), good['xmax'] - good['xmin'], good['ymax'] - good['ymin'],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(good['xmin'], good['ymin'], f"Goods ID: {good['goods_id']}", color='r', fontsize=12,
                verticalalignment='bottom')

        # 画出标签
        for label in good['labelingood']:
            rect_label = patches.Rectangle((label['xmin'], label['ymin']), label['xmax'] - label['xmin'],
                                           label['ymax'] - label['ymin'], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect_label)
            label_text = f"Label ID: {label['label_id']}\nOCR: {label['ocr_result']}"
            ax.text(label['xmin'], label['ymin'], label_text, color='b', fontsize=10, verticalalignment='top')

    # 设置坐标轴限制并显示图像
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)  # 反转 y 轴以符合图像坐标

    plt.show()
    plt.close(fig)

    t1 = time.time()
    txt = detector.detect_all(img_path2,"2")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path6,"3")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path4,"4")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path5,"5")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path6,"6")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path7, "7")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path8, "8")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from detector import Detector
# from PIL import Image
# import os
#
# def process_and_save_image(detector, img_path, output_path):
#     txt = detector.detect_all(img_path, "1")
#     goods = txt
#
#     # 打开图像
#     image = Image.open(img_path)
#     fig, ax = plt.subplots(figsize=(15, 15))
#     ax.imshow(image)
#
#     # 添加货物矩形和标签
#     for good in goods:
#         # 画出货物的矩形
#         rect = patches.Rectangle((good['xmin'], good['ymin']), good['xmax'] - good['xmin'], good['ymax'] - good['ymin'],
#                                  linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         ax.text(good['xmin'], good['ymin'], f"Goods ID: {good['goods_id']}", color='r', fontsize=12,
#                 verticalalignment='bottom')
#
#         # 画出标签
#         for label in good['labelingood']:
#             rect_label = patches.Rectangle((label['xmin'], label['ymin']), label['xmax'] - label['xmin'],
#                                            label['ymax'] - label['ymin'], linewidth=1, edgecolor='b', facecolor='none')
#             ax.add_patch(rect_label)
#             label_text = f"Label ID: {label['label_id']}\nOCR: {label['ocr_result']}\nPaddleOCR: {label['PaddleOcr']}"
#             ax.text(label['xmin'], label['ymin'], label_text, color='b', fontsize=10, verticalalignment='top')
#
#     # 设置坐标轴限制并保存图像
#     ax.set_xlim(0, image.width)
#     ax.set_ylim(image.height, 0)  # 反转y轴以符合图像坐标
#     output_img_path = os.path.join(output_path, os.path.basename(img_path))
#     fig.savefig(output_img_path)
#     plt.close(fig)
#
# def main(input_folder, output_folder):
#     detector = Detector()  # 仅加载一次模型
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for img_filename in os.listdir(input_folder):
#         if img_filename.endswith(('.jpeg', '.jpg', '.png')):
#             img_path = os.path.join(input_folder, img_filename)
#             process_and_save_image(detector, img_path, output_folder)
#             print(f"处理并保存了 {img_filename}")
#
# if __name__ == "__main__":
#     input_folder = r"C:\Users\86195\Documents\WeChat Files\wxid_9moku3u5p15d22\FileStorage\File\2024-07\7.1下午4点半之后整排照片"
#     output_folder = r"C:\Users\86195\Documents\WeChat Files\wxid_9moku3u5p15d22\FileStorage\File\2024-07\output横排"
#     main(input_folder, output_folder)



