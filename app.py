import time
from detector import Detector

def run_detection(detector, img_path, task, measure_time=False):
    start_time = time.time() if measure_time else None

    result = detector.detect_all(img_path, task)

    if measure_time:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

    print(f"Task {task} - Image: {img_path}")
    print(f"Result: {result}")
    print("-" * 50)


if __name__ == "__main__":
    detector = Detector()

    test_cases = [
        ("ceshitu/ceshi/zhengpai.jpg", "1", True),
        ("ceshitu/ceshi/zhengpai.jpg", "2", True),
        ("ceshitu/ceshi/bangdai.jpeg", "3", True),
        ("ceshitu/ceshi/fengtiao.jpg", "4", True),
        ("ceshitu/ceshi/huoguihao.jpg", "5", True),
        ("ceshitu/ceshi/chaituo.jpg", "6", True),
        ("ceshitu/ceshi/yiwujiance.jpeg", "7", True),
        ("ceshitu/ceshi/qianhouzhedang.jpg", "8", True),
        ("ceshitu/ceshi/zhedang.jpeg", "9", True)
    ]

    # (1：整排，2：托盘角，3：绑带，4：封条，5：货柜号，6：拆托，7：异物检测，8前后遮挡货物,9柜内遮挡)
    for img_path, task, measure_time in test_cases:
        run_detection(detector, img_path, task, measure_time)