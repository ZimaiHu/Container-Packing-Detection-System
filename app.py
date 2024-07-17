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
        ("ceshitu/0-0.jpg", "1", False),
        ("ceshitu/202406262.jpg", "2", False),
        ("ceshitu/9832.jpeg", "3", False),
        ("ceshitu/0333.jpeg", "4", True),
        ("ceshitu/zhedang.jpeg", "5", True),
        ("ceshitu/521.jpg", "6", True),
        ("ceshitu/333.jpeg", "7", True),
        ("ceshitu/d04c.jpg", "8", True),
        ("ceshitu/zhedang1.jpg", "9", True)
    ]

    # (1：货物标签，2：托盘角，3：绑带，4：封条，5：货柜，6：拆托，7：异物检测，8前后遮挡货物,9柜内遮挡)
    for img_path, task, measure_time in test_cases:
        run_detection(detector, img_path, task, measure_time)