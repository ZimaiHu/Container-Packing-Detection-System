import time
from detector import Detector
if __name__ == "__main__":
    detector = Detector()
    img_path1=r"ceshitu/333.jpeg"
    img_path2 = r"ceshitu/333.jpeg"
    img_path3=r"ceshitu/9832.jpeg"
    img_path4 = r"ceshitu/0333.jpeg"
    img_path5 = r"ceshitu/shutest.jpg"
    img_path6 = r"ceshitu/521.jpg"
    img_path7 = r"ceshitu/333.jpeg"
    #(1：货物标签，2：托盘角，3：绑带，4：封条，5：货柜，6：拆托，7：异物检测)
    t1 = time.time()
    txt = detector.detect_all(img_path1,"1")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path2,"2")
    t2 = time.time()
    print(t2 - t1)
    print(txt)
    t1 = time.time()
    txt = detector.detect_all(img_path3,"3")
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