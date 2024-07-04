from flask import Flask, request, jsonify
import numpy as np
import time
from detector import Detector

app = Flask(__name__)
detector = Detector()  # 创建 Detector 的实例


def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64, np.intc)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


@app.route('/predict', methods=['POST'])
def submit():
    # 获取通过POST请求发送的数据
    data = request.get_json()
    print("Raw data:", request.data)
    # 检查数据是否包含名为'some_string'的键
    if 'url' not in data:
        return jsonify({'error': 'Missing some_string parameter'}), 400

    # 读取字符串
    url = data['url']
    id = data['state']
    print(url)

# 货柜号1
# 货物2
# 托脚3
# 封条4
# 绑带
# 空柜检测 未做


        # 根据id执行不同的检测
    if id == "1":
        start_time = time.time()
        result_dict = detector.detect_all(url, "5")
        print(result_dict)
        print("货柜号识别")
    elif id == "2":
        start_time = time.time()
        result_dict = detector.detect_all(url, "1")
        print(result_dict)
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f" 货物检{elapsed_time:.4f} seconds")
        print("货物检测")
    elif id == "3":
        result_dict = detector.detect_all(url, "2")
            # result_dict = str(result_dict)
        print(result_dict)
        print("托脚检测")
    elif id == "4":
        result_dict = detector.detect_all(url, "4")
        print(result_dict)
        print("封条检测")
    elif id == "5":
        result_dict = detector.detect_all(url, "6")
        print(result_dict)
        print("拆托检测")
    elif id == "6":
        result_dict = detector.detect_all(url, "3")
        print(result_dict)
        print("绑带")
    elif id == "7":
        result_dict = detector.detect_all(url, "7")
        print(result_dict)
        print("空柜检测")
    elif id == "8":
        result_dict = detector.detect_all(url, "8")
        print(result_dict)
        print("遮挡检测")
    else:
        result_dict = "无结果"

    result_dict = convert_numpy(result_dict)


    return result_dict



if __name__ == '__main__':
    app.run(threaded=True,debug=True)

