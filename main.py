from flask import Flask, request, jsonify
import numpy as np
import time
from detector import Detector
from functools import wraps

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


def validate_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()
        if not data or 'url' not in data or 'state' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
        return f(*args, **kwargs)

    return decorated_function


DETECTION_TYPES = {
    "1": ("5", "货柜号识别"),
    "2": ("1", "货物检测"),
    "3": ("2", "托脚检测"),
    "4": ("4", "封条检测"),
    "5": ("6", "拆托检测"),
    "6": ("3", "绑带"),
    "7": ("7", "空柜检测"),
    "8": ("8", "前后遮挡检测"),
    "9": ("9", "货柜遮挡检测")
}


@app.route('/predict', methods=['POST'])
@validate_input
def submit():
    data = request.get_json()
    url = data['url']
    state = data['state']

    if state not in DETECTION_TYPES:
        return jsonify({'error': 'Invalid state parameter'}), 400

    detection_type, detection_name = DETECTION_TYPES[state]

    start_time = time.time()
    result_dict = detector.detect_all(url, detection_type)
    end_time = time.time()

    print(f"{detection_name} - 耗时: {end_time - start_time:.4f} seconds")
    print(result_dict)

    return jsonify(convert_numpy(result_dict))


if __name__ == '__main__':
    app.run(threaded=True, debug=True)