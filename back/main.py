from flask import Flask, request
from entity.Detector import Detector
from entity.Classify import Classify
import json
import os

print(__name__)
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def diagnose():
    print(request.files)
    if 'image' not in request.files or request.files['image'].filename == '':
        return json.dumps({
            'code':400,
            'msg': '未上传图片',
            'data':{}
        })
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    file = request.files['image']
    image_path = f'tmp/{file.filename}'
    file.save(image_path)
    result_json = detect_predict(image_path)
    #删除临时文件
    os.remove(image_path)
    return result_json

def detect_predict(image_path):
    box = detector.detect(image_path)
    if not box:
        return json.dumps({'code': 201, 'msg': '未检测到舌头', 'data': {}})
    class_name, prob = classify.predict(image_path,box)
    result_json = {
        'class': class_name,
        'confidence': prob,
        'bbox': box
    }
    return json.dumps({
            'code': 200,
            'msg': '成功',
            'data': result_json
        },ensure_ascii=False)


if __name__ == '__main__':
    detector = Detector()
    classify = Classify()
    app.run(debug=True, host='0.0.0.0')
