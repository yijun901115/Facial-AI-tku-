import base64

import cv2
from flask import Blueprint, request, jsonify, send_file, Response
from app.services.hologram_service import main

hologram_bp = Blueprint('hologram_bp', __name__)

# 臉部全息位置、清除障礙物的RestfulAPI(參考此程式)
@hologram_bp.route('/hologram', methods=['POST'])
def hologram():
    # 呼叫hologram_service的main
    result_img = main(request)

    # 編碼圖片為 base64
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "status": "success",
        "image": img_base64
    })
    # 回傳 HTML 內容(測試)
    # html_img_tag = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # return Response(html_img_tag, mimetype='text/html')