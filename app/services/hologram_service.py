import os
import tempfile

import dlib
import cv2
import numpy as np


# 額頭區塊
def detect_forehead(landmarks):
    eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
    min_y = min([p[1] for p in eyebrow_points])
    left_x = landmarks.part(17).x
    right_x = landmarks.part(26).x
    forehead_height = int((right_x - left_x) * 0.4)
    segment_height = forehead_height // 3
    y_coords = [min_y - i * segment_height

                for i in range(4)]

    boxes, centers = [], []
    for i in range(3):
        top_left = (left_x, y_coords[i + 1])
        bottom_right = (right_x, y_coords[i])
        center_x = (left_x + right_x) // 2
        center_y = (y_coords[i] + y_coords[i + 1]) // 2
        boxes.append((top_left, bottom_right))
        centers.append((center_x - 10, center_y + 5))
    return boxes, centers

# 切出位置存成圖片

# def save_box_image(raw_img, top_left, bottom_right,filename):
def extract_box_image(raw_img, top_left, bottom_right,filename):
    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w = raw_img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    cropped_img = raw_img[y1:y2, x1:x2]
    if cropped_img.size > 0:
        return cropped_img,filename

# 臉頰
def draw_vertical_segments(img, pt1_idx, pt2_idx, color, label_prefix, raw_img=None, landmarks=None):
    x1, y1 = landmarks.part(pt1_idx).x, landmarks.part(pt1_idx).y
    x2, y2 = landmarks.part(pt2_idx).x, landmarks.part(pt2_idx).y
    top_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))

    box_height = bottom_right[1] - top_left[1]
    segment_height = box_height // 3

    rois = []

    for i in range(3):
        seg_top = top_left[1] + i * segment_height
        seg_bottom = seg_top + segment_height if i < 2 else bottom_right[1]
        seg_top_left = (top_left[0], seg_top)
        seg_bottom_right = (bottom_right[0], seg_bottom)

        cv2.rectangle(img, seg_top_left, seg_bottom_right, color, 2)
        label = f"{label_prefix}_part{i + 1}"
        cv2.putText(img, label, (seg_top_left[0], seg_top_left[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if raw_img is not None:
            roi = extract_box_image(raw_img, seg_top_left, seg_bottom_right,label)
            if roi is not None:
                rois.append(roi)


    return rois,label

# 鼻子
def draw_square_on_point(img, point_index, color, label, raw_img=None, landmarks=None):
    cx = landmarks.part(point_index).x
    cy = landmarks.part(point_index).y
    size = 20
    top_left = (cx - size // 2, cy - size // 2)
    bottom_right = (cx + size // 2, cy + size // 2)
    cv2.rectangle(img, top_left, bottom_right, color, 2)
    cv2.putText(img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if raw_img is not None:
        roi=extract_box_image(raw_img, top_left, bottom_right, label)
        return roi


# ---------- 臉部偵測與處理 ---------- #
def detect_and_process_faces(img_path):
    predictor_path = "C:/Users/Yun/PycharmProjects/PythonProject/shape_predictor_68_face_landmarks.dat"#dlid套件
    region_list = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)



    img = cv2.imread(img_path)
    if img is None:
        print("無法加載圖像")
        return None

    img_raw = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)


        # 額頭區域三等分
        forehead_boxes, centers = detect_forehead(landmarks)
        for i, (box, pos) in enumerate(zip(forehead_boxes, centers)):
            cv2.rectangle(img, box[0], box[1], (0, 0, 255), 1)
            roi=extract_box_image(img_raw, box[0], box[1],f"forehead_{i+1}")
            if roi:
                region_list.append(roi)

        # 左右臉頰垂直三等分
        right_rois, _ = draw_vertical_segments(img, 40, 4, (0, 255, 0), "right_cheek", img_raw, landmarks)
        left_rois, _ = draw_vertical_segments(img, 47, 12, (0, 255, 0), "left_cheek", img_raw, landmarks)
        region_list.extend(right_rois + left_rois)

        # 其他區域
        draw_rect = lambda *args, **kwargs: None  # 保留占位
        draw_rect(img, 32, 52, (0, 255, 255), "philtrum", img_raw, landmarks)
        draw_rect(img, 39, 32, (255, 0, 255), "right_nostril", img_raw, landmarks)
        draw_rect(img, 42, 34, (255, 0, 255), "left_nostril", img_raw, landmarks)
        # 鼻子
        nose_tip = draw_square_on_point(img, 30, (0, 0, 0), "nose_tip", img_raw, landmarks)
        nose_root = draw_square_on_point(img, 27, (0, 0, 0), "nose_root", img_raw, landmarks)
        if nose_tip:
            region_list.append(nose_tip)
        if nose_root:
            region_list.append(nose_root)


    return img,region_list

# ---------- 圖像格狀分析 ---------- #
def grid_analysis(image, name='unnamed'):

    h, w = image.shape[:2]


    if h < 400 or w < 400:
        scale = max(400 / h, 400 / w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    grid_rows, grid_cols = 100, 100
    cell_h, cell_w = image.shape[0] // grid_rows, image.shape[1] // grid_cols
    avg_all = image.mean(axis=(0, 1)).astype(np.uint8)
    out_img = np.zeros_like(image)
    replace_img = image.copy()

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1, y2 = row * cell_h, min((row + 1) * cell_h, image.shape[0])
            x1, x2 = col * cell_w, min((col + 1) * cell_w, image.shape[1])
            cell = image[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            avg = cell.mean(axis=(0, 1)).astype(np.uint8)
            bright = avg.mean()
            out_img[y1:y2, x1:x2] = avg
            if bright < 122:
                cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                replace_img[y1:y2, x1:x2] = avg_all

    for i in range(1, grid_rows):
        cv2.line(out_img, (0, i * cell_h), (image.shape[1], i * cell_h), (0, 0, 0), 1)
    for j in range(1, grid_cols):
        cv2.line(out_img, (j * cell_w, 0), (j * cell_w, image.shape[0]), (0, 0, 0), 1)

    return {
        'filename': name,
        'original': image,
        'grid': out_img,
        'dark_blocks': replace_img
    }

def main(request):
    if 'image' not in request.files:
        return 'No image uploaded', 400
    image_bytes = request.files['image'].read()


    # 將 image_bytes 寫入暫存檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        # 呼叫你的核心處理函式
        final_img, regions = detect_and_process_faces(temp_path)

        # 儲存處理後圖像至 Bytes
        if final_img is None:
            return None

        _, buffer = cv2.imencode('.png', final_img)
        return final_img
    finally:
        os.remove(temp_path)




