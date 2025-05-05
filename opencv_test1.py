import dlib
import cv2
import numpy as np

# 加載預訓練的人臉檢測器
detector = dlib.get_frontal_face_detector()

# 加載人臉特徵點檢測器
predictor = dlib.shape_predictor("E:/113_FaceDetected-main/shape_predictor_68_face_landmarks.dat")


def detect_forehead(landmarks):
    # 獲取眉毛特徵點（17-26）
    eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
    min_y = min([p[1] for p in eyebrow_points])  # 眉毛最高點

    # 眉毛兩端座標
    left_x = landmarks.part(17).x
    right_x = landmarks.part(26).x

    # 估計髮際線高度
    forehead_height = int((right_x - left_x) * 0.4)
    hairline_y = min_y - forehead_height

    # 創建額頭輪廓點集
    outline_points = []
    for x in range(left_x, right_x + 1, 3):
        outline_points.append((x, min_y))
    for y in range(min_y, hairline_y - 1, -3):
        outline_points.append((right_x, y))
    for x in range(right_x, left_x - 1, -3):
        outline_points.append((x, hairline_y))
    for y in range(hairline_y, min_y + 1, 3):
        outline_points.append((left_x, y))

    # 計算三等分
    segment_height = forehead_height // 3
    y_coords = [min_y - i * segment_height for i in range(4)]  # 四個分界線y座標

    # 三個分區的方框和中心點（用於標記序號）
    boxes = []
    centers = []
    for i in range(3):
        top_left = (left_x, y_coords[i + 1])
        bottom_right = (right_x, y_coords[i])
        center_x = (left_x + right_x) // 2
        center_y = (y_coords[i] + y_coords[i + 1]) // 2
        boxes.append((top_left, bottom_right))
        centers.append((center_x - 10, center_y + 5))  # 序號位置微調

    return outline_points, boxes, centers


# 新增：標記人中區域的矩形框
def detect_philtrum_box(landmarks):
    # 人中特徵點（33-35）
    point33 = (landmarks.part(33).x, landmarks.part(33).y)  # 鼻尖下方
    point51 = (landmarks.part(51).x, landmarks.part(51).y)  # 上唇中央

    # 計算矩形範圍（稍微擴展寬度）
    width_padding = 20  # 左右擴展的像素
    top_left = (point33[0] - width_padding, point33[1])
    bottom_right = (point51[0] + width_padding, point51[1])

    return top_left, bottom_right


def draw_dotted_rectangle(img, pt1, pt2, color, dot_radius=2, dot_spacing=5):

    x1, y1 = pt1
    x2, y2 = pt2

    # 上邊
    for x in range(x1, x2, dot_spacing):
        cv2.circle(img, (x, y1), dot_radius, color, -1)
    # 右邊
    for y in range(y1, y2, dot_spacing):
        cv2.circle(img, (x2, y), dot_radius, color, -1)
    # 下邊
    for x in range(x2, x1, -dot_spacing):
        cv2.circle(img, (x, y2), dot_radius, color, -1)
    # 左邊
    for y in range(y2, y1, -dot_spacing):
        cv2.circle(img, (x1, y), dot_radius, color, -1)


# 讀取圖像
img = cv2.imread('Photo_human_2.jpeg')

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint8:
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)

            # 繪製面部特徵點（68點，紅色小圓點）
            for n in range(68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

            # 獲取額頭分區資訊
            forehead_outline_points, forehead_boxes, number_positions = detect_forehead(landmarks)

            # 繪製額頭輪廓紅點
            for point in forehead_outline_points:
                cv2.circle(img, point, 1, (0, 0, 255), -1)

            # 繪製三等分方框並標記序號
            box_colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255)]  # 紅色框
            for i, (box, pos) in enumerate(zip(forehead_boxes, number_positions)):
                cv2.rectangle(img, box[0], box[1], box_colors[i], 1)
                cv2.putText(img, str(i + 1), pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 繪製臉頰區域（紅色線條）
            left_cheek_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 9)]
            right_cheek_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(8, 17)]
            cv2.polylines(img, [np.array(left_cheek_points)], True, (0, 0, 255), 2)
            cv2.polylines(img, [np.array(right_cheek_points)], True, (0, 0, 255), 2)

            pt1, pt2 = detect_philtrum_box(landmarks)
            draw_dotted_rectangle(img, pt1, pt2, (0, 0, 255), 1, 4)



        cv2.imshow("Face Detection with Forehead and Philtrum Box", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()