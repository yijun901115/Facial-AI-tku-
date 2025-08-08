import cv2
import numpy as np
import os
import insightface
import mediapipe as mp

# ---------- 初始化 ----------
face_app = insightface.app.FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# ---------- 圖片資料夾設定 ----------
input_folder = "images"
output_folder = "output_organs"
os.makedirs(output_folder, exist_ok=True)

# ---------- 對應圖上的臟腑部位與 landmark ----------
organ_landmarks = {
    "brain": 10,
    "lung": 8,
    "heart": 168,
    "liver": 197,
    "spleen": 4,
    "Left_kidney": 411,
    "Right_kidney": 187,
    "Left_stomach": 294,
    "Right_stomach": 64,
    "Left_long": 330,
    "Right_long": 101,
    "bladder": 164,
}

organ_crop_sizes = {
    "brain": (80, 80),  # 腦
    "lung": (20, 20),   # 肺
    "heart": (20, 20),  # 心
    "liver": (20, 20),  # 肝
    "spleen": (20, 20), # 脾
    "Left_kidney": (20, 20),  # 左腎
    "Right_kidney": (20, 20), # 右腎
    "Left_stomach": (20, 20),  # 左胃
    "Right_stomach": (20, 20), # 右胃
    "Left_long": (20, 20),  # 左大腸
    "Right_long": (20, 20), # 右大腸
    "bladder": (20, 20),   # 膀胱或子宮
}

# ---------- 圖像格狀分析 ----------
def grid_analysis(input_folder='output_organs', output_folder='processed'):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.bmp')):
            path = os.path.join(input_folder, filename)
            image = cv2.imread(path)
            if image is None:
                continue

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

            base = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(output_folder, f'original_{base}.jpg'), image)
            cv2.imwrite(os.path.join(output_folder, f'grid_{base}.jpg'), out_img)
            cv2.imwrite(os.path.join(output_folder, f'dark_blocks_{base}.jpg'), replace_img)
            print(f"已處理：{filename}")

# ---------- 批次處理每張圖 ----------
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # InsightFace 偵測人臉
    faces = face_app.get(img)
    if not faces:
        print(f"{filename}：❌ InsightFace 未偵測到人臉")
        continue

    # MediaPipe 偵測細節
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        print(f"{filename}：❌ MediaPipe 未偵測到臉部點位")
        continue

    landmarks = results.multi_face_landmarks[0]

    for organ, idx in organ_landmarks.items():
        pt = landmarks.landmark[idx]
        cx, cy = int(pt.x * w), int(pt.y * h)

        crop_w, crop_h = organ_crop_sizes.get(organ, (64, 64))
        x1 = max(cx - crop_w // 2, 0)
        y1 = max(cy - crop_h // 2, 0)
        x2 = min(cx + crop_w // 2, w)
        y2 = min(cy + crop_h // 2, h)

        cropped = img[y1:y2, x1:x2]
        save_name = f"{os.path.splitext(filename)[0]}_{organ}.jpg"
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, cropped)

print("✅ 批次臟腑區域裁切完成，圖片儲存在 output_organs/")

# ---------- 執行圖像格狀分析 ----------
grid_analysis(input_folder="output_organs", output_folder="processed")
