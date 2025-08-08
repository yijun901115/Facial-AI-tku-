import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
import insightface
import mediapipe as mp
from enum import Enum
import json
import warnings


class FaceRegion(Enum):
    """面部區域定義"""
    BRAIN = "腦"
    LUNG = "肺"
    HEART = "心"
    LIVER = "肝"
    SPLEEN = "脾"
    LEFT_KIDNEY = "左腎"
    RIGHT_KIDNEY = "右腎"
    LEFT_STOMACH = "左胃"
    RIGHT_STOMACH = "右胃"
    LEFT_LONG = "左大腸"
    RIGHT_LONG = "右大腸"
    BLADDER = "膀胱"
    LEFT_EYE_WHITE = "左眼白"
    RIGHT_EYE_WHITE = "右眼白"
    CHIN = "下巴"


class SkinCondition(Enum):
    """膚色狀態定義"""
    NORMAL = "正常"
    DARK = "發黑"
    RED = "發紅"
    PALE = "發白"
    YELLOW = "發黃"
    CYAN = "發青"


class FaceSkinAnalyzer:
    def __init__(self):
        self.face_app = None
        self.face_mesh = None
        self.diagnosis_results = {}
        self.face_regions = {}
        self.current_face_rect = None
        self.init_detector()

    def init_detector(self):
        """初始化檢測器"""
        try:
            # 抑制警告
            warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

            # 檢查 PyTorch
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    print("🚀 檢測到CUDA支持，使用GPU加速")
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                else:
                    print("💻 使用CPU模式運行")
                    providers = ['CPUExecutionProvider']
                    ctx_id = -1
            except ImportError:
                print("⚠️  未安裝 PyTorch，使用 CPU 模式")
                providers = ['CPUExecutionProvider']
                ctx_id = -1

            # 初始化 InsightFace
            try:
                self.face_app = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=providers
                )
                self.face_app.prepare(ctx_id=ctx_id)
                print("✅ InsightFace 初始化成功")
            except Exception as e:
                print(f"❌ InsightFace 初始化失敗: {e}")
                print("💡 嘗試僅使用 CPU 模式...")
                self.face_app = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=-1)
                print("✅ InsightFace CPU 模式初始化成功")

            # 初始化 MediaPipe
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
            print("✅ MediaPipe 初始化成功")
            print("✅ 檢測器初始化完成")
            return True

        except Exception as e:
            print(f"❌ 檢測器初始化失敗: {e}")
            return False

    def base64_to_image(self, base64_string):
        """將base64字符串轉換為OpenCV圖像"""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]

            image_data = base64.b64decode(base64_string)
            image_pil = Image.open(io.BytesIO(image_data))

            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')

            image_array = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            return image_bgr
        except Exception as e:
            raise Exception(f"base64轉換圖像失敗: {e}")

    def image_to_base64(self, image):
        """將OpenCV圖像轉換為base64字符串"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            buffer = io.BytesIO()
            image_pil.save(buffer, format='PNG')
            buffer.seek(0)

            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            raise Exception(f"圖像轉換base64失敗: {e}")

    def safe_face_detection(self, image):
        """安全的人臉檢測"""
        try:
            if self.face_app is None:
                return []
            faces = self.face_app.get(image)
            return faces if faces else []
        except Exception as e:
            print(f"❌ 人臉檢測失敗: {e}")
            return []

    def safe_mediapipe_detection(self, image_rgb):
        """安全的MediaPipe檢測"""
        try:
            if self.face_mesh is None:
                return None
            results = self.face_mesh.process(image_rgb)
            return results
        except Exception as e:
            print(f"❌ MediaPipe 檢測失敗: {e}")
            return None

    def detect_faces_with_landmarks(self, image):
        """檢測人臉並返回特徵點"""
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # InsightFace 偵測人臉
        faces = self.safe_face_detection(image)
        if not faces:
            return []

        # MediaPipe 偵測細節
        results = self.safe_mediapipe_detection(img_rgb)
        if not results or not results.multi_face_landmarks:
            return []

        landmarks = results.multi_face_landmarks[0]

        face_data = []
        for face in faces:
            bbox = face.bbox
            face_rect = {
                'left': int(bbox[0]),
                'top': int(bbox[1]),
                'width': int(bbox[2] - bbox[0]),
                'height': int(bbox[3] - bbox[1])
            }

            face_data.append({
                'rect': face_rect,
                'landmarks': landmarks,
                'image_size': (w, h)
            })

        return face_data

    def get_all_face_regions(self, landmarks, image_size):
        """獲取所有面部區域"""
        w, h = image_size

        organ_landmarks = {
            FaceRegion.BRAIN: 10,
            FaceRegion.LUNG: 8,
            FaceRegion.HEART: 168,
            FaceRegion.LIVER: 197,
            FaceRegion.SPLEEN: 4,
            FaceRegion.LEFT_KIDNEY: 411,
            FaceRegion.RIGHT_KIDNEY: 187,
            FaceRegion.LEFT_STOMACH: 294,
            FaceRegion.RIGHT_STOMACH: 64,
            FaceRegion.LEFT_LONG: 330,
            FaceRegion.RIGHT_LONG: 101,
            FaceRegion.BLADDER: 164,
            FaceRegion.LEFT_EYE_WHITE: 33,
            FaceRegion.RIGHT_EYE_WHITE: 263,
            FaceRegion.CHIN: 175,
        }

        organ_crop_sizes = {
            FaceRegion.BRAIN: (80, 80),
            FaceRegion.LUNG: (20, 20),
            FaceRegion.HEART: (20, 20),
            FaceRegion.LIVER: (20, 20),
            FaceRegion.SPLEEN: (20, 20),
            FaceRegion.LEFT_KIDNEY: (20, 20),
            FaceRegion.RIGHT_KIDNEY: (20, 20),
            FaceRegion.LEFT_STOMACH: (20, 20),
            FaceRegion.RIGHT_STOMACH: (20, 20),
            FaceRegion.LEFT_LONG: (20, 20),
            FaceRegion.RIGHT_LONG: (20, 20),
            FaceRegion.BLADDER: (20, 20),
            FaceRegion.LEFT_EYE_WHITE: (30, 15),
            FaceRegion.RIGHT_EYE_WHITE: (30, 15),
            FaceRegion.CHIN: (40, 30),
        }

        regions = {}
        for organ, idx in organ_landmarks.items():
            pt = landmarks.landmark[idx]
            cx, cy = int(pt.x * w), int(pt.y * h)

            crop_w, crop_h = organ_crop_sizes.get(organ, (20, 20))
            x1 = max(cx - crop_w // 2, 0)
            y1 = max(cy - crop_h // 2, 0)
            x2 = min(cx + crop_w // 2, w)
            y2 = min(cy + crop_h // 2, h)

            regions[organ] = (x1, y1, x2 - x1, y2 - y1)

        return regions

    def analyze_skin_color_for_region(self, image, region_rect):
        """分析特定區域的膚色"""
        x, y, w, h = region_rect

        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)

        if w <= 0 or h <= 0:
            return (153, 134, 117)

        region = image[y:y + h, x:x + w]
        blurred = cv2.GaussianBlur(region, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        avg_brightness = np.mean(hsv[:, :, 2])

        if avg_brightness < 50:
            lower_skin = np.array([0, 5, 20])
            upper_skin = np.array([40, 255, 255])
        else:
            lower_skin = np.array([0, 10, 40])
            upper_skin = np.array([40, 255, 255])

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        if cv2.countNonZero(skin_mask) < (w * h * 0.1):
            lower_skin_loose = np.array([0, 8, 30])
            upper_skin_loose = np.array([50, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin_loose, upper_skin_loose)

            if cv2.countNonZero(skin_mask) == 0:
                skin_mask = np.ones((h, w), dtype=np.uint8) * 255

        skin_region = cv2.bitwise_and(blurred, blurred, mask=skin_mask)
        mean_color = cv2.mean(skin_region, skin_mask)

        return mean_color[:3]

    def diagnose_skin_condition(self, mean_color, region=None):
        """根據RGB值判斷膚色狀態"""
        b, g, r = mean_color

        # 特殊處理眼白區域
        if region in [FaceRegion.LEFT_EYE_WHITE, FaceRegion.RIGHT_EYE_WHITE]:
            total_color = r + g + b
            if total_color > 0:
                yellow_ratio = (r + g) / (2 * total_color)
                blue_ratio = b / total_color

                if yellow_ratio > 0.6 and blue_ratio < 0.25 and g > 120:
                    return SkinCondition.YELLOW

            return SkinCondition.NORMAL

        # 一般膚色區域診斷
        brightness = (r + g + b) / 3.0
        max_color = max(r, g, b)
        min_color = min(r, g, b)

        saturation = (max_color - min_color) / max_color if max_color > 0 else 0

        total_color = r + g + b
        if total_color > 0:
            red_ratio = r / total_color
            green_ratio = g / total_color
            blue_ratio = b / total_color
        else:
            red_ratio = green_ratio = blue_ratio = 0.33

        # 判斷膚色狀態 - 降低紅色敏感度
        if brightness < 70:
            return SkinCondition.DARK
        elif brightness > 200 and min_color > 150 and saturation < 0.1:
            return SkinCondition.PALE
        elif red_ratio > 0.48 and r > 170 and saturation > 0.15:
            return SkinCondition.RED
        elif (green_ratio > red_ratio and green_ratio > blue_ratio and
              green_ratio > 0.38 and g > 130):
            if red_ratio > 0.3:
                return SkinCondition.YELLOW
        elif (blue_ratio > red_ratio and blue_ratio > green_ratio and
              blue_ratio > 0.38 and b > 130):
            if green_ratio > 0.25:
                return SkinCondition.CYAN
        else:
            return SkinCondition.NORMAL

    def draw_face_regions(self, image):
        """在圖像上繪製面部區域"""
        if not self.face_regions:
            return image

        annotated_image = image.copy()

        condition_colors = {
            SkinCondition.NORMAL: (0, 255, 0),
            SkinCondition.DARK: (0, 0, 139),
            SkinCondition.RED: (0, 0, 255),
            SkinCondition.PALE: (255, 255, 255),
            SkinCondition.YELLOW: (0, 255, 255),
            SkinCondition.CYAN: (255, 255, 0)
        }

        for region, region_rect in self.face_regions.items():
            x, y, w, h = region_rect
            condition = self.diagnosis_results.get(region, SkinCondition.NORMAL)
            color = condition_colors.get(condition, (0, 255, 0))

            thickness = 3 if condition != SkinCondition.NORMAL else 2
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)

            label_text = region.value
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            cv2.rectangle(annotated_image,
                          (x, y - text_height - 5),
                          (x + text_width + 5, y),
                          color, -1)

            text_color = (0, 0, 0) if condition in [SkinCondition.PALE, SkinCondition.YELLOW, SkinCondition.CYAN] else (
            255, 255, 255)
            cv2.putText(annotated_image, label_text, (x + 2, y - 2),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return annotated_image

    def draw_abnormal_regions_on_original(self, image):
        """在原圖上只標註異常區域"""
        if not self.face_regions:
            return image, 0

        annotated_image = image.copy()

        condition_colors = {
            SkinCondition.DARK: (0, 0, 139),
            SkinCondition.RED: (0, 0, 255),
            SkinCondition.PALE: (128, 128, 128),
            SkinCondition.YELLOW: (0, 255, 255),
            SkinCondition.CYAN: (255, 255, 0)
        }

        abnormal_count = 0
        for region, region_rect in self.face_regions.items():
            condition = self.diagnosis_results.get(region, SkinCondition.NORMAL)

            if condition == SkinCondition.NORMAL:
                continue

            abnormal_count += 1
            x, y, w, h = region_rect
            color = condition_colors.get(condition, (0, 0, 255))

            thickness = 4
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)

            label_text = f"{region.value}: {condition.value}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            label_bg_height = text_height + 10
            label_bg_width = text_width + 10

            label_y = max(y - label_bg_height, label_bg_height)
            label_x = min(x, image.shape[1] - label_bg_width)

            cv2.rectangle(annotated_image,
                          (label_x, label_y - label_bg_height),
                          (label_x + label_bg_width, label_y),
                          color, -1)

            text_y = label_y - 5
            text_x = label_x + 5
            cv2.putText(annotated_image, label_text, (text_x, text_y),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return annotated_image, abnormal_count

    def grid_analysis(self, image):
        """圖像格狀分析"""
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
            'original': image,
            'grid': out_img,
            'dark_blocks': replace_img
        }

    def analyze_from_base64(self, base64_string):
        """從base64字符串分析圖像"""
        try:
            self.diagnosis_results.clear()
            self.face_regions.clear()
            self.current_face_rect = None

            image = self.base64_to_image(base64_string)
            face_data = self.detect_faces_with_landmarks(image)

            if not face_data:
                return {
                    "success": False,
                    "error": "未能檢測到面部特徵點。\n\n請確保：\n• 臉部完整且清晰可見\n• 光線充足且均勻\n• 避免過暗或逆光\n• 正對鏡頭\n\n調整後重新拍攝或選擇照片。",
                    "original_image": base64_string,
                    "annotated_image": None,
                    "abnormal_only_image": None,
                    "overall_color": None,
                    "region_results": None,
                    "grid_analysis": None
                }

            face_info = face_data[0]
            landmarks = face_info['landmarks']
            face_rect = face_info['rect']
            image_size = face_info['image_size']

            self.current_face_rect = (face_rect['left'], face_rect['top'],
                                      face_rect['width'], face_rect['height'])

            self.face_regions = self.get_all_face_regions(landmarks, image_size)

            # 分析每個區域
            for region, region_rect in self.face_regions.items():
                mean_color = self.analyze_skin_color_for_region(image, region_rect)
                condition = self.diagnose_skin_condition(mean_color, region)
                self.diagnosis_results[region] = condition

            overall_color = self.analyze_skin_color_for_region(image, self.current_face_rect)
            annotated_image = self.draw_face_regions(image)
            abnormal_only_image, abnormal_count = self.draw_abnormal_regions_on_original(image)
            grid_results = self.grid_analysis(image)

            annotated_base64 = self.image_to_base64(annotated_image)
            abnormal_only_base64 = self.image_to_base64(abnormal_only_image)
            grid_base64 = self.image_to_base64(grid_results['grid'])
            dark_blocks_base64 = self.image_to_base64(grid_results['dark_blocks'])

            all_regions = {region.value: condition.value for region, condition in self.diagnosis_results.items()}
            abnormal_regions = {region.value: condition.value for region, condition in
                                self.diagnosis_results.items() if condition != SkinCondition.NORMAL}

            return {
                "success": True,
                "error": None,
                "original_image": base64_string,
                "annotated_image": annotated_base64,
                "abnormal_only_image": abnormal_only_base64,
                "abnormal_count": abnormal_count,
                "overall_color": {
                    "r": int(overall_color[2]),
                    "g": int(overall_color[1]),
                    "b": int(overall_color[0]),
                    "hex": f"#{int(overall_color[2]):02X}{int(overall_color[1]):02X}{int(overall_color[0]):02X}"
                },
                "all_region_results": all_regions,
                "region_results": abnormal_regions,
                "grid_analysis": {
                    "grid_image": grid_base64,
                    "dark_blocks_image": dark_blocks_base64
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"分析過程中發生錯誤：{str(e)}",
                "original_image": base64_string,
                "annotated_image": None,
                "abnormal_only_image": None,
                "overall_color": None,
                "region_results": None,
                "grid_analysis": None
            }


def save_base64_image(base64_string, output_path):
    """將base64字符串保存為圖像文件"""
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)

        with open(output_path, 'wb') as f:
            f.write(image_data)

        return True
    except Exception as e:
        print(f"保存圖像失敗：{e}")
        return False


def check_dependencies():
    """檢查系統依賴包安裝情況"""


    dependencies = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'pillow': 'PIL',
        'insightface': 'insightface',
        'mediapipe': 'mediapipe'
    }

    missing_packages = []

    for package_name, import_name in dependencies.items():
        try:
            __import__(import_name)

        except ImportError:
            print(f"❌ {package_name} - 未安裝")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n⚠️  缺少依賴包: {', '.join(missing_packages)}")
        print("\n💡 安裝命令:")
        print("pip install " + " ".join(missing_packages))
        return False
    else:

        return True


def direct_face_analysis_and_annotation(input_folder="images", output_folder="face_analysis_results"):
    """
    直接在原圖上識別各個器官部位並進行膚色分析
    """
    print("=== 開始直接面部膚色分析與標註 ===")

    # 檢查依賴包
    if not check_dependencies():
        print("❌ 請先安裝缺少的依賴包後再運行")
        return {
            "success": False,
            "error": "缺少必要的依賴包",
            "processing_summary": {"total_images": 0, "successful_analyses": 0, "failed_analyses": 0}
        }

    os.makedirs(output_folder, exist_ok=True)

    # 初始化分析器
    print("🔧 初始化分析器...")
    analyzer = FaceSkinAnalyzer()

    # 檢查初始化是否成功
    if analyzer.face_app is None or analyzer.face_mesh is None:
        print("❌ 分析器初始化失敗，無法繼續")
        return {
            "success": False,
            "error": "分析器初始化失敗",
            "processing_summary": {"total_images": 0, "successful_analyses": 0, "failed_analyses": 0}
        }

    # 處理統計
    success_count = 0
    fail_count = 0
    all_results = []

    print(f"📁 開始處理 {input_folder} 資料夾中的圖像...")

    # 檢查輸入資料夾
    if not os.path.exists(input_folder):
        print(f"❌ 輸入資料夾不存在: {input_folder}")
        return {
            "success": False,
            "error": f"輸入資料夾不存在: {input_folder}",
            "processing_summary": {"total_images": 0, "successful_analyses": 0, "failed_analyses": 0}
        }

    # 遍歷圖像文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"❌ {input_folder} 資料夾中沒有找到圖像文件")
        return {
            "success": False,
            "error": "沒有找到圖像文件",
            "processing_summary": {"total_images": 0, "successful_analyses": 0, "failed_analyses": 0}
        }

    print(f"📊 找到 {len(image_files)} 個圖像文件")

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"處理: {filename}")

        try:
            # 讀取並轉換圖像為base64
            with open(img_path, 'rb') as f:
                image_data = f.read()

            base64_string = base64.b64encode(image_data).decode('utf-8')
            if filename.lower().endswith('.png'):
                base64_string = f"data:image/png;base64,{base64_string}"
            else:
                base64_string = f"data:image/jpeg;base64,{base64_string}"

            # 執行面部分析
            result = analyzer.analyze_from_base64(base64_string)

            if result["success"]:
                success_count += 1

                # 創建該圖像的輸出資料夾
                image_output_folder = os.path.join(output_folder, base_name)
                os.makedirs(image_output_folder, exist_ok=True)

                # 保存原始圖像
                original_path = os.path.join(image_output_folder, f"{base_name}_original.jpg")
                with open(original_path, 'wb') as f:
                    f.write(image_data)

                # 保存所有區域標註圖像
                if result["annotated_image"]:
                    annotated_path = os.path.join(image_output_folder, f"{base_name}_all_regions_annotated.png")
                    save_base64_image(result["annotated_image"], annotated_path)

                # 保存只標註異常區域的圖像
                if result.get("abnormal_only_image"):
                    abnormal_only_path = os.path.join(image_output_folder, f"{base_name}_abnormal_only.png")
                    save_base64_image(result["abnormal_only_image"], abnormal_only_path)

                # 保存格狀分析圖像
                if result["grid_analysis"]:
                    grid_folder = os.path.join(image_output_folder, "grid_analysis")
                    os.makedirs(grid_folder, exist_ok=True)

                    grid_path = os.path.join(grid_folder, f"{base_name}_grid.png")
                    dark_blocks_path = os.path.join(grid_folder, f"{base_name}_dark_blocks.png")

                    save_base64_image(result["grid_analysis"]["grid_image"], grid_path)
                    save_base64_image(result["grid_analysis"]["dark_blocks_image"], dark_blocks_path)

                # 保存分析結果為JSON
                json_path = os.path.join(image_output_folder, f"{base_name}_analysis_result.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    # 移除base64圖像數據以減小JSON文件大小
                    result_copy = result.copy()
                    result_copy["original_image"] = None
                    result_copy["annotated_image"] = None
                    result_copy["abnormal_only_image"] = None
                    result_copy["grid_analysis"] = None

                    json.dump(result_copy, f, ensure_ascii=False, indent=2)

                print(f"  ✅ 成功處理: {filename}")

                # 顯示分析結果
                if result.get("abnormal_count", 0) > 0:
                    print(f"    發現 {result['abnormal_count']} 個異常區域:")
                    for region, condition in result["region_results"].items():
                        print(f"      {region}: {condition}")
                else:
                    print(f"    ✅ 所有區域膚色狀態正常")

                # 添加到總結果
                all_results.append({
                    "filename": filename,
                    "success": True,
                    "abnormal_count": result.get("abnormal_count", 0),
                    "abnormal_regions": result["region_results"],
                    "all_regions": result["all_region_results"],
                    "overall_color": result["overall_color"],
                    "output_folder": image_output_folder
                })

            else:
                fail_count += 1
                print(f"  ❌ 處理失敗: {filename} - {result['error']}")

                all_results.append({
                    "filename": filename,
                    "success": False,
                    "error": result['error']
                })

        except Exception as e:
            fail_count += 1
            print(f"  ❌ 處理失敗: {filename} - {str(e)}")

            all_results.append({
                "filename": filename,
                "success": False,
                "error": str(e)
            })

    # 生成統計數據
    total_abnormal_regions = 0
    abnormal_images = []
    organ_statistics = {}

    for result in all_results:
        if result.get("success", False):
            abnormal_count = result.get("abnormal_count", 0)
            total_abnormal_regions += abnormal_count

            if abnormal_count > 0:
                abnormal_images.append(result)

            # 統計各器官異常情況
            for region, condition in result.get("abnormal_regions", {}).items():
                if region not in organ_statistics:
                    organ_statistics[region] = {"count": 0, "conditions": {}}
                organ_statistics[region]["count"] += 1
                if condition not in organ_statistics[region]["conditions"]:
                    organ_statistics[region]["conditions"][condition] = 0
                organ_statistics[region]["conditions"][condition] += 1

    # 生成最終報告
    final_report = {
        "processing_summary": {
            "total_images": success_count + fail_count,
            "successful_analyses": success_count,
            "failed_analyses": fail_count,
            "success_rate": success_count / (success_count + fail_count) * 100 if (
                                                                                              success_count + fail_count) > 0 else 0
        },
        "analysis_summary": {
            "images_with_abnormalities": len(abnormal_images),
            "images_normal": success_count - len(abnormal_images),
            "total_abnormal_regions": total_abnormal_regions,
            "abnormality_rate": len(abnormal_images) / success_count * 100 if success_count > 0 else 0
        },
        "organ_statistics": organ_statistics,
        "abnormal_images": abnormal_images,
        "all_results": all_results,
        "output_folder": output_folder
    }

    # 保存最終報告
    report_path = os.path.join(output_folder, "face_analysis_final_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 面部膚色分析完成！")
    print(f"📊 處理統計:")
    print(f"  - 總圖像數: {success_count + fail_count}")
    print(f"  - 成功分析: {success_count}")
    print(f"  - 分析失敗: {fail_count}")
    print(f"  - 成功率: {final_report['processing_summary']['success_rate']:.1f}%")
    print(f"  - 有異常的圖像: {len(abnormal_images)}")
    print(f"  - 正常圖像: {success_count - len(abnormal_images)}")
    print(f"  - 總異常區域數: {total_abnormal_regions}")
    print(f"  - 異常率: {final_report['analysis_summary']['abnormality_rate']:.1f}%")

    if organ_statistics:
        print(f"\n📋 器官異常統計:")
        for organ, stats in organ_statistics.items():
            print(f"  - {organ}: {stats['count']} 次異常")
            for condition, count in stats['conditions'].items():
                print(f"    └─ {condition}: {count} 次")

    print(f"\n📁 結果保存在: {output_folder}")
    print(f"📄 最終報告: {report_path}")

    if abnormal_images:
        print(f"\n⚠️ 發現異常的圖像:")
        for img_result in abnormal_images[:10]:  # 只顯示前10個
            filename = img_result['filename']
            abnormal_count = img_result['abnormal_count']
            abnormal_regions = img_result['abnormal_regions']
            print(f"  - {filename} ({abnormal_count} 個異常區域):")
            for region, condition in abnormal_regions.items():
                print(f"    └─ {region}: {condition}")

        if len(abnormal_images) > 10:
            print(f"  ... 還有 {len(abnormal_images) - 10} 張圖像有異常（詳見報告）")

    return final_report


def analyze_face_from_base64(base64_string):
    """便捷函數：從base64字符串分析面部膚色"""
    analyzer = FaceSkinAnalyzer()
    return analyzer.analyze_from_base64(base64_string)


def analyze_face_from_file(file_path):
    """便捷函數：從文件路徑分析面部膚色"""
    try:
        # 讀取圖像文件
        with open(file_path, 'rb') as f:
            image_data = f.read()

        # 轉換為base64
        base64_string = base64.b64encode(image_data).decode('utf-8')

        # 添加適當的前綴
        if file_path.lower().endswith('.png'):
            base64_string = f"data:image/png;base64,{base64_string}"
        elif file_path.lower().endswith(('.jpg', '.jpeg')):
            base64_string = f"data:image/jpeg;base64,{base64_string}"
        else:
            base64_string = f"data:image/png;base64,{base64_string}"

        # 分析
        analyzer = FaceSkinAnalyzer()
        return analyzer.analyze_from_base64(base64_string)

    except Exception as e:
        return {
            "success": False,
            "error": f"讀取文件失敗：{str(e)}",
            "original_image": None,
            "annotated_image": None,
            "abnormal_only_image": None,
            "overall_color": None,
            "region_results": None,
            "grid_analysis": None
        }


# 使用示例
if __name__ == "__main__":
    # 抑制ONNX Runtime的CUDA警告
    os.environ['OMP_NUM_THREADS'] = '1'
    warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
    warnings.filterwarnings("ignore", message=".*onnxruntime.*")




    # 檢查輸入資料夾
    if os.path.exists("images"):
        image_count = len([f for f in os.listdir('images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\n📁 偵測到 'images' 資料夾，包含 {image_count} 張圖像")

        if image_count > 0:
            print("\n🚀 開始自動執行面部分析流程...")

            # 執行直接面部分析
            final_result = direct_face_analysis_and_annotation(
                input_folder="images",
                output_folder="face_analysis_results"
            )

            if final_result.get("success", True):  # 如果沒有明確失敗，認為成功
                print(f"\n🎉 面部膚色分析流程完成!")
            else:
                print(f"\n❌ 分析失敗: {final_result.get('error', '未知錯誤')}")

        else:
            print("📂 'images' 資料夾為空，請添加圖像文件後再試。")
    else:
        print("\n📂 請先創建 'images' 資料夾並放入圖像文件。")
        print("然後重新運行此腳本，系統將自動執行:")
        print("1. 面部區域識別")
        print("2. 膚色分析")
        print("3. 異常標註")
        print("4. 生成完整報告")

