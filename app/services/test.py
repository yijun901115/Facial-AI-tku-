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
    """膚色狀態定義 - 简化版本"""
    NORMAL = "正常"
    DARK = "發黑"
    RED = "發紅"
    PALE = "發白"
    YELLOW = "發黃"
    CYAN_GREEN = "青綠"  # 偏綠的青色
    CYAN_BLACK = "青黑"  # 偏黑的青色


class FaceSkinAnalyzer:
    def __init__(self):
        self.face_app = None
        self.face_mesh = None
        self.diagnosis_results = {}
        self.face_regions = {}
        self.current_face_rect = None
        self.init_detector()

    def get_face_position_description(self, region):
        """获取面部区域的具体位置描述"""
        position_map = {
            FaceRegion.BRAIN: "额头中央",
            FaceRegion.LUNG: "额头上方",
            FaceRegion.HEART: "眉心区域",
            FaceRegion.LIVER: "右眉外侧",
            FaceRegion.SPLEEN: "鼻梁上方",
            FaceRegion.LEFT_KIDNEY: "左太阳穴",
            FaceRegion.RIGHT_KIDNEY: "右太阳穴",
            FaceRegion.LEFT_STOMACH: "左脸颊上方",
            FaceRegion.RIGHT_STOMACH: "右脸颊上方",
            FaceRegion.LEFT_LONG: "左脸颊下方",
            FaceRegion.RIGHT_LONG: "右脸颊下方",
            FaceRegion.BLADDER: "人中区域",
            FaceRegion.LEFT_EYE_WHITE: "左眼白",
            FaceRegion.RIGHT_EYE_WHITE: "右眼白",
            FaceRegion.CHIN: "下巴"
        }
        return position_map.get(region, region.value)

    def init_detector(self):
        """初始化檢測器"""
        try:
            # 屏蔽所有警告和日志信息
            warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
            warnings.filterwarnings("ignore", message=".*rcond.*")
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # 屏蔽TensorFlow和ONNX信息
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['GLOG_minloglevel'] = '3'  # 屏蔽ONNX Runtime日志

            # 屏蔽标准输出中的模型加载信息
            import sys
            import contextlib
            from io import StringIO

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

            try:
                # 临时屏蔽标准输出来隐藏模型加载信息
                f = StringIO()
                with contextlib.redirect_stdout(f):
                    self.face_app = insightface.app.FaceAnalysis(
                        name='buffalo_l',
                        providers=providers
                    )
                    self.face_app.prepare(ctx_id=ctx_id)

            except Exception as e:
                print(f"❌ InsightFace 初始化失敗: {e}")
                print("💡 嘗試僅使用 CPU 模式...")
                f = StringIO()
                with contextlib.redirect_stdout(f):
                    self.face_app = insightface.app.FaceAnalysis(
                        name='buffalo_l',
                        providers=['CPUExecutionProvider']
                    )
                    self.face_app.prepare(ctx_id=-1)


            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
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

        faces = self.safe_face_detection(image)
        if not faces:
            return []

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
            FaceRegion.BRAIN: (100, 100),  # 从80x80增加到100x100
            FaceRegion.LUNG: (35, 35),  # 从20x20增加到35x35
            FaceRegion.HEART: (35, 35),  # 从20x20增加到35x35
            FaceRegion.LIVER: (35, 35),  # 从20x20增加到35x35
            FaceRegion.SPLEEN: (35, 35),  # 从20x20增加到35x35
            FaceRegion.LEFT_KIDNEY: (35, 35),  # 从20x20增加到35x35
            FaceRegion.RIGHT_KIDNEY: (35, 35),  # 从20x20增加到35x35
            FaceRegion.LEFT_STOMACH: (35, 35),  # 从20x20增加到35x35
            FaceRegion.RIGHT_STOMACH: (35, 35),  # 从20x20增加到35x35
            FaceRegion.LEFT_LONG: (35, 35),  # 从20x20增加到35x35
            FaceRegion.RIGHT_LONG: (35, 35),  # 从20x20增加到35x35
            FaceRegion.BLADDER: (35, 35),  # 从20x20增加到35x35
            FaceRegion.LEFT_EYE_WHITE: (30, 15),  # 保持不变
            FaceRegion.RIGHT_EYE_WHITE: (30, 15),  # 保持不变
            FaceRegion.CHIN: (50, 40),  # 从40x30增加到50x40
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

    def analyze_cyan_subdivision_simple(self, mean_color, region=None):
        """
        简化版青色細分：只區分青綠和青黑
        """
        b, g, r = mean_color

        brightness = (r + g + b) / 3.0
        total_color = r + g + b

        if total_color == 0:
            return None

        green_ratio = g / total_color
        blue_ratio = b / total_color
        red_ratio = r / total_color

        # 青色系判斷基準
        is_cyan_base = (
                (blue_ratio > red_ratio and blue_ratio > 0.35) or
                (green_ratio > red_ratio and green_ratio > 0.35 and blue_ratio > 0.3)
        )

        if not is_cyan_base:
            return None

        # 核心判斷邏輯：亮度 + 綠藍比例
        gb_ratio = green_ratio / blue_ratio if blue_ratio > 0 else 0
        brightness_threshold = 85
        gb_threshold = 1.05

        # 青綠：較亮且綠色比例較高
        if brightness > brightness_threshold and gb_ratio > gb_threshold:
            return SkinCondition.CYAN_GREEN
        # 青黑：較暗或藍色比例較高
        else:
            return SkinCondition.CYAN_BLACK

    def diagnose_skin_condition(self, mean_color, region=None):
        """根據RGB值判斷膚色狀態 - 删除面部发黄，保留眼白发黄"""
        b, g, r = mean_color

        # 特殊處理眼白區域 - 保留眼白发黄识别
        if region in [FaceRegion.LEFT_EYE_WHITE, FaceRegion.RIGHT_EYE_WHITE]:
            total_color = r + g + b
            if total_color > 0:
                yellow_ratio = (r + g) / (2 * total_color)
                blue_ratio = b / total_color

                if yellow_ratio > 0.6 and blue_ratio < 0.25 and g > 120:
                    return SkinCondition.YELLOW

            return SkinCondition.NORMAL

        # 基本參數計算
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

        # 获取具体位置描述
        position_desc = self.get_face_position_description(region)

        # 面部区域异常检测 - 添加位置信息输出
        if brightness < 65:
            print(f"{region.value}({position_desc}): RGB({int(r)},{int(g)},{int(b)}) -> 發黑")
            return SkinCondition.DARK
        elif brightness > 200 and min_color > 150 and saturation < 0.1:
            print(f"{region.value}({position_desc}): RGB({int(r)},{int(g)},{int(b)}) -> 發白")
            return SkinCondition.PALE
        elif red_ratio > 0.50 and r > 175 and saturation > 0.17:
            print(f"{region.value}({position_desc}): RGB({int(r)},{int(g)},{int(b)}) -> 發紅")
            return SkinCondition.RED

        # 青色系判斷
        cyan_result = self.analyze_cyan_subdivision_simple(mean_color, region)
        if cyan_result:
            print(f"{region.value}({position_desc}): RGB({int(r)},{int(g)},{int(b)}) -> {cyan_result.value}")
            return cyan_result

        return SkinCondition.NORMAL

    def get_condition_colors(self):
        """简化版顏色映射"""
        return {
            SkinCondition.NORMAL: (0, 255, 0),  # 綠色
            SkinCondition.DARK: (0, 0, 139),  # 深藍
            SkinCondition.RED: (0, 0, 255),  # 紅色
            SkinCondition.PALE: (255, 255, 255),  # 白色
            SkinCondition.YELLOW: (0, 255, 255),  # 黃色
            SkinCondition.CYAN_GREEN: (255, 255, 0),  # 青綠色
            SkinCondition.CYAN_BLACK: (128, 128, 0),  # 青黑色
        }

    def generate_cyan_report(self):
        """生成简化版青色報告"""
        cyan_green_regions = []
        cyan_black_regions = []

        for region, condition in self.diagnosis_results.items():
            if condition == SkinCondition.CYAN_GREEN:
                cyan_green_regions.append(region.value)
            elif condition == SkinCondition.CYAN_BLACK:
                cyan_black_regions.append(region.value)

        total_cyan = len(cyan_green_regions) + len(cyan_black_regions)

        if total_cyan == 0:
            return None

        return {
            "青綠區域": cyan_green_regions,
            "青黑區域": cyan_black_regions,
            "青綠數量": len(cyan_green_regions),
            "青黑數量": len(cyan_black_regions),
            "總青色區域": total_cyan,
            "主要類型": "青綠" if len(cyan_green_regions) > len(cyan_black_regions) else "青黑"
        }

    def draw_face_regions(self, image):
        """在圖像上繪製面部區域"""
        if not self.face_regions:
            return image

        annotated_image = image.copy()
        condition_colors = self.get_condition_colors()

        for region, region_rect in self.face_regions.items():
            x, y, w, h = region_rect
            condition = self.diagnosis_results.get(region, SkinCondition.NORMAL)
            color = condition_colors.get(condition, (0, 255, 0))

            thickness = 3 if condition != SkinCondition.NORMAL else 2
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)

            label_text = condition.value
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            cv2.rectangle(annotated_image,
                          (x, y - text_height - 5),
                          (x + text_width + 5, y),
                          color, -1)

            text_color = (0, 0, 0) if condition in [SkinCondition.PALE, SkinCondition.YELLOW,
                                                    SkinCondition.CYAN_GREEN] else (255, 255, 255)
            cv2.putText(annotated_image, label_text, (x + 2, y - 2),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return annotated_image

    def draw_abnormal_regions_on_original(self, image):
        """在原圖上只標註異常區域"""
        if not self.face_regions:
            return image, 0

        annotated_image = image.copy()
        condition_colors = self.get_condition_colors()

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
                    "grid_analysis": None,
                    "cyan_report": None
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

            # 生成青色分析報告
            cyan_report = self.generate_cyan_report()

            annotated_base64 = self.image_to_base64(annotated_image)
            abnormal_only_base64 = self.image_to_base64(abnormal_only_image)
            grid_base64 = self.image_to_base64(grid_results['grid'])
            dark_blocks_base64 = self.image_to_base64(grid_results['dark_blocks'])

            all_regions = {region.value: condition.value for region, condition in self.diagnosis_results.items()}
            abnormal_regions = {region.value: condition.value for region, condition in
                                self.diagnosis_results.items() if condition != SkinCondition.NORMAL}

            print("分析完成！")
            if abnormal_count > 0:
                print(f"異常區域數量: {abnormal_count}")
            else:
                print("所有區域膚色狀態正常！！！")
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
                },
                "cyan_report": cyan_report
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
                "grid_analysis": None,
                "cyan_report": None
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





def simple_face_analysis(input_folder="images", output_folder="face_analysis_simple"):
    """
    简化版面部膚色分析，支持青綠青黑診斷
    """

    if not check_dependencies():
        print("❌ 請先安裝缺少的依賴包後再運行")
        return False

    os.makedirs(output_folder, exist_ok=True)

    print("🔧 初始化分析器...")
    analyzer = FaceSkinAnalyzer()

    if analyzer.face_app is None or analyzer.face_mesh is None:
        print("❌ 分析器初始化失敗")
        return False

    success_count = 0
    fail_count = 0
    cyan_statistics = {
        "青綠區域總數": 0,
        "青黑區域總數": 0,
        "有青色的圖像": []
    }

    print(f"📁 處理 {input_folder} 資料夾...")

    if not os.path.exists(input_folder):
        print(f"❌ 資料夾不存在: {input_folder}")
        return False

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"❌ 沒有找到圖像文件")
        return False

    print(f"📊 找到 {len(image_files)} 個圖像文件")

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"\n處理: {filename}")

        try:
            with open(img_path, 'rb') as f:
                image_data = f.read()

            base64_string = base64.b64encode(image_data).decode('utf-8')
            if filename.lower().endswith('.png'):
                base64_string = f"data:image/png;base64,{base64_string}"
            else:
                base64_string = f"data:image/jpeg;base64,{base64_string}"

            result = analyzer.analyze_from_base64(base64_string)

            if result["success"]:
                success_count += 1

                image_output_folder = os.path.join(output_folder, base_name)
                os.makedirs(image_output_folder, exist_ok=True)

                # 保存原始圖像
                original_path = os.path.join(image_output_folder, f"{base_name}_original.jpg")
                with open(original_path, 'wb') as f:
                    f.write(image_data)

                # 保存標註圖像
                if result["annotated_image"]:
                    annotated_path = os.path.join(image_output_folder, f"{base_name}_annotated.png")
                    save_base64_image(result["annotated_image"], annotated_path)

                if result.get("abnormal_only_image"):
                    abnormal_path = os.path.join(image_output_folder, f"{base_name}_abnormal_only.png")
                    save_base64_image(result["abnormal_only_image"], abnormal_path)

                # 保存分析結果
                json_path = os.path.join(image_output_folder, f"{base_name}_analysis.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    result_copy = result.copy()
                    result_copy["original_image"] = None
                    result_copy["annotated_image"] = None
                    result_copy["abnormal_only_image"] = None
                    result_copy["grid_analysis"] = None
                    json.dump(result_copy, f, ensure_ascii=False, indent=2)

                # 統計青色系
                cyan_report = result.get("cyan_report")
                if cyan_report:
                    cyan_statistics["青綠區域總數"] += cyan_report["青綠數量"]
                    cyan_statistics["青黑區域總數"] += cyan_report["青黑數量"]
                    cyan_statistics["有青色的圖像"].append({
                        "文件名": filename,
                        "青綠區域": cyan_report["青綠區域"],
                        "青黑區域": cyan_report["青黑區域"]
                    })

                print(f"成功: {filename}")

            else:
                fail_count += 1
                print(f"  ❌ 失敗: {filename} - {result['error']}")

        except Exception as e:
            fail_count += 1
            print(f"  ❌ 錯誤: {filename} - {str(e)}")

    # 生成最終報告
    final_report = {
        "處理統計": {
            "總圖像數": success_count + fail_count,
            "成功處理": success_count,
            "處理失敗": fail_count,
            "成功率": f"{success_count / (success_count + fail_count) * 100:.1f}%" if (
                                                                                                  success_count + fail_count) > 0 else "0%"
        },
        "青色系統計": cyan_statistics,
        "輸出資料夾": output_folder
    }

    report_path = os.path.join(output_folder, "simple_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 分析完成！")
    print(f"📊 統計:")
    print(f"  - 總圖像: {success_count + fail_count}")
    print(f"  - 成功處理: {success_count}")
    print(f"  - 處理失敗: {fail_count}")

    if cyan_statistics["青綠區域總數"] > 0 or cyan_statistics["青黑區域總數"] > 0:
        print(f"\n🔍 青色系統計:")
        print(f"  - 青綠區域總數: {cyan_statistics['青綠區域總數']}")
        print(f"  - 青黑區域總數: {cyan_statistics['青黑區域總數']}")
        print(f"  - 有青色異常的圖像: {len(cyan_statistics['有青色的圖像'])}")

        for img_info in cyan_statistics["有青色的圖像"][:3]:  # 只顯示前3個
            print(f"    └─ {img_info['文件名']}: 青綠{len(img_info['青綠區域'])}個 青黑{len(img_info['青黑區域'])}個")

    print(f"\n📁 結果保存在: {output_folder}")
    print(f"📄 報告: {report_path}")

    return True


def analyze_face_from_base64(base64_string):
    """便捷函數：從base64字符串分析面部膚色"""
    analyzer = FaceSkinAnalyzer()
    return analyzer.analyze_from_base64(base64_string)


def analyze_face_from_file(file_path):
    """便捷函數：從文件路徑分析面部膚色"""
    try:
        with open(file_path, 'rb') as f:
            image_data = f.read()

        base64_string = base64.b64encode(image_data).decode('utf-8')

        if file_path.lower().endswith('.png'):
            base64_string = f"data:image/png;base64,{base64_string}"
        elif file_path.lower().endswith(('.jpg', '.jpeg')):
            base64_string = f"data:image/jpeg;base64,{base64_string}"
        else:
            base64_string = f"data:image/png;base64,{base64_string}"

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
            "grid_analysis": None,
            "cyan_report": None
        }


# 使用示例
if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽TensorFlow信息

    # 屏蔽所有相关警告
    warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
    warnings.filterwarnings("ignore", message=".*onnxruntime.*")
    warnings.filterwarnings("ignore", message=".*rcond.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("面部膚色分析系統")

    # 處理圖像
    if os.path.exists("images"):
        image_count = len([f for f in os.listdir('images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\n📁 發現 'images' 資料夾，包含 {image_count} 張圖像")

        if image_count > 0:
            print("\n2️⃣ 開始處理圖像...")

            result = simple_face_analysis(
                input_folder="images",
                output_folder="face_analysis_simple"
            )

            if result:
                print(f"\n🎉 處理完成!")
            else:
                print(f"\n❌ 處理失敗")

        else:
            print("📂 'images' 資料夾為空，請添加圖像文件。")
    else:
        print("\n📂 請創建 'images' 資料夾並放入圖像文件。")
        print("系統將執行:")
        print("1. 青色邏輯測試")
        print("2. 面部區域識別")
        print("3. 青綠、青黑診斷")
        print("4. 生成標註圖像")
        print("5. 輸出分析報告")