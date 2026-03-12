"""
美甲换手 API
使用 MediaPipe 检测手部关键点，结合颜色和边缘检测精确识别指甲区域
"""
import os
import io
import uuid
import base64
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# 导入指甲检测模块
from nail_detector_precise import refine_nail_detection_precise

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="美甲换手 API", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# 下载手部关键点模型（首次运行时）
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    import urllib.request
    logger.info("下载手部关键点模型...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    logger.info("模型下载完成")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2
)
hand_landmarker = HandLandmarker.create_from_options(options)

# 存储结果的临时目录
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def detect_hand_landmarks(image_np: np.ndarray) -> Optional[list]:
    """检测手部关键点"""
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hand_landmarker.detect(mp_image)
    
    if not results.hand_landmarks:
        return None
    
    # 返回第一个检测到的手的关键点
    return results.hand_landmarks[0]


def detect_nail_region(hand_image: np.ndarray, fingertip_pos: tuple, finger_dir: tuple) -> tuple:
    """
    使用颜色分割和边缘检测精确识别指甲区域
    返回：(中心点, 长轴, 短轴, 角度)
    """
    height, width = hand_image.shape[:2]
    fx, fy = fingertip_pos
    dir_x, dir_y = finger_dir
    
    # 在指尖附近提取ROI区域
    roi_size = 150
    x1 = max(0, fx - roi_size)
    y1 = max(0, fy - roi_size)
    x2 = min(width, fx + roi_size)
    y2 = min(height, fy + roi_size)
    
    roi = hand_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # 转换到HSV空间
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 检测指甲区域（通常比皮肤颜色浅或深）
    # 皮肤颜色范围
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
    
    # 指甲通常比皮肤亮
    # 使用亮度阈值
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值检测指甲
    nail_mask = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 排除皮肤区域
    nail_mask = cv2.bitwise_and(nail_mask, cv2.bitwise_not(skin_mask))
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel)
    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_OPEN, kernel)
    
    # 找到最大的轮廓（指甲）
    contours, _ = cv2.findContours(nail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 椭圆拟合
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (ex, ey), (ea, eb), eangle = ellipse
        
        # 转换回原图坐标
        center_x = int(ex + x1)
        center_y = int(ey + y1)
        
        return (center_x, center_y), int(ea/2), int(eb/2), eangle
    
    return None


def get_nail_regions(landmarks, image_shape) -> list:
    """
    根据手部关键点计算每个手指指甲的精确形状和位置
    返回每个手指指甲的详细信息：(中心点, 长轴, 短轴, 旋转角度, 指尖方向)
    """
    height, width = image_shape[:2]
    nail_info_list = []
    
    # 每个手指的关键点索引 (指尖, 第二关节, 第一关节)
    finger_indices = [
        (4, 3, 2),   # 拇指
        (8, 7, 6),   # 食指
        (12, 11, 10), # 中指
        (16, 15, 14), # 无名指
        (20, 19, 18)  # 小指
    ]
    
    for fingertip_idx, second_joint_idx, first_joint_idx in finger_indices:
        fingertip = landmarks[fingertip_idx]
        second_joint = landmarks[second_joint_idx]
        first_joint = landmarks[first_joint_idx]
        
        # 指尖位置
        fx = int(fingertip.x * width)
        fy = int(fingertip.y * height)
        
        # 第二关节位置
        sx = int(second_joint.x * width)
        sy = int(second_joint.y * height)
        
        # 第一关节位置
        jx = int(first_joint.x * width)
        jy = int(first_joint.y * height)
        
        # 计算手指方向（从第一关节到指尖，更稳定）
        dx = fx - jx
        dy = fy - jy
        finger_length = np.sqrt(dx**2 + dy**2)
        
        if finger_length == 0:
            continue
        
        # 手指方向单位向量
        dir_x = dx / finger_length
        dir_y = dy / finger_length
        
        # 垂直于手指方向的向量
        perp_x = -dir_y
        perp_y = dir_x
        
        # 指甲中心位置（指尖往回一点点，约3%）
        nail_center_x = int(fx - dir_x * finger_length * 0.03)
        nail_center_y = int(fy - dir_y * finger_length * 0.03)
        
        # 指甲大小（更精确的估算）
        # 真实指甲约占手指长度的 25-30%
        nail_length = int(finger_length * 0.28)  # 沿手指方向
        nail_width = int(nail_length * 0.80)     # 垂直手指方向
        
        # 不同手指的指甲大小调整
        if fingertip_idx == 4:  # 拇指
            nail_length = int(nail_length * 1.2)
            nail_width = int(nail_width * 1.3)
        elif fingertip_idx == 8:  # 食指
            nail_length = int(nail_length * 1.05)
            nail_width = int(nail_width * 1.05)
        elif fingertip_idx == 20:  # 小指
            nail_length = int(nail_length * 0.9)
            nail_width = int(nail_width * 0.9)
        
        # 计算指甲旋转角度（与手指方向一致）
        angle = np.degrees(np.arctan2(dir_y, dir_x))
        
        nail_info_list.append({
            'center': (nail_center_x, nail_center_y),
            'length': nail_length,
            'width': nail_width,
            'angle': angle,
            'direction': (dir_x, dir_y)
        })
    
    return nail_info_list


def extract_nail_pattern(nail_art_image: np.ndarray, target_size: tuple) -> np.ndarray:
    """从美甲图片中提取图案"""
    # 调整大小到目标尺寸
    pattern = cv2.resize(nail_art_image, target_size)
    return pattern


def apply_nail_art_to_hand(
    hand_image: np.ndarray,
    nail_art_image: np.ndarray,
    nail_info_list: list
) -> np.ndarray:
    """
    将美甲图案精确贴合到每个手指的指甲上
    1. 先将美甲图片压缩到指甲大小
    2. 按指甲形状裁剪
    3. 完全覆盖指甲区域
    """
    result = hand_image.copy()
    import random
    
    for nail_info in nail_info_list:
        center_x, center_y = nail_info['center']
        nail_length = nail_info['length']
        nail_width = nail_info['width']
        angle = nail_info['angle']
        
        if nail_length <= 0 or nail_width <= 0:
            continue
        
        # 1. 先将美甲图片调整到指甲大小（椭圆形外接矩形）
        nail_size = max(nail_length, nail_width) * 2
        nail_art_resized = cv2.resize(nail_art_image, (nail_size, nail_size))
        
        # 2. 创建指甲形状的椭圆形遮罩
        mask = np.zeros((nail_size, nail_size), dtype=np.float32)
        center = nail_size // 2
        
        # 绘制椭圆形指甲遮罩
        cv2.ellipse(
            mask,
            (center, center),
            (int(nail_length), int(nail_width)),
            0, 0, 360,
            1.0, -1
        )
        
        # 3. 按遮罩裁剪美甲图案
        nail_pattern = np.zeros((nail_size, nail_size, 3), dtype=np.uint8)
        for c in range(3):
            nail_pattern[:, :, c] = (nail_art_resized[:, :, c] * mask).astype(np.uint8)
        
        # 4. 旋转美甲图案以匹配指甲角度
        rotation_matrix = cv2.getRotationMatrix2D(
            (center, center),
            angle + 90,  # 调整角度
            1.0
        )
        
        nail_pattern_rotated = cv2.warpAffine(
            nail_pattern,
            rotation_matrix,
            (nail_size, nail_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 旋转遮罩
        mask_rotated = cv2.warpAffine(
            mask,
            rotation_matrix,
            (nail_size, nail_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 5. 将美甲图案贴合到手指上
        paste_x = center_x - center
        paste_y = center_y - center
        
        img_h, img_w = result.shape[:2]
        
        # 应用美甲图案
        for y in range(nail_size):
            for x in range(nail_size):
                px = paste_x + x
                py = paste_y + y
                
                if 0 <= px < img_w and 0 <= py < img_h:
                    alpha = mask_rotated[y, x]
                    # 只在遮罩完全覆盖的区域应用（阈值提高到0.5）
                    if alpha > 0.5:
                        # 100% 覆盖指甲区域，完全替换原图
                        result[py, px] = nail_pattern_rotated[y, x]
    
    return result


def simple_color_transfer(hand_image: np.ndarray, nail_art_image: np.ndarray) -> np.ndarray:
    """
    简化版本：如果没有检测到手，直接在图像中央区域应用美甲
    用于演示和测试
    """
    result = hand_image.copy()
    height, width = hand_image.shape[:2]
    
    # 在图像下方区域（通常是手的位置）应用美甲
    roi_y1 = height // 2
    roi_y2 = height
    roi_x1 = width // 4
    roi_x2 = width * 3 // 4
    
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
    
    # 调整美甲图片大小
    nail_resized = cv2.resize(nail_art_image, (roi_width // 2, roi_height // 3))
    
    # 在 ROI 中心位置放置美甲
    center_x = (roi_x1 + roi_x2) // 2 - nail_resized.shape[1] // 2
    center_y = roi_y1 + roi_height // 2
    
    # 混合图像
    nh, nw = nail_resized.shape[:2]
    for y in range(nh):
        for x in range(nw):
            py = center_y + y
            px = center_x + x
            if 0 <= py < height and 0 <= px < width:
                # 简单混合
                result[py, px] = (
                    result[py, px] * 0.3 + nail_resized[y, x] * 0.7
                ).astype(np.uint8)
    
    return result


@app.post("/api/transfer")
async def transfer_nail_art(
    nail_art: UploadFile = File(...),
    hand: UploadFile = File(...)
):
    """
    美甲换手主接口
    
    - nail_art: 美甲款式图片
    - hand: 手部照片
    """
    try:
        # 读取图片
        nail_art_bytes = await nail_art.read()
        hand_bytes = await hand.read()
        
        # 转换为 numpy 数组
        nail_art_np = np.array(Image.open(io.BytesIO(nail_art_bytes)))
        hand_np = np.array(Image.open(io.BytesIO(hand_bytes)))
        
        # 转换 RGBA 为 RGB
        if nail_art_np.shape[-1] == 4:
            nail_art_np = cv2.cvtColor(nail_art_np, cv2.COLOR_RGBA2RGB)
        if hand_np.shape[-1] == 4:
            hand_np = cv2.cvtColor(hand_np, cv2.COLOR_RGBA2RGB)
        
        # 转换为 BGR (OpenCV 格式)
        nail_art_bgr = cv2.cvtColor(nail_art_np, cv2.COLOR_RGB2BGR)
        hand_bgr = cv2.cvtColor(hand_np, cv2.COLOR_RGB2BGR)
        
        logger.info(f"美甲图片尺寸: {nail_art_bgr.shape}")
        logger.info(f"手部图片尺寸: {hand_bgr.shape}")
        
        # 检测手部关键点
        landmarks = detect_hand_landmarks(hand_bgr)
        
        if landmarks:
            logger.info("检测到手部关键点")
            # 使用最终版本指甲检测 - 参考商用方案
            nail_regions = refine_nail_detection_precise(hand_bgr, landmarks, hand_bgr.shape)
            result = apply_nail_art_to_hand(hand_bgr, nail_art_bgr, nail_regions)
        else:
            logger.info("未检测到手部，使用简化处理")
            result = simple_color_transfer(hand_bgr, nail_art_bgr)
        
        # 转换回 RGB
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # 保存结果
        result_id = str(uuid.uuid4())
        result_path = os.path.join(RESULTS_DIR, f"{result_id}.jpg")
        Image.fromarray(result_rgb).save(result_path, quality=95)
        
        # 转换为 base64 返回
        buffer = io.BytesIO()
        Image.fromarray(result_rgb).save(buffer, format='JPEG', quality=95)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info(f"处理完成，结果ID: {result_id}")
        
        return JSONResponse({
            "success": True,
            "result_id": result_id,
            "result_url": f"data:image/jpeg;base64,{result_base64}"
        })
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "nail-art-transfer"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
