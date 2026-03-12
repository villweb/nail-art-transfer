"""
指甲检测算法 V3 - 基于关键点 + 局部区域分析
最稳定的方案
"""
import cv2
import numpy as np


def detect_nail_in_roi(hand_image: np.ndarray, fingertip_pos: tuple, finger_dir: tuple) -> dict:
    """
    在指尖局部区域检测指甲
    使用颜色分析和椭圆拟合
    """
    height, width = hand_image.shape[:2]
    fx, fy = fingertip_pos
    dir_x, dir_y = finger_dir
    
    # 提取指尖附近的小区域（减少干扰）
    roi_size = 100
    x1 = max(0, fx - roi_size)
    y1 = max(0, fy - roi_size)
    x2 = min(width, fx + roi_size)
    y2 = min(height, fy + roi_size)
    
    roi = hand_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # 转换到灰度图
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    
    # 自适应阈值（区分指甲和皮肤）
    thresh = cv2.adaptiveThreshold(
        roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 找到最大的轮廓（指甲）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 过滤太小的轮廓
    if cv2.contourArea(largest_contour) < 100:
        return None
    
    # 椭圆拟合
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (ex, ey), (ea, eb), eangle = ellipse
        
        # 转换回原图坐标
        center_x = int(ex + x1)
        center_y = int(ey + y1)
        
        # 使用手指方向作为椭圆角度
        finger_angle = np.degrees(np.arctan2(dir_y, dir_x))
        final_angle = finger_angle + 90
        
        return {
            'center': (center_x, center_y),
            'length': int(ea / 2),
            'width': int(eb / 2),
            'angle': final_angle
        }
    
    return None


def estimate_nail_from_keypoints(fingertip_pos: tuple, finger_dir: tuple, finger_length: float, fingertip_idx: int) -> dict:
    """
    基于关键点估算指甲位置和大小（备选方案）
    """
    fx, fy = fingertip_pos
    dir_x, dir_y = finger_dir
    
    # 指甲大小估算
    nail_length = int(finger_length * 0.25)
    nail_width = int(nail_length * 0.75)
    
    # 不同手指调整
    if fingertip_idx == 4:  # 拇指
        nail_length = int(nail_length * 1.3)
        nail_width = int(nail_width * 1.4)
    elif fingertip_idx == 8:  # 食指
        nail_length = int(nail_length * 1.1)
        nail_width = int(nail_width * 1.1)
    elif fingertip_idx == 12:  # 中指
        nail_length = int(nail_length * 1.05)
        nail_width = int(nail_width * 1.05)
    elif fingertip_idx == 20:  # 小指
        nail_length = int(nail_length * 0.85)
        nail_width = int(nail_width * 0.85)
    
    # 指甲中心（指尖往回一点）
    nail_center_x = int(fx - dir_x * finger_length * 0.02)
    nail_center_y = int(fy - dir_y * finger_length * 0.02)
    
    # 角度
    angle = np.degrees(np.arctan2(dir_y, dir_x)) + 90
    
    return {
        'center': (nail_center_x, nail_center_y),
        'length': nail_length,
        'width': nail_width,
        'angle': angle
    }


def refine_nail_detection_v3(hand_image: np.ndarray, landmarks, image_shape) -> list:
    """
    改进的指甲检测 V3 - 基于关键点 + 局部区域分析
    最稳定的方案
    """
    height, width = image_shape[:2]
    nail_info_list = []
    
    finger_indices = [
        (4, 3, 2, "拇指"),
        (8, 7, 6, "食指"),
        (12, 11, 10, "中指"),
        (16, 15, 14, "无名指"),
        (20, 19, 18, "小指")
    ]
    
    for fingertip_idx, second_joint_idx, first_joint_idx, name in finger_indices:
        fingertip = landmarks[fingertip_idx]
        first_joint = landmarks[first_joint_idx]
        
        # 指尖位置
        fx = int(fingertip.x * width)
        fy = int(fingertip.y * height)
        
        # 第一关节位置
        jx = int(first_joint.x * width)
        jy = int(first_joint.y * height)
        
        # 手指方向和长度
        dx = fx - jx
        dy = fy - jy
        finger_length = np.sqrt(dx**2 + dy**2)
        
        if finger_length == 0:
            continue
        
        dir_x = dx / finger_length
        dir_y = dy / finger_length
        
        # 尝试在ROI内检测指甲
        nail_info = detect_nail_in_roi(hand_image, (fx, fy), (dir_x, dir_y))
        
        if nail_info:
            nail_info_list.append(nail_info)
        else:
            # ROI检测失败，使用关键点估算
            nail_info = estimate_nail_from_keypoints(
                (fx, fy), (dir_x, dir_y), finger_length, fingertip_idx
            )
            nail_info_list.append(nail_info)
    
    return nail_info_list
