"""
指甲区域精确检测算法
参考自动美甲机的技术思路
"""
import cv2
import numpy as np


def detect_nail_by_color_and_edge(hand_image: np.ndarray, fingertip_pos: tuple, finger_dir: tuple) -> dict:
    """
    使用多种方法精确识别指甲区域
    """
    height, width = hand_image.shape[:2]
    fx, fy = fingertip_pos
    dir_x, dir_y = finger_dir
    
    # 在指尖附近提取ROI区域
    roi_size = 200
    x1 = max(0, fx - roi_size)
    y1 = max(0, fy - roi_size)
    x2 = min(width, fx + roi_size)
    y2 = min(height, fy + roi_size)
    
    roi = hand_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # 转换到不同颜色空间
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 方法1: HSV颜色分割（浅色指甲）
    lower_light1 = np.array([0, 0, 180])
    upper_light1 = np.array([180, 80, 255])
    nail_mask1 = cv2.inRange(roi_hsv, lower_light1, upper_light1)
    
    # 方法2: LAB空间亮度检测
    l_channel = roi_lab[:,:,0]
    _, nail_mask2 = cv2.threshold(l_channel, 180, 255, cv2.THRESH_BINARY)
    
    # 方法3: 自适应阈值
    nail_mask3 = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    
    # 合并多个检测结果
    nail_mask = cv2.bitwise_or(nail_mask1, nail_mask2)
    nail_mask = cv2.bitwise_or(nail_mask, nail_mask3)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 边缘检测增强
    edges = cv2.Canny(roi_gray, 50, 150)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 找到轮廓
    contours, _ = cv2.findContours(nail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 过滤太小的轮廓
    min_area = 200
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return None
    
    # 选择最大的轮廓
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # 椭圆拟合
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (ex, ey), (ea, eb), eangle = ellipse
        
        # 转换回原图坐标
        center_x = int(ex + x1)
        center_y = int(ey + y1)
        
        # ea, eb 是直径，需要除以2
        return {
            'center': (center_x, center_y),
            'length': int(ea / 2),
            'width': int(eb / 2),
            'angle': eangle
        }
    
    return None


def refine_nail_detection(hand_image: np.ndarray, landmarks, image_shape) -> list:
    """
    改进的指甲检测：基于手部关键点，使用更精确的参数
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
        second_joint = landmarks[second_joint_idx]
        first_joint = landmarks[first_joint_idx]
        
        # 指尖位置
        fx = int(fingertip.x * width)
        fy = int(fingertip.y * height)
        
        # 第一关节位置
        jx = int(first_joint.x * width)
        jy = int(first_joint.y * height)
        
        # 手指方向
        dx = fx - jx
        dy = fy - jy
        finger_length = np.sqrt(dx**2 + dy**2)
        
        if finger_length == 0:
            continue
        
        dir_x = dx / finger_length
        dir_y = dy / finger_length
        
        # 指甲参数（优化后）
        # 真实指甲约占手指长度的 25-30%
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
        
        # 指甲中心（指尖往回一点点）
        nail_center_x = int(fx - dir_x * finger_length * 0.02)
        nail_center_y = int(fy - dir_y * finger_length * 0.02)
        
        angle = np.degrees(np.arctan2(dir_y, dir_x))
        
        nail_info_list.append({
            'center': (nail_center_x, nail_center_y),
            'length': nail_length,
            'width': nail_width,
            'angle': angle + 90
        })
    
    return nail_info_list
