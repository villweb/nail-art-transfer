"""
指甲检测算法 - 参考商用方案
结合多种技术，提高稳定性
"""
import cv2
import numpy as np


def multi_method_nail_detection(hand_image: np.ndarray, fingertip_pos: tuple, finger_dir: tuple, finger_length: float, fingertip_idx: int) -> dict:
    """
    多方法融合的指甲检测
    参考：Perfect Corp YouCam Nail, O'2Nails
    """
    height, width = hand_image.shape[:2]
    fx, fy = fingertip_pos
    dir_x, dir_y = finger_dir
    
    # 提取指尖ROI（参考商用方案的局部处理策略）
    roi_size = int(finger_length * 0.5)
    x1 = max(0, fx - roi_size)
    y1 = max(0, fy - roi_size)
    x2 = min(width, fx + roi_size)
    y2 = min(height, fy + roi_size)
    
    roi = hand_image[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
        return None
    
    # 方法1: 颜色分析（参考 Perfect Corp）
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    # 指甲通常比皮肤颜色浅或深
    # 使用 LAB 的 L 通道（亮度）
    l_channel = roi_lab[:,:,0]
    
    # 自适应阈值（Otsu）
    _, thresh1 = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 方法2: 边缘检测（参考 O'2Nails）
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi_gray, 50, 150)
    
    # 方法3: 颜色聚类（参考 Nailbot）
    # K-means 分2类：指甲和皮肤
    roi_reshaped = roi.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(roi_reshaped, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 选择更亮的类别作为指甲
    brightness = np.mean(centers, axis=1)
    nail_label = np.argmax(brightness)
    thresh2 = (labels.flatten() == nail_label).reshape(roi.shape[:2]).astype(np.uint8) * 255
    
    # 融合多个方法的结果
    combined = cv2.bitwise_or(thresh1, thresh2)
    combined = cv2.bitwise_or(combined, edges)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 找轮廓
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 100 and len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (ex, ey), (ea, eb), eangle = ellipse
            
            # 转换回原图坐标
            center_x = int(ex + x1)
            center_y = int(ey + y1)
            
            # 使用手指方向作为角度
            finger_angle = np.degrees(np.arctan2(dir_y, dir_x))
            final_angle = finger_angle + 90
            
            return {
                'center': (center_x, center_y),
                'length': int(ea / 2),
                'width': int(eb / 2),
                'angle': final_angle
            }
    
    # 如果检测失败，使用估算
    return estimate_nail_position((fx, fy), (dir_x, dir_y), finger_length, fingertip_idx)


def estimate_nail_position(fingertip_pos: tuple, finger_dir: tuple, finger_length: float, fingertip_idx: int) -> dict:
    """
    基于关键点估算指甲位置（商用方案的备选策略）
    """
    fx, fy = fingertip_pos
    dir_x, dir_y = finger_dir
    
    # 指甲大小（参考商用应用的参数）
    nail_length = int(finger_length * 0.25)
    nail_width = int(nail_length * 0.75)
    
    # 不同手指调整（基于真实指甲比例）
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


def refine_nail_detection_final(hand_image: np.ndarray, landmarks, image_shape) -> list:
    """
    最终版本指甲检测 - 参考商用方案
    多方法融合 + 备选策略
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
        
        # 多方法融合检测
        nail_info = multi_method_nail_detection(
            hand_image, (fx, fy), (dir_x, dir_y), finger_length, fingertip_idx
        )
        
        nail_info_list.append(nail_info)
    
    return nail_info_list
