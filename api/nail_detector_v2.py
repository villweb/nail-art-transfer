"""
指甲检测算法 V2 - 参考成熟美甲机方案
使用 GrabCut 算法精确分割指甲区域
"""
import cv2
import numpy as np


def detect_nail_with_grabcut(hand_image: np.ndarray, fingertip_pos: tuple, finger_dir: tuple) -> dict:
    """
    使用 GrabCut 算法精确分割指甲区域
    
    参考：Nailbot 美甲机器人方案
    """
    height, width = hand_image.shape[:2]
    fx, fy = fingertip_pos
    dir_x, dir_y = finger_dir
    
    # 1. 确定ROI区域（指尖附近）
    roi_size = 150
    x1 = max(0, fx - roi_size)
    y1 = max(0, fy - roi_size)
    x2 = min(width, fx + roi_size)
    y2 = min(height, fy + roi_size)
    
    # 确保ROI有效
    if x2 - x1 < 50 or y2 - y1 < 50:
        return None
    
    roi = hand_image[y1:y2, x1:x2].copy()
    
    # 2. GrabCut 分割
    # 初始化 mask
    mask = np.zeros(roi.shape[:2], np.uint8)
    
    # 定义背景和前景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # 定义可能包含前景的矩形（指甲在中心区域）
    rect_x = int(roi.shape[1] * 0.3)
    rect_y = int(roi.shape[0] * 0.3)
    rect_w = int(roi.shape[1] * 0.4)
    rect_h = int(roi.shape[0] * 0.4)
    
    try:
        # 运行 GrabCut
        cv2.grabCut(roi, mask, (rect_x, rect_y, rect_w, rect_h), 
                    bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # 提取前景mask
        nail_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 3. 形态学操作优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 4. 找到指甲轮廓
        contours, _ = cv2.findContours(nail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 过滤太小的区域
        if cv2.contourArea(largest_contour) < 300:
            return None
        
        # 5. 椭圆拟合
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (ex, ey), (ea, eb), eangle = ellipse
            
            # 转换回原图坐标
            center_x = int(ex + x1)
            center_y = int(ey + y1)
            
            # 使用手指方向调整椭圆角度
            # 手指方向角度
            finger_angle = np.degrees(np.arctan2(dir_y, dir_x))
            
            # 椭圆角度应该与手指方向一致（+90度，因为椭圆长轴垂直于手指）
            final_angle = finger_angle + 90
            
            return {
                'center': (center_x, center_y),
                'length': int(ea / 2),
                'width': int(eb / 2),
                'angle': final_angle
            }
    except Exception as e:
        # GrabCut 失败，返回 None
        pass
    
    return None


def refine_nail_detection_v2(hand_image: np.ndarray, landmarks, image_shape) -> list:
    """
    改进的指甲检测 V2 - 使用 GrabCut 算法
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
        
        # 手指方向
        dx = fx - jx
        dy = fy - jy
        finger_length = np.sqrt(dx**2 + dy**2)
        
        if finger_length == 0:
            continue
        
        dir_x = dx / finger_length
        dir_y = dy / finger_length
        
        # 尝试 GrabCut 检测
        nail_info = detect_nail_with_grabcut(hand_image, (fx, fy), (dir_x, dir_y))
        
        if nail_info:
            nail_info_list.append(nail_info)
        else:
            # GrabCut 失败，使用估算（作为备选）
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
