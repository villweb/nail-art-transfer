"""
指甲检测算法 V5 - 精确版
重点：避免美甲溢出指甲区域
"""
import cv2
import numpy as np


def refine_nail_detection_precise(hand_image: np.ndarray, landmarks, image_shape: tuple) -> list:
    """
    精确的指甲检测 - 保守策略
    宁可覆盖不足，不要溢出
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

        # 计算手指方向
        dx = fx - jx
        dy = fy - jy
        finger_length = np.sqrt(dx**2 + dy**2)

        if finger_length == 0:
            continue

        # 手指方向单位向量
        dir_x = dx / finger_length
        dir_y = dy / finger_length

        # 指甲中心位置（指尖往回5%，更保守）
        nail_center_x = int(fx - dir_x * finger_length * 0.05)
        nail_center_y = int(fy - dir_y * finger_length * 0.05)

        # 指甲大小（更保守的估算，避免溢出）
        nail_length = int(finger_length * 0.20)  # 从25%降到20%
        nail_width = int(nail_length * 0.70)     # 从80%降到70%

        # 不同手指的指甲大小调整（更保守）
        if fingertip_idx == 4:  # 拇指
            nail_length = int(nail_length * 1.1)  # 从1.2降到1.1
            nail_width = int(nail_width * 1.2)    # 从1.3降到1.2
        elif fingertip_idx == 8:  # 食指
            nail_length = int(nail_length * 1.0)  # 从1.05降到1.0
            nail_width = int(nail_width * 1.0)
        elif fingertip_idx == 20:  # 小指
            nail_length = int(nail_length * 0.85)  # 从0.9降到0.85
            nail_width = int(nail_width * 0.85)

        # 计算指甲旋转角度
        angle = np.degrees(np.arctan2(dir_y, dir_x))

        nail_info_list.append({
            'center': (nail_center_x, nail_center_y),
            'length': nail_length,
            'width': nail_width,
            'angle': angle,
            'direction': (dir_x, dir_y)
        })

    return nail_info_list
