"""
生成演示图片：美甲款式 + 手部照片
"""
from PIL import Image, ImageDraw
import os

# 创建演示目录
demo_dir = os.path.join(os.path.dirname(__file__), "demo")
os.makedirs(demo_dir, exist_ok=True)

# 1. 生成美甲款式图片（彩色渐变 + 装饰）
nail_art = Image.new('RGB', (400, 400), (255, 255, 255))
draw = ImageDraw.Draw(nail_art)

# 渐变背景
for i in range(400):
    color = (
        int(255 - i * 0.3),
        int(100 + i * 0.2),
        int(200 - i * 0.1)
    )
    draw.line([(0, i), (400, i)], fill=color)

# 添加装饰点
for _ in range(50):
    import random
    x = random.randint(50, 350)
    y = random.randint(50, 350)
    r = random.randint(5, 15)
    color = (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255)
    )
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

# 添加闪粉效果
for _ in range(100):
    x = random.randint(0, 400)
    y = random.randint(0, 400)
    draw.point((x, y), fill=(255, 255, 255))

nail_art_path = os.path.join(demo_dir, "nail_art_demo.jpg")
nail_art.save(nail_art_path, quality=95)
print(f"✅ 美甲款式图片已生成: {nail_art_path}")

# 2. 生成手部照片（简化的手掌轮廓）
hand = Image.new('RGB', (600, 800), (240, 220, 200))
draw = ImageDraw.Draw(hand)

# 手掌主体（肉色）
palm_color = (255, 220, 185)
draw.ellipse([150, 400, 450, 700], fill=palm_color)

# 五个手指
fingers = [
    ([80, 350, 150, 550], -15),   # 拇指
    ([170, 150, 230, 400], 0),    # 食指
    ([260, 120, 320, 400], 0),    # 中指
    ([350, 150, 410, 400], 0),    # 无名指
    ([440, 200, 500, 420], 10),   # 小指
]

for finger_coords, angle in fingers:
    # 手指主体
    draw.ellipse(finger_coords, fill=palm_color)
    
    # 指甲区域（白色椭圆）
    nail_y = finger_coords[1] + 20
    nail_x = (finger_coords[0] + finger_coords[2]) // 2
    nail_w = (finger_coords[2] - finger_coords[0]) // 2
    nail_h = nail_w * 1.2
    draw.ellipse([
        nail_x - nail_w//2,
        nail_y,
        nail_x + nail_w//2,
        nail_y + nail_h
    ], fill=(255, 240, 230), outline=(200, 180, 160))

hand_path = os.path.join(demo_dir, "hand_demo.jpg")
hand.save(hand_path, quality=95)
print(f"✅ 手部照片已生成: {hand_path}")

print("\n演示图片已准备完毕！")
print(f"- 美甲款式: {nail_art_path}")
print(f"- 手部照片: {hand_path}")
