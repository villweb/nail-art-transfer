# 美甲换手智能体

将美甲款式图片应用到真实手部照片上的 AI 应用。

## 项目结构

```
nail-art-transfer/
├── app/           # 前端 (React + Vite)
├── api/           # 后端 (FastAPI + MediaPipe)
└── README.md
```

## 快速开始

### 1. 启动后端 API

```bash
cd api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

后端将在 http://localhost:8000 启动

### 2. 启动前端

```bash
cd app
npm install
npm run dev
```

前端将在 http://localhost:3000 启动

### 3. 使用

1. 打开 http://localhost:3000
2. 上传美甲款式图片
3. 上传手部照片
4. 点击"开始换美甲"
5. 查看结果并下载

## 技术栈

- **前端**: React + TypeScript + Vite
- **后端**: FastAPI + Python
- **AI**: MediaPipe (手部关键点检测) + OpenCV (图像处理)

## 功能

- ✅ 手部关键点自动检测
- ✅ 五个手指指甲区域定位
- ✅ 美甲图案智能贴图
- ✅ 椭圆形指甲遮罩
- ✅ 实时预览和结果对比
- ✅ 结果下载

## 注意事项

- 照片尽量清晰，手部完整可见效果更好
- 美甲图片建议使用纯色背景
- 目前为本地演示版本，后续可接入更强大的云端 AI API
