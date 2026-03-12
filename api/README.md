# 美甲换手 API

使用 MediaPipe 检测手部关键点，定位指甲区域，将美甲图案应用到手上。

## 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

服务将在 http://localhost:8000 启动。

## API 接口

### POST /api/transfer

上传美甲图片和手部照片，返回换甲后的图片。

**请求：**
- `nail_art`: 美甲款式图片（multipart/form-data）
- `hand`: 手部照片（multipart/form-data）

**响应：**
```json
{
  "success": true,
  "result_id": "uuid",
  "result_url": "data:image/jpeg;base64,..."
}
```

### GET /health

健康检查接口。
