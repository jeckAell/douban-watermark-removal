# 豆包水印去除工具

去除图片和视频中的豆包水印的本地工具。

## 功能特性

- 图片水印检测与去除
- 视频水印处理（提取帧→处理→合成）
- 批量处理目录中的所有图片
- 两种图像修复算法可选

## 安装

```bash
pip install -r requirements.txt
```

## 依赖

- Python 3.8+
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- Pillow >= 10.0.0

## 使用方法

### 处理单张图片

```bash
python watermark_remover.py -i input.jpg
```

### 手动指定水印位置（更精确）

```bash
# 指定角落位置
python watermark_remover.py -i input.jpg --pos bottom-right

# 指定精确区域 (x, y, w, h)
python watermark_remover.py -i input.jpg --pos 500,200,150,40
```

支持的角落位置：
- `bottom-right` - 右下角
- `bottom-left` - 左下角
- `top-right` - 右上角
- `top-left` - 左上角

### 保存到指定位置

```bash
python watermark_remover.py -i input.jpg -o output.jpg
```

### 批量处理图片

```bash
# 处理目录下所有图片
python watermark_remover.py -i ./images --batch

# 输出到指定目录
python watermark_remover.py -i ./images -o ./output --batch
```

### 处理视频

```bash
python watermark_remover.py -i video.mp4 -v
```

### 选择修复算法

- `--method teleanda` (默认): Telea 算法，速度快效果平滑
- `--method navier-stokes`: Navier-Stokes 算法，边缘保持更好

```bash
python watermark_remover.py -i input.jpg --method navier-stokes
```

## 工作原理

1. **水印检测**: 使用边缘检测和形态学操作定位浅色文字水印区域
2. **区域扩展**: 对检测到的区域进行小幅扩展确保完全覆盖
3. **图像修复**: 使用 OpenCV 的 INPAINT_TELEA 或 INPAINT_NS 算法填充

## 注意事项

- 水印检测基于豆包水印通常为白色/浅色文字的特征
- 视频处理时假设所有帧水印位置相同
- 处理视频会创建临时目录存储帧，处理完成后自动清理
- 建议处理前备份原始文件

## 限制

- 仅对白色/浅色水印效果较好
- 水印必须位于画面边缘区域（顶部或底部）
- 视频处理速度取决于视频长度和分辨率
