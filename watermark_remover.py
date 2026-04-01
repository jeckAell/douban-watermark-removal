#!/usr/bin/env python3
"""
豆包水印去除工具
支持图片和视频中的水印检测与去除
"""

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image


class WatermarkDetector:
    """水印检测器"""

    def __init__(self):
        # 豆包水印通常为半透明白色文字，底部或顶部居中
        self.watermark_color_range = {
            'lower': np.array([200, 200, 200]),
            'upper': np.array([255, 255, 255])
        }
        # 半透明水印阈值（更低以检测淡色文字）
        self.semi_transparent_thresh = 150

    def detect_watermark_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测水印区域
        返回: (x, y, w, h) 或 None
        """
        h, w = image.shape[:2]

        # 转换到HSV更好地检测浅色区域
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 尝试多种方法检测水印

        # 方法1: 角落水印检测（右下/左下 - 半透明白色文字）
        corner_region = self._detect_corner_watermark(gray, image, h, w)
        if corner_region:
            return corner_region

        # 方法2: 底部区域检测（豆包水印常见位置）
        bottom_region = self._detect_bottom_watermark(gray, image, h, w)
        if bottom_region:
            return bottom_region

        # 方法3: 顶部区域检测
        top_region = self._detect_top_watermark(gray, image, h, w)
        if top_region:
            return top_region

        # 方法4: 基于边缘检测的水印检测
        edge_region = self._detect_edge_watermark(gray, image, h, w)
        if edge_region:
            return edge_region

        return None

    def _detect_bottom_watermark(self, gray: np.ndarray, image: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """检测底部水印（居中或角落）"""
        # 底部 25% 区域
        bottom_start = int(h * 0.75)
        bottom_roi = gray[bottom_start:, :]

        # 使用较低阈值检测半透明白色文字
        _, thresh = cv2.threshold(bottom_roi, self.semi_transparent_thresh, 255, cv2.THRESH_BINARY)

        # 找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # 水印在底部区域，高度适中
            if ch >= 5 and ch < h * 0.15 and cw >= 20:
                return (x, bottom_start + y, cw, ch)

        return None

    def _detect_top_watermark(self, gray: np.ndarray, image: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """检测顶部水印"""
        # 顶部 20% 区域
        top_end = int(h * 0.20)
        top_roi = gray[:top_end, :]

        _, thresh = cv2.threshold(top_roi, self.semi_transparent_thresh, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch >= 5 and ch < h * 0.12 and cw >= 20:
                return (x, y, cw, ch)

        return None

    def _detect_corner_watermark(self, gray: np.ndarray, image: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """检测角落水印（右下/左下/右上/左上）- 针对半透明水印"""
        # 角落区域大小（宽高的20-30%）
        corner_h = max(20, int(h * 0.18))
        corner_w = max(50, int(w * 0.25))

        corners = {
            'bottom-right': (h - corner_h, w - corner_w, corner_h, corner_w),
            'bottom-left': (h - corner_h, 0, corner_h, corner_w),
            'top-right': (0, w - corner_w, corner_h, corner_w),
            'top-left': (0, 0, corner_h, corner_w),
        }

        for name, (y, x, ch, cw) in corners.items():
            roi = gray[y:y+ch, x:x+cw]
            if roi.size == 0:
                continue

            # 检测ROI中是否有比周围略亮的区域（半透明文字特征）
            mean_val = roi.mean()
            max_val = roi.max()

            # 半透明文字：max值比mean高，但不是很亮
            if max_val > mean_val + 5 and max_val > 150:
                # 使用形态学检测文字笔画
                # 创建稍低阈值的mask
                _, mask = cv2.threshold(roi, max(140, mean_val + 10), 255, cv2.THRESH_BINARY)

                # 找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # 合并所有检测到的区域
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    # 取前几个小轮廓（文字笔画）
                    total_x, total_y, total_w, total_h = x, y, 0, 0
                    for cnt in contours[:10]:
                        bx, by, bcw, bch = cv2.boundingRect(cnt)
                        if bcw > 3 and bch > 3:  # 过滤噪点
                            total_x = min(total_x, x + bx)
                            total_y = min(total_y, y + by)
                            total_w = max(total_w, x + bx + bcw)
                            total_h = max(total_h, y + by + bch)

                    if total_w > total_x and total_h > total_y:
                        final_w = total_w - total_x
                        final_h = total_h - total_y
                        if final_w > 30 and final_h > 8:  # 最小水印尺寸
                            print(f"在 {name} 角检测到水印")
                            return (total_x, total_y, final_w, final_h)

        return None

    def _detect_edge_watermark(self, gray: np.ndarray, image: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """基于边缘检测水印"""
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 膨胀连接边缘
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:10]:  # 只检查前10个最大轮廓
            x, y, cw, ch = cv2.boundingRect(cnt)
            # 检查是否在边缘区域
            at_edge = (y < h * 0.15 or y + ch > h * 0.85)
            reasonable_size = (cw > w * 0.05 and ch < h * 0.15)

            if at_edge and reasonable_size:
                return (x, y, cw, ch)

        return None


class ImageInpainter:
    """图像修复器"""

    def __init__(self, method: str = 'telea'):
        """
        初始化修复器
        method: 'telea' 或 'navier-stokes'
        """
        self.method = method

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        修复图像
        image: BGR 格式
        mask: 白色区域表示需要修复
        """
        if self.method == 'telea':
            flags = cv2.INPAINT_TELEA
        else:
            flags = cv2.INPAINT_NS

        # 确保mask是uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # 确保mask是单通道
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        result = cv2.inpaint(image, mask, inpaintRadius=5, flags=flags)
        return result


class WatermarkRemover:
    """水印去除器"""

    def __init__(self, inpaint_method: str = 'telea'):
        self.detector = WatermarkDetector()
        self.inpainter = ImageInpainter(method=inpaint_method)

    def remove_watermark(self, image_path: str, output_path: str = None,
                         expand_mask: int = 5,
                         manual_region: Tuple[int, int, int, int] = None) -> bool:
        """
        去除图片水印
        manual_region: 手动指定区域 (x, y, w, h)
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return False

        # 优先使用手动指定的区域
        if manual_region:
            region = manual_region
            print(f"使用手动指定区域: {region}")
        else:
            # 检测水印
            region = self.detector.detect_watermark_region(image)

            if region is None:
                print(f"未检测到水印: {image_path}")
                return False

        x, y, w, h = region
        print(f"水印区域: ({x}, {y}, {w}, {h})")

        # 创建扩展mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1 = max(0, x - expand_mask)
        y1 = max(0, y - expand_mask)
        x2 = min(image.shape[1], x + w + expand_mask)
        y2 = min(image.shape[0], y + h + expand_mask)
        mask[y1:y2, x1:x2] = 255

        # 修复图像
        result = self.inpainter.inpaint(image, mask)

        # 保存结果
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_no_watermark{ext}"

        cv2.imwrite(output_path, result)
        print(f"已保存到: {output_path}")
        return True

    def remove_video_watermark(self, video_path: str, output_path: str = None) -> bool:
        """
        去除视频水印（处理所有关键帧）
        """
        if output_path is None:
            name, ext = os.path.splitext(video_path)
            output_path = f"{name}_no_watermark{ext}"

        # 创建临时目录
        temp_dir = f"/tmp/watermark_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误: 无法打开视频 {video_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

            # 提取帧
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            frame_paths = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_idx += 1

                if frame_idx % 100 == 0:
                    print(f"已提取 {frame_idx}/{total_frames} 帧")

            cap.release()
            print(f"共提取 {frame_idx} 帧")

            # 处理帧
            processed_dir = os.path.join(temp_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)

            # 检测水印区域（只检测第一帧，假设水印位置固定）
            sample_frame = cv2.imread(frame_paths[0])
            region = self.detector.detect_watermark_region(sample_frame)

            if region is None:
                print("未在视频中检测到水印")
                # 仍然创建视频但复制原始帧
                processed_frames = frame_paths
            else:
                x, y, w, h = region
                print(f"检测到水印区域: ({x}, {y}, {w}, {h})")

                # 创建固定mask
                mask = np.zeros(sample_frame.shape[:2], dtype=np.uint8)
                x1 = max(0, x - 5)
                y1 = max(0, y - 5)
                x2 = min(sample_frame.shape[1], x + w + 5)
                y2 = min(sample_frame.shape[0], y + h + 5)
                mask[y1:y2, x1:x2] = 255

                for i, frame_path in enumerate(frame_paths):
                    frame = cv2.imread(frame_path)
                    result = self.inpainter.inpaint(frame, mask)
                    output_frame_path = os.path.join(processed_dir, f"frame_{i:06d}.jpg")
                    cv2.imwrite(output_frame_path, result)

                    if (i + 1) % 100 == 0:
                        print(f"已处理 {i + 1}/{len(frame_paths)} 帧")

                processed_frames = [os.path.join(processed_dir, f"frame_{i:06d}.jpg") for i in range(len(frame_paths))]

            # 合成视频
            print("正在合成视频...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame_path in processed_frames:
                frame = cv2.imread(frame_path)
                out.write(frame)

            out.release()
            print(f"已保存到: {output_path}")
            return True

        finally:
            # 清理临时文件
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def process_batch(image_dir: str, output_dir: str = None,
                  pattern: str = "*.jpg", inpaint_method: str = 'telea') -> int:
    """
    批量处理图片
    """
    remover = WatermarkRemover(inpaint_method=inpaint_method)

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dir_path = Path(image_dir)
    image_files = list(dir_path.glob(pattern))

    # 也搜索png
    image_files.extend(list(dir_path.glob("*.png")))

    if not image_files:
        print(f"在 {image_dir} 中未找到匹配 {pattern} 的图片")
        return 0

    print(f"找到 {len(image_files)} 张图片")

    success_count = 0
    for img_path in image_files:
        if output_dir:
            output_path = os.path.join(output_dir, f"{img_path.stem}_no_watermark{img_path.suffix}")
        else:
            output_path = None

        if remover.remove_watermark(str(img_path), output_path):
            success_count += 1

    print(f"成功处理 {success_count}/{len(image_files)} 张图片")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="豆包水印去除工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例命令:
  # 处理单张图片
  python watermark_remover.py -i input.jpg

  # 处理图片并保存到指定位置
  python watermark_remover.py -i input.jpg -o output.jpg

  # 批量处理目录下所有图片
  python watermark_remover.py -i ./images --batch

  # 批量处理并输出到指定目录
  python watermark_remover.py -i ./images -o ./output --batch

  # 处理视频
  python watermark_remover.py -i video.mp4 -v

  # 使用Navier-Stokes算法修复
  python watermark_remover.py -i input.jpg --method navier-stokes
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='输入文件路径或目录')
    parser.add_argument('-o', '--output',
                        help='输出文件路径或目录')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理模式（当输入为目录时）')
    parser.add_argument('-v', '--video', action='store_true',
                        help='处理视频文件')
    parser.add_argument('--method', choices=['telea', 'navier-stokes'],
                        default='telea',
                        help='图像修复算法 (默认: teleanda)')
    parser.add_argument('--pattern', default='*.jpg',
                        help='批量处理时的文件匹配模式 (默认: *.jpg)')
    parser.add_argument('--pos',
                        help='手动指定水印位置，格式: x,y,w,h 或 corners (bottom-right,bottom-left,top-right,top-left)')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"错误: 输入路径不存在 {input_path}")
        sys.exit(1)

    # 解析手动位置参数
    manual_region = None
    if args.pos:
        # 支持角落名称
        corner_map = {
            'bottom-right': None,  # 需要根据图片尺寸计算
            'bottom-left': None,
            'top-right': None,
            'top-left': None,
        }
        if args.pos.lower() in corner_map:
            # 先读取图片获取尺寸
            img = cv2.imread(input_path)
            if img is not None:
                h, w = img.shape[:2]
                corner_h = int(h * 0.15)
                corner_w = int(w * 0.20)
                if args.pos.lower() == 'bottom-right':
                    manual_region = (w - corner_w, h - corner_h, corner_w, corner_h)
                elif args.pos.lower() == 'bottom-left':
                    manual_region = (0, h - corner_h, corner_w, corner_h)
                elif args.pos.lower() == 'top-right':
                    manual_region = (w - corner_w, 0, corner_w, corner_h)
                elif args.pos.lower() == 'top-left':
                    manual_region = (0, 0, corner_w, corner_h)
                print(f"使用角落位置: {args.pos} -> 区域 {manual_region}")
        elif ',' in args.pos:
            # x,y,w,h 格式
            try:
                parts = [int(x.strip()) for x in args.pos.split(',')]
                if len(parts) == 4:
                    manual_region = tuple(parts)
                    print(f"使用手动区域: {manual_region}")
            except ValueError:
                print(f"错误: 无效的区域格式 {args.pos}，应使用 x,y,w,h")
                sys.exit(1)

    remover = WatermarkRemover(inpaint_method=args.method)

    if os.path.isdir(input_path):
        # 批量处理（不支持手动区域）
        process_batch(input_path, output_path, args.pattern, args.method)
    elif args.video or input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 视频处理
        remover.remove_video_watermark(input_path, output_path)
    else:
        # 单张图片处理
        remover.remove_watermark(input_path, output_path, manual_region=manual_region)


if __name__ == '__main__':
    main()
