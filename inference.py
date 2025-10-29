#!/usr/bin/env python3
"""
批次处理脚本 - 通过单独的Python进程处理每个批次
解决内存泄漏问题
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import tempfile
import shutil

def get_video_info(video_path):
    """获取视频信息"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

def split_video_into_batches(video_path, batch_size, output_dir):
    """将视频分割成多个批次文件（保留音频）"""
    import cv2
    import subprocess

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    batch_files = []
    batch_num = 0

    for start_frame in range(0, total_frames, batch_size):
        end_frame = min(start_frame + batch_size, total_frames)
        batch_file = os.path.join(output_dir, f"batch_{batch_num:03d}.mp4")

        # 计算时间戳
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        # 使用ffmpeg分割视频（保留音频）
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libopenh264',  # 使用可用的OpenH264编码器
            '-c:a', 'aac',          # 音频编码
            '-g', '30',             # GOP大小，确保关键帧
            '-avoid_negative_ts', 'make_zero',
            '-y',  # 覆盖输出文件
            batch_file
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            batch_files.append(batch_file)
            batch_num += 1
            print(f"✅ 创建批次 {batch_num}: {batch_file} (帧 {start_frame}-{end_frame-1}, 时间 {start_time:.2f}s-{start_time+duration:.2f}s)")
        except subprocess.CalledProcessError as e:
            print(f"❌ 创建批次 {batch_num} 失败: {e}")
            print(f"ffmpeg错误输出: {e.stderr}")
            print(f"ffmpeg命令: {' '.join(cmd)}")
            continue

    return batch_files

def process_single_batch(source_path, batch_video_path, output_dir, batch_name):
    """处理单个批次"""
    cmd = [
        sys.executable, "inference_optimized.py",
        "-s", source_path,
        "-d", batch_video_path,
        "-o", output_dir,
        "-n", batch_name
    ]

    print(f"🚀 处理批次: {batch_name}")
    print(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 批次 {batch_name} 处理成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 批次 {batch_name} 处理失败:")
        print(f"错误输出: {e.stderr}")
        return False

def merge_batch_results(batch_output_files, final_output_path, original_video_path):
    """合并批次结果（保留音频）"""
    import subprocess

    if not batch_output_files:
        print("❌ 没有批次输出文件可以合并")
        return False

    # 创建文件列表
    file_list_path = final_output_path + ".txt"
    with open(file_list_path, 'w') as f:
        for batch_file in batch_output_files:
            if os.path.exists(batch_file):
                f.write(f"file '{batch_file}'\n")
            else:
                print(f"⚠️ 批次文件不存在: {batch_file}")

    # 使用ffmpeg合并视频
    temp_output = final_output_path + ".temp.mp4"
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', file_list_path,
        '-vf', 'setpts=PTS-STARTPTS',  # 重置时间戳，修复黑屏问题
        '-c:v', 'libx264',      # 使用H.264编码器
        '-crf', '18',           # 高质量编码
        '-preset', 'fast',      # 编码速度
        '-c:a', 'copy',         # 音频直接复制
        '-y', temp_output
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 视频合并完成: {temp_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 视频合并失败: {e}")
        os.remove(file_list_path)
        return False

    # 添加音频
    cmd = [
        'ffmpeg', '-i', temp_output,
        '-i', original_video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y', final_output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"🎉 最终视频（带音频）保存到: {final_output_path}")

        # 清理临时文件
        os.remove(temp_output)
        os.remove(file_list_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 添加音频失败: {e}")
        # 如果添加音频失败，至少保留无音频版本
        os.rename(temp_output, final_output_path)
        os.remove(file_list_path)
        print(f"⚠️ 保存无音频版本: {final_output_path}")
        return True

def main():
    parser = argparse.ArgumentParser(description="批次处理LivePortrait")
    parser.add_argument("-s", "--source", required=True, help="源图像路径")
    parser.add_argument("-d", "--driving", required=True, help="驱动视频路径")
    parser.add_argument("-o", "--output_dir", required=True, help="输出目录")
    parser.add_argument("-n", "--output_name", required=True, help="输出文件名")
    parser.add_argument("--batch_size", type=int, default=3000, help="每批次帧数")
    parser.add_argument("--temp_dir", help="临时目录（默认使用系统临时目录）")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建临时目录
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="liveportrait_batch_")

    try:
        print(f"📊 开始批次处理...")
        print(f"源图像: {args.source}")
        print(f"驱动视频: {args.driving}")
        print(f"批次大小: {args.batch_size}")
        print(f"临时目录: {temp_dir}")

        # 获取视频信息
        total_frames, fps = get_video_info(args.driving)
        print(f"视频总帧数: {total_frames}, FPS: {fps}")

        # 分割视频
        print("🔪 分割视频到批次...")
        batch_files = split_video_into_batches(args.driving, args.batch_size, temp_dir)

        # 处理每个批次
        batch_output_files = []
        for i, batch_file in enumerate(batch_files):
            batch_name = f"{args.output_name}_batch_{i:03d}"
            batch_output_dir = os.path.join(temp_dir, f"output_batch_{i:03d}")
            os.makedirs(batch_output_dir, exist_ok=True)

            # 处理单个批次
            success = process_single_batch(args.source, batch_file, batch_output_dir, batch_name)

            if success:
                # 查找输出文件
                expected_output = os.path.join(batch_output_dir, f"{batch_name}.mp4")
                if os.path.exists(expected_output):
                    batch_output_files.append(expected_output)
                else:
                    print(f"⚠️ 未找到批次输出文件: {expected_output}")

            # 清理批次输入文件以节省空间
            if os.path.exists(batch_file):
                os.remove(batch_file)

        # 合并结果
        if batch_output_files:
            final_output_path = os.path.join(args.output_dir, f"{args.output_name}.mp4")
            print(f"🔗 合并 {len(batch_output_files)} 个批次结果...")
            merge_batch_results(batch_output_files, final_output_path, args.driving)
        else:
            print("❌ 没有成功的批次结果可以合并")

    finally:
        # 清理临时目录
        if not args.temp_dir:  # 只有当使用系统临时目录时才清理
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 清理临时目录: {temp_dir}")
            except Exception as e:
                print(f"⚠️ 清理临时目录失败: {e}")

if __name__ == "__main__":
    main()
