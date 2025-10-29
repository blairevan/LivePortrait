# coding: utf-8

"""
The entrance of humans - Optimized Version with Batch Processing and Memory Management
"""

import os
import os.path as osp
import tyro
import subprocess
import psutil
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items()
                          if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def check_system_resources():
    """检查系统资源"""
    try:
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)

        print(f"系统内存: {memory_gb:.1f}GB")
        print(f"当前内存使用: {memory.percent:.1f}%")

        if memory.percent > 90:
            print("⚠️ 警告: 系统内存使用率过高，建议关闭其他程序")
            return False
        return True
    except Exception as e:
        print(f"无法获取系统资源信息: {e}")
        return True


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # 检查系统资源
    if not check_system_resources():
        print("系统资源不足，是否继续? (y/N): ", end="")
        if input().lower() != 'y':
            return

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg "
            "(including ffmpeg and ffprobe) before running this script. "
            "https://ffmpeg.org/download.html"
        )

    fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    print("🚀 启动优化版本的LivePortrait...")
    print("📊 启用批处理和内存管理功能")

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    live_portrait_pipeline.execute(args)


if __name__ == "__main__":
    main()
