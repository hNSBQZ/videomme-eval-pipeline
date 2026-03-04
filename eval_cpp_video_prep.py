"""
视频帧采样与临时 JPG 保存。

帧采样逻辑与 evalkit 的 encode_video() 完全对齐：
  - 解码器优先级：decord → ffmpeg → torchvision
  - 短视频（≤ MAX_NUM_FRAMES 秒）：1fps
  - 长视频（> MAX_NUM_FRAMES 秒）：0.1s 粒度候选帧 → uniform_sample 到 MAX_NUM_FRAMES 帧
"""
import os
import shutil
import warnings
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from eval_cpp_config import MAX_NUM_FRAMES, MAX_FPS, FRAME_TEMP_DIR

logger = logging.getLogger(__name__)

# 尝试导入 decord
try:
    from decord import VideoReader, cpu as decord_cpu
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

# 尝试导入 torchvision（可选 fallback）
try:
    import torch
    from torchvision import io as tv_io
    _HAS_TORCHVISION = True
except Exception:
    torch = None
    tv_io = None
    _HAS_TORCHVISION = False


# ==================== 帧采样核心逻辑 ====================

def _uniform_sample(seq, n):
    """均匀采样：从 seq 中等间隔取 n 个元素（取每段中点）。"""
    gap = len(seq) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [seq[i] for i in idxs]


def _sample_frame_indices(
    num_total_frames: int,
    avg_fps: float,
    max_num_frames: int,
) -> Tuple[List[int], List[float]]:
    """
    根据视频元信息计算采样帧下标和时间戳。

    策略：
      - 长视频（duration > max_num_frames）：0.1s 粒度生成候选帧 → uniform_sample 到 max_num_frames
      - 短视频（duration ≤ max_num_frames）：1fps，每秒取一帧
    """
    duration = num_total_frames / avg_fps

    if duration > max_num_frames:
        step = 0.1
        num_steps = int(duration / step)
        timestamps = [round(i * step, 1) for i in range(num_steps)]
        frame_idx = [min(int(ts * avg_fps), num_total_frames - 1) for ts in timestamps]
        if len(frame_idx) > max_num_frames:
            frame_idx = _uniform_sample(frame_idx, max_num_frames)
            timestamps = _uniform_sample(timestamps, max_num_frames)
    else:
        int_duration = int(duration)
        frame_idx = [int(i * avg_fps) for i in range(int_duration)]
        timestamps = [float(i) for i in range(int_duration)]

    return frame_idx, timestamps


# ==================== Backend: decord ====================

def _load_frames_decord(
    video_path: str, max_num_frames: int, max_fps: float,
) -> Tuple[List[Image.Image], List[int], float, int]:
    """用 decord 加载帧。返回 (frames, frame_idx, avg_fps, total_frames)。"""
    if not _HAS_DECORD:
        return [], [], 0.0, 0
    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
    except Exception as e:
        warnings.warn(f"[decord] cannot open: {video_path}, {e}")
        return [], [], 0.0, 0

    try:
        avg_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        frame_idx, _ = _sample_frame_indices(total_frames, avg_fps, max_num_frames)
        if not frame_idx:
            return [], [], avg_fps, total_frames
        frames_np = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in frames_np]
        return frames, frame_idx, avg_fps, total_frames
    except Exception as e:
        warnings.warn(f"[decord] decode failed: {video_path}, {e}")
        return [], [], 0.0, 0


# ==================== Backend: ffmpeg CLI ====================

def _load_frames_ffmpeg(
    video_path: str, max_num_frames: int, max_fps: float,
) -> Tuple[List[Image.Image], List[int], float, int]:
    """用 ffmpeg 命令行抽帧（fallback #1）。"""
    if max_num_frames <= 0:
        max_num_frames = 1
    target_fps = float(max_fps) if max_fps and max_fps > 0 else 1.0

    if not os.path.exists(video_path):
        return [], [], 0.0, 0

    tmp_dir = tempfile.mkdtemp(prefix="cpp_eval_ffmpeg_")
    try:
        pattern = os.path.join(tmp_dir, "frame_%05d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={target_fps}",
            "-q:v", "2",
            "-vframes", str(max_num_frames),
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            warnings.warn("[ffmpeg] ffmpeg not found on PATH")
            return [], [], 0.0, 0
        except subprocess.CalledProcessError as e:
            warnings.warn(f"[ffmpeg] failed: {video_path}, {e}")
            return [], [], 0.0, 0

        frame_files = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        if not frame_files:
            return [], [], 0.0, 0

        frames, frame_idx = [], []
        for idx, fp in enumerate(frame_files):
            try:
                with Image.open(fp) as img:
                    frames.append(img.convert("RGB").copy())
                frame_idx.append(idx)
            except Exception as e:
                warnings.warn(f"[ffmpeg] read frame failed: {fp}, {e}")

        return frames, frame_idx, target_fps, len(frames)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ==================== Backend: torchvision ====================

def _load_frames_torchvision(
    video_path: str, max_num_frames: int, max_fps: float,
) -> Tuple[List[Image.Image], List[int], float, int]:
    """用 torchvision.io.read_video 加载帧（fallback #2）。"""
    if not _HAS_TORCHVISION:
        return [], [], 0.0, 0

    try:
        with torch.no_grad():
            video, _audio, info = tv_io.read_video(
                video_path, start_pts=0.0, end_pts=None,
                pts_unit="sec", output_format="TCHW",
            )
    except Exception as e:
        warnings.warn(f"[torchvision] read_video failed: {video_path}, {e}")
        return [], [], 0.0, 0

    total_frames = int(video.size(0))
    if total_frames == 0:
        return [], [], 0.0, 0

    video_fps = float(info.get("video_fps", 0.0) or 0.0)
    if video_fps <= 0:
        video_fps = 25.0

    duration = total_frames / video_fps
    if max_fps > 0:
        n_by_fps = int(duration * max_fps + 0.5)
        nframes = max(1, min(max_num_frames, total_frames, n_by_fps or max_num_frames))
    else:
        nframes = max(1, min(max_num_frames, total_frames))

    with torch.no_grad():
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        sampled = video[idx]
        frames_np = sampled.permute(0, 2, 3, 1).contiguous().to(torch.uint8).cpu().numpy()

    frames = [Image.fromarray(v).convert("RGB") for v in frames_np]
    return frames, idx.tolist(), video_fps, total_frames


# ==================== 统一入口 ====================

def encode_video(
    video_path: str,
    max_num_frames: int = MAX_NUM_FRAMES,
    max_fps: float = MAX_FPS,
) -> List[Image.Image]:
    """
    从视频中采样帧，返回 PIL.Image 列表。

    解码器优先级：decord → ffmpeg → torchvision
    """
    # 1) decord
    frames, frame_idx, avg_fps, total_frames = _load_frames_decord(video_path, max_num_frames, max_fps)
    backend = "decord"

    # 2) ffmpeg fallback
    if not frames:
        frames, frame_idx, avg_fps, total_frames = _load_frames_ffmpeg(video_path, max_num_frames, max_fps)
        backend = "ffmpeg" if frames else backend

    # 3) torchvision fallback
    if not frames:
        frames, frame_idx, avg_fps, total_frames = _load_frames_torchvision(video_path, max_num_frames, max_fps)
        backend = "torchvision" if frames else "none"

    if not frames:
        warnings.warn(f"[encode_video] no frames decoded from: {video_path}")
        return []

    sample_fps = 0.0
    if total_frames > 0 and avg_fps > 0 and frame_idx:
        sample_fps = len(frame_idx) / max(total_frames, 1e-6) * avg_fps

    logger.info(
        f"Video: {video_path}, backend={backend}, sample_fps={sample_fps:.3f}, "
        f"frames={len(frames)}, total={total_frames}, raw_fps={avg_fps:.3f}"
    )
    return frames


# ==================== 帧文件管理 ====================

def save_frames_as_jpg(
    frames: List[Image.Image],
    video_id: str,
    tmp_dir: str = FRAME_TEMP_DIR,
    quality: int = 95,
) -> List[str]:
    """
    将 PIL.Image 帧列表保存为临时 JPG 文件。

    目录结构：{tmp_dir}/{video_id}/frame_000.jpg, frame_001.jpg, ...
    返回文件绝对路径列表（有序）。
    """
    video_dir = os.path.join(tmp_dir, video_id)
    os.makedirs(video_dir, exist_ok=True)

    paths: List[str] = []
    for idx, frame in enumerate(frames):
        path = os.path.join(video_dir, f"frame_{idx:03d}.jpg")
        frame.save(path, format="JPEG", quality=quality)
        paths.append(os.path.abspath(path))
    return paths


def cleanup_frames(video_id: str, tmp_dir: str = FRAME_TEMP_DIR):
    """清理单个视频的临时帧文件。"""
    video_dir = os.path.join(tmp_dir, video_id)
    if os.path.isdir(video_dir):
        shutil.rmtree(video_dir, ignore_errors=True)


def cleanup_all_frames(tmp_dir: str = FRAME_TEMP_DIR):
    """清理所有临时帧文件。"""
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp frames dir: {tmp_dir}")


def prepare_video_frames(
    video_path: str,
    video_id: str,
    max_num_frames: int = MAX_NUM_FRAMES,
    max_fps: float = MAX_FPS,
) -> List[str]:
    """
    一步完成：采样视频帧 → 保存为临时 JPG → 返回路径列表。
    """
    frames = encode_video(video_path, max_num_frames, max_fps)
    if not frames:
        logger.warning(f"No frames extracted from video: {video_path}")
        return []
    paths = save_frames_as_jpg(frames, video_id)
    logger.info(f"Prepared {len(paths)} frames for video {video_id}")
    return paths
