"""
Daily-Omni CPP 评测 Pipeline 主控脚本。

功能：
  1. 加载 JSONL 数据集（1197 条 MCQ）
  2. 均匀分配到 N 个 GPU worker
  3. 启动 N 个 llama-server 进程（每卡一个）
  4. 调用 omni_init 初始化
  5. N 线程并发处理：视频帧采样 + 音频切分 + 交错 prefill + decode
  6. 收集结果，输出 JSON
  7. 可选：重跑失败题目
  8. 可选：调用 Daily-Omni 评分脚本

用法：
  python eval_cpp_pipeline.py [--num-gpus 8] [--output output.json]
"""
import os
import sys
import json
import re
import time
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from eval_cpp_config import (
    DATASET_DIR, ANNOTATION_PATH, OUTPUT_DIR, OUTPUT_JSON,
    NUM_GPUS, BASE_PORT, MAX_TOKENS, MEDIA_TYPE, USE_TTS,
    USER_PROMPT_TEMPLATE, MAX_NUM_FRAMES,
)
from eval_cpp_server_manager import start_all_servers, stop_all_servers
from eval_cpp_http_client import OmniServerClient
from eval_cpp_video_prep import prepare_video_frames, cleanup_sample_media, cleanup_all_media
from eval_cpp_audio_prep import prepare_audio_segments

logger = logging.getLogger(__name__)


# ==================== 数据集加载 ====================

def load_dataset(annotation_path: str = ANNOTATION_PATH, limit: int = 0) -> List[Dict[str, Any]]:
    """加载 Daily-Omni JSONL 数据集。"""
    logger.info(f"Loading dataset from {annotation_path}")
    samples = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if limit > 0:
        samples = samples[:limit]
        logger.info(f"Limited to first {limit} samples")
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def build_paths(sample: Dict[str, Any], data_dir: str = DATASET_DIR) -> Dict[str, str]:
    """从 JSONL 记录构建视频和音频的完整路径。"""
    return {
        "video_path": os.path.join(data_dir, sample["VideoPath"]),
        "audio_path": os.path.join(data_dir, sample["WavPath"]),
    }


def split_into_chunks(samples: List[Dict], n: int) -> List[List[Dict]]:
    """
    将样本分成 n 份，同时保证同一 video_id 不会跨 chunk。

    背景：临时媒体目录按 sample(video_id) 命名。
    若同一 video_id 跨 GPU 并发处理，会出现互相清理临时目录的问题。
    """
    chunks = [[] for _ in range(n)]
    chunk_sizes = [0 for _ in range(n)]

    # 先按 video_id 分组，保持组内样本顺序不变
    groups: Dict[str, List[Dict]] = {}
    for idx, sample in enumerate(samples):
        # video_id 缺失时退化为“单样本单组”，避免意外把未知样本绑定到同一组
        video_id = sample.get("video_id") or f"__missing_video_id_{idx}"
        groups.setdefault(video_id, []).append(sample)

    # 组级别分配：每次把当前组分给“样本数最少”的 chunk，尽量均衡负载
    for group_samples in groups.values():
        target = min(range(n), key=lambda i: chunk_sizes[i])
        chunks[target].extend(group_samples)
        chunk_sizes[target] += len(group_samples)

    duplicate_video_groups = sum(1 for g in groups.values() if len(g) > 1)
    logger.info(
        f"Split with video_id pinning: total_groups={len(groups)}, "
        f"duplicate_video_groups={duplicate_video_groups}"
    )
    for i, c in enumerate(chunks):
        logger.info(f"  Chunk {i}: {len(c)} samples")
    return chunks


# ==================== Prompt 构建 ====================

def build_prompt(question: str, choices: list) -> str:
    """
    构建评测文本 prompt。

    对齐 evalkit _build_options_prompt：逐项添加 "A. " 前缀 + 尾部换行，
    再 .rstrip() 去掉末尾空白。实际数据 choices 已含 "A. xxx" 前缀，
    Python 端会产生 "A. A. xxx" 双前缀，此处严格对齐该行为。
    """
    KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    options_prompt = ""
    for key, choice in zip(KEYS[:len(choices)], choices):
        options_prompt += f"{key}. {choice}\n"
    options_prompt = options_prompt.rstrip()
    return USER_PROMPT_TEMPLATE.format(question=question, options=options_prompt)


# ==================== 答案提取 ====================

def extract_answer(response_text: str) -> str:
    """
    从模型输出中提取 A/B/C/D 答案字母。

    策略：
      1. strip 后处理 <|tts_eos|> 标记
      2. 整体只有一个字母 A-D → 直接返回
      3. 匹配独立出现的 A/B/C/D
      4. 都找不到 → 返回空字符串
    """
    text = response_text.replace("<|tts_eos|>", "").strip()
    if not text:
        return ""

    cleaned = text.rstrip(".").strip()
    if len(cleaned) == 1 and cleaned.upper() in "ABCD":
        return cleaned.upper()

    match = re.search(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', text)
    if match:
        return match.group(1)
    return ""


# ==================== 单样本处理 ====================

def process_sample(
    client: OmniServerClient,
    sample: Dict[str, Any],
    data_dir: str = DATASET_DIR,
) -> Dict[str, Any]:
    """
    处理单个 Daily-Omni 样本。

    流程：
      1. 采样视频帧并保存 JPG
      2. 加载音频并按时间戳切分保存 WAV
      3. reset KV cache
      4. 交错 prefill (frame, audio_seg, frame, audio_seg, ...)
      5. prefill 文本 prompt
      6. decode 获取模型回答
      7. 提取答案字母
      8. 清理临时文件
    """
    paths = build_paths(sample, data_dir)
    video_path = paths["video_path"]
    audio_path = paths["audio_path"]
    sample_id = sample.get("video_id", "unknown")

    result = {
        "video_id": sample_id,
        "question": sample["question"],
        "choices": sample["choices"],
        "gt_answer": sample["gt_answer"],
        "prediction": "",
        "raw_response": "",
        "audio_speed": 1.0,
        "audio_trim_end": 0.0,
    }
    for key in ["qa_type", "content_parent_category", "content_fine_category",
                "video_category", "video_duration"]:
        if key in sample:
            result[key] = sample[key]

    if not os.path.isfile(video_path):
        logger.error(f"Video not found: {video_path}")
        result["prediction"] = "[ERROR] video file not found"
        return result

    try:
        # 1. 视频帧采样
        frame_paths, timestamps = prepare_video_frames(video_path, sample_id)
        if not frame_paths:
            result["prediction"] = "[ERROR] no frames extracted"
            return result

        # 2. 音频切分
        audio_seg_paths = []
        if os.path.isfile(audio_path):
            audio_seg_paths = prepare_audio_segments(
                audio_path, timestamps, sample_id,
            )
        else:
            logger.warning(f"Audio not found: {audio_path}, proceeding without audio")

        # 3. Reset
        client.reset()

        # 4. 交错 prefill
        total_cnt = client.prefill_interleaved(
            frame_paths=frame_paths,
            audio_paths=audio_seg_paths,
            skip_system_prompt=True,
        )

        # 5. Prefill 文本
        prompt = build_prompt(sample["question"], sample["choices"])
        client.prefill_text(prompt, cnt=total_cnt)

        # 6. Decode
        raw_response = client.decode(round_idx=0)
        result["raw_response"] = raw_response

        # 7. 提取答案
        result["prediction"] = extract_answer(raw_response)

    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {e}")
        result["raw_response"] = f"[ERROR] {e}"
        result["prediction"] = ""
    finally:
        cleanup_sample_media(sample_id)

    return result


# ==================== Worker 线程 ====================

def process_chunk(
    gpu_id: int,
    port: int,
    chunk: List[Dict[str, Any]],
    stop_event: threading.Event,
    data_dir: str = DATASET_DIR,
) -> List[Dict[str, Any]]:
    """单个 GPU worker：串行处理分配到的所有样本。"""
    client = OmniServerClient(f"http://127.0.0.1:{port}")
    all_results = []
    total = len(chunk)

    for i, sample in enumerate(chunk):
        if stop_event.is_set():
            logger.info(f"[GPU {gpu_id}] Stop requested, break chunk loop")
            break
        sid = sample.get("video_id", "?")
        logger.info(f"[GPU {gpu_id}] ({i+1}/{total}) Processing sample {sid}")
        t0 = time.time()
        result = process_sample(client, sample, data_dir=data_dir)
        elapsed = time.time() - t0
        all_results.append(result)

        pred = result.get("prediction", "")
        gt = result.get("gt_answer", "")
        resp_short = repr(result.get("raw_response", ""))[:80]
        logger.info(
            f"[GPU {gpu_id}] ({i+1}/{total}) {sid} done in {elapsed:.1f}s, "
            f"GT={gt} Pred={pred} Resp={resp_short}"
        )

    client.close()
    return all_results


# ==================== 结果输出 ====================

def format_output(
    results: List[Dict[str, Any]],
    dataset_name: str = "daily_omni",
) -> Dict[str, Any]:
    """
    格式化输出（对齐 evalkit 的推理输出格式）。
    """
    predictions = []
    for r in results:
        pred_entry = {
            "prediction": r.get("prediction", ""),
            "annotation": {
                "question": r.get("question", ""),
                "choices": r.get("choices", []),
                "gt_answer": r.get("gt_answer", ""),
                "video_id": r.get("video_id", ""),
            },
            "audio_speed": r.get("audio_speed", 1.0),
            "audio_trim_end": r.get("audio_trim_end", 0.0),
        }
        for key in ["qa_type", "content_parent_category", "content_fine_category",
                    "video_category", "video_duration"]:
            if key in r:
                pred_entry["annotation"][key] = r[key]
        predictions.append(pred_entry)

    return {
        "predictions": predictions,
        "dataset_name": dataset_name,
    }


def save_results(output: Dict, path: str = OUTPUT_JSON):
    """保存结果 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to {path}")


# ==================== Main ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Daily-Omni CPP Evaluation Pipeline")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS, help="Number of GPUs to use")
    parser.add_argument("--base-port", type=int, default=BASE_PORT, help="Base port for servers")
    parser.add_argument("--annotation", type=str, default=ANNOTATION_PATH, help="Path to JSONL annotation file")
    parser.add_argument("--data-dir", type=str, default=DATASET_DIR, help="Dataset root directory")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Only load first N samples (0 = all)")
    parser.add_argument("--skip-rerun", action="store_true", help="Skip rerun of failed questions")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring after evaluation")
    parser.add_argument("--rerun-gpu", type=int, default=0, help="GPU id for rerun server")
    parser.add_argument("--rerun-port", type=int, default=9080, help="Port for rerun server")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    stop_event = threading.Event()
    interrupted = False
    servers = []

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Daily-Omni CPP Evaluation Pipeline")
    logger.info(f"  GPUs: {args.num_gpus}")
    logger.info(f"  Base port: {args.base_port}")
    logger.info(f"  Annotation: {args.annotation}")
    logger.info(f"  Data dir: {args.data_dir}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    # 1. 加载数据集
    samples = load_dataset(args.annotation, limit=args.limit)
    chunks = split_into_chunks(samples, args.num_gpus)

    try:
        # 2. 启动 servers
        logger.info("Starting llama-servers...")
        servers = start_all_servers(args.num_gpus, args.base_port)

        # 3. 初始化 omni 上下文
        logger.info("Initializing omni contexts...")
        for srv in servers:
            if stop_event.is_set():
                break
            client = OmniServerClient(srv.base_url)
            client.omni_init(media_type=MEDIA_TYPE, use_tts=USE_TTS, n_predict=MAX_TOKENS)
            client.close()

        # 4. 并发处理
        logger.info("Starting evaluation...")
        t_start = time.time()
        all_results = []

        pool = ThreadPoolExecutor(max_workers=args.num_gpus)
        futures = {}
        try:
            for gpu_id, chunk in enumerate(chunks):
                port = args.base_port + gpu_id
                fut = pool.submit(
                    process_chunk, gpu_id, port, chunk, stop_event,
                    data_dir=args.data_dir,
                )
                futures[fut] = gpu_id

            for fut in as_completed(futures):
                gpu_id = futures[fut]
                try:
                    results = fut.result()
                    all_results.extend(results)
                    logger.info(f"GPU {gpu_id} completed: {len(results)} samples")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} failed: {e}")
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            logger.warning("KeyboardInterrupt received, stopping workers and servers...")
            for fut in futures:
                fut.cancel()
            stop_all_servers(servers)
            servers = []
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        elapsed = time.time() - t_start
        logger.info(f"Evaluation done: {len(all_results)} samples in {elapsed:.1f}s")

        # 5. 格式化并保存结果
        output = format_output(all_results)
        save_results(output, args.output)

        # 6. 简单统计
        correct = sum(1 for r in all_results if r.get("prediction", "") == r.get("gt_answer", ""))
        total = len(all_results)
        if total > 0:
            logger.info(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        else:
            logger.info("No results")

    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        logger.warning("Interrupted by user (Ctrl+C).")
    finally:
        logger.info("Stopping servers...")
        stop_all_servers(servers)
        cleanup_all_media()

    if interrupted:
        logger.warning("Pipeline interrupted, skip rerun and scoring.")
        return

    # 7. 重跑失败题目
    if not args.skip_rerun:
        from rerun_failed import find_failed, rerun_failed_samples, patch_output
        from eval_cpp_server_manager import start_server, wait_server_ready, stop_server
        failed_indices = find_failed(args.output)
        if failed_indices:
            logger.info(f"Rerunning {len(failed_indices)} failed samples...")
            server = start_server(args.rerun_gpu, args.rerun_port)
            if wait_server_ready(server):
                try:
                    client = OmniServerClient(f"http://127.0.0.1:{args.rerun_port}")
                    client.omni_init(media_type=MEDIA_TYPE, use_tts=USE_TTS, n_predict=MAX_TOKENS)
                    rerun_results = rerun_failed_samples(
                        client, args.output, failed_indices,
                        data_dir=args.data_dir,
                    )
                    client.close()
                finally:
                    stop_server(server)
                patch_output(args.output, rerun_results)
            else:
                stop_server(server)
                logger.error("Rerun server failed to start, skipping.")
        else:
            logger.info("All predictions valid, no rerun needed.")

    # 8. 评分
    if not args.skip_scoring:
        from eval_daily_omni_result import eval_daily_omni_results
        logger.info("Running Daily-Omni scoring...")
        eval_daily_omni_results(args.output)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
