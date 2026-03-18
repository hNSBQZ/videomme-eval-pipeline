# Daily-Omni CPP Evaluation Pipeline

基于 llama.cpp-omni 的 Daily-Omni 音视频理解评测流水线，支持多卡并行推理、音视频交错 prefill、自动重跑失败题目、自动评分。

## 架构概述

复用 VideoMME 评测的 server 管理和 HTTP 通信框架，新增音频处理模块，适配 Daily-Omni 数据集格式。

```
Python 端                          C++ 端 (llama-server)
┌─────────────────────┐           ┌─────────────────────┐
│ 视频帧采样 (decord)  │           │                     │
│ 音频加载 (librosa)   │───HTTP──→│ omni_init            │
│ 音频切分             │           │ reset                │
│ 交错 prefill 调用    │           │ prefill (image/audio)│
│ 结果收集与评分       │←──SSE───│ decode (stream)      │
└─────────────────────┘           └─────────────────────┘
```

## 环境准备

```bash
# 在 VideoMME 依赖基础上额外安装
pip install librosa soundfile

# 完整依赖
pip install pandas requests python-dotenv decord Pillow librosa soundfile
```

## 配置

复制 `.env.example` 或直接编辑 `.env`：

```bash
EXTRA_LD_LIBRARY_PATH=/path/to/cuda/lib
LLAMA_SERVER_BIN=/path/to/llama.cpp-omni/build/bin/llama-server
GGUF_MODEL_DIR=/path/to/gguf-model-dir
LLM_MODEL_PATH=/path/to/gguf-model-dir/MiniCPM-o-4_5-Q4_K_M.gguf
DATASET_DIR=/cache/hanqingzhe/daily-omni
ANNOTATION_PATH=/cache/hanqingzhe/daily-omni/daily_omni.jsonl
CTX_SIZE=40960
```

## 使用

### 完整流水线

```bash
python eval_cpp_pipeline.py --num-gpus 8 --base-port 9080
```

### 推荐启动（保存日志）

```bash
LOG="log/daily_omni_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "python -u eval_cpp_pipeline.py --num-gpus 4 --base-port 9080 2>&1 | tee \"$LOG\"" >/dev/null 2>&1 &
echo "log: $LOG"
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-gpus` | 使用 GPU 数量 | 8 |
| `--base-port` | Server 起始端口 | 8080 |
| `--annotation` | JSONL 标注文件路径 | 配置文件中的默认值 |
| `--data-dir` | 数据集根目录 | 配置文件中的默认值 |
| `--limit N` | 只取前 N 条（测试用） | 0（全量） |
| `--skip-rerun` | 跳过重跑失败题目 | false |
| `--skip-scoring` | 跳过评分 | false |
| `--rerun-gpu` | 重跑使用的 GPU | 0 |
| `--rerun-port` | 重跑 server 端口 | 9080 |

### 快速测试

```bash
python eval_cpp_pipeline.py --num-gpus 1 --base-port 9080 --limit 5
```

### 单独重跑失败题目

```bash
python rerun_failed.py --gpu 0 --port 9080
```

### 单独评分

```bash
python eval_daily_omni_result.py --results-file output/output_daily_omni_cpp.json
```

## 数据集

Daily-Omni 数据集位于 `/cache/hanqingzhe/daily-omni/`：

```
daily-omni/
├── daily_omni.jsonl          # 1197 条 MCQ 标注
├── qa.json                   # 原始标注
└── Videos/
    └── {video_id}/
        ├── {video_id}_video.mp4
        └── {video_id}_audio.wav
```

## 评测参数（对齐 evalkit）

| 参数 | 值 | 说明 |
|------|-----|------|
| max_frames | 64 | 最大采样帧数 |
| max_fps | 1.0 | 采样频率上限 |
| audio_sr | 16000 | 音频采样率 |
| max_tokens | 128 | 最大生成 token 数 |
| sampling | greedy | 贪心解码 |
| max_slice_nums | 1 | 不分块 |
| interleave | true | 音视频交错模式 |

## 文件结构

```
daily-omni/
├── eval_cpp_pipeline.py       # 主流水线
├── eval_cpp_config.py         # 配置
├── eval_cpp_server_manager.py # Server 生命周期管理
├── eval_cpp_http_client.py    # HTTP 客户端（含音频 prefill）
├── eval_cpp_video_prep.py     # 视频帧采样（Daily-Omni 算法）
├── eval_cpp_audio_prep.py     # 音频加载/切分/保存
├── eval_daily_omni_result.py  # 评分脚本
├── rerun_failed.py            # 重跑失败题目
├── CPP_CHANGES.md             # C++ 端需确认的改动
├── PLAN.md                    # 实现计划
├── .env                       # 本机路径配置
├── output/                    # 评测结果
└── log/                       # Server 日志
```

## C++ 端注意事项

详见 [CPP_CHANGES.md](CPP_CHANGES.md)，主要需要确认：
1. 音频 prefill 是否支持从 WAV 文件路径加载
2. 交错 prefill（image/audio 交替）是否正确处理
3. thinking 模式的空 think 块注入
