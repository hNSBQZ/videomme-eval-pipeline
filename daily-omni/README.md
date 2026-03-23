# Daily-Omni CPP Evaluation Pipeline

基于 llama.cpp-omni 的 Daily-Omni 多模态（视频 + 音频）评测流水线，支持多卡并行推理、自动重跑失败题目、自动评分。

## 关于 llama.cpp-omni 的修改

本仓库包含了一份修改后的 [llama.cpp-omni](https://github.com/anthropic-ai/llama.cpp-omni) 源码（`llama.cpp-omni/` 目录）。为了支持 Daily-Omni 视频+音频评测流程，对以下文件做了修改：

- `tools/omni/omni.cpp` — 扩展 omni 推理接口，支持多帧视频 + 音频交错 prefill
- `tools/omni/omni.h` — 对应的头文件声明
- `tools/server/server.cpp` — 调整 server 端 streaming API 路由

编译方式与原始 llama.cpp-omni 一致，请参考其文档进行 CMake 构建：

```bash
cd llama.cpp-omni
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j
```

编译产物位于 `llama.cpp-omni/build/bin/`。

## 环境准备

```bash
pip install pandas pyarrow requests python-dotenv decord Pillow soundfile
```

## 配置

复制 `.env.example` 创建 `.env`，填入本机路径：

```bash
# CUDA 动态库路径（编译 llama-server 时使用的 CUDA 版本）
EXTRA_LD_LIBRARY_PATH=/path/to/cuda/lib

# llama-server 可执行文件路径
LLAMA_SERVER_BIN=/path/to/llama.cpp-omni/build/bin/llama-server

# GGUF 模型目录（包含 vision、audio 等子目录）
GGUF_MODEL_DIR=/path/to/gguf-model-dir

# LLM 主模型文件
LLM_MODEL_PATH=/path/to/gguf-model-dir/MiniCPM-o-4_5-F16.gguf

# Daily-Omni 数据集
DATASET_DIR=~/daily-omni
ANNOTATION_PATH=~/daily-omni/daily_omni.jsonl
```

其他参数（GPU 数量、端口、ctx_size 等）可通过环境变量或命令行参数覆盖，详见 `eval_cpp_config.py`。

## 使用

### 完整流水线（推理 + 重跑 + 评分）

```bash
python eval_cpp_pipeline.py --num-gpus 8 --base-port 9080
```

### 推荐启动（保存 Python 日志到文件）

```bash
LOG="log/daily_omni_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "python -u eval_cpp_pipeline.py --num-gpus 8 --base-port 9080 2>&1 | tee \"$LOG\"" >/dev/null 2>&1 &
echo "log: $LOG"
```

### 后 4 张卡跑（假设共 8 卡，使用物理卡 4,5,6,7）

```bash
LOG="log/daily_omni_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash -c "python -u eval_cpp_pipeline.py --num-gpus 4 --base-port 9080 2>&1 | tee \"$LOG\"" >/dev/null 2>&1 &
echo "log: $LOG"
```

- `2>&1 | tee ...` 会把 Python 的 stdout/stderr 同时输出到屏幕和日志文件
- 按 `Ctrl+C` 会触发优雅中断：停止 worker 线程处理并回收 llama-server 进程
- llama-server 日志写入 `log/server_gpu{gpu_id}.log`，并自动轮转历史日志（默认保留最近 5 份）

流水线会依次执行：
1. 加载 JSONL 数据集（1197 条 MCQ）
2. 启动 N 个 llama-server，多线程并发推理
3. 对每个样本：视频帧采样 + 音频切分 → 交错 prefill → decode
4. 保存结果到 `output/output_daily_omni_cpp.json`
5. 自动扫描不合法的 response 并用单卡重跑
6. 按 qa_type、content_parent_category、video_category 等维度评分

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-gpus` | 使用 GPU 数量 | 8 |
| `--base-port` | Server 起始端口 | 8080 |
| `--limit N` | 只取前 N 条数据（测试用） | 0（全量） |
| `--annotation` | JSONL 标注文件路径 | `~/daily-omni/daily_omni.jsonl` |
| `--data-dir` | 数据集根目录 | `~/daily-omni` |
| `--output` | 输出 JSON 路径 | `output/output_daily_omni_cpp.json` |
| `--skip-rerun` | 跳过重跑失败题目 | false |
| `--skip-scoring` | 跳过评分 | false |
| `--rerun-gpu` | 重跑使用的 GPU | 0 |
| `--rerun-port` | 重跑 server 端口 | 9080 |
| `--log-level` | 日志级别 | INFO |

### 快速测试（1 卡 + 6 条数据）

```bash
python eval_cpp_pipeline.py --num-gpus 1 --base-port 9080 --limit 6
```

如需全链路记录 HTTP 请求内容（URL + 完整 JSON payload），可使用 DEBUG 日志并落盘：

```bash
LOG="log/daily_omni_$(date +%Y%m%d_%H%M%S).log"
python -u eval_cpp_pipeline.py --num-gpus 1 --base-port 9080 --limit 6 --log-level DEBUG 2>&1 | tee "$LOG"
echo "log: $LOG"
```

### 单独重跑失败题目

```bash
python rerun_failed.py --gpu 0 --port 9080
```

### 单独评分

```bash
python eval_daily_omni_result.py --results-file output/output_daily_omni_cpp.json
```

## 与 Video-MME 的区别

| 特性 | Video-MME | Daily-Omni |
|------|-----------|------------|
| 数据集格式 | Parquet | JSONL |
| 模态 | 纯视频帧 | 视频帧 + 音频 |
| Prefill 方式 | 逐帧 prefill | 帧-音频交错 prefill |
| 样本数 | 900 视频 × 3 题 | 1197 MCQ |
| Prompt 模板 | 无 leading `\n` | 带 leading `\n` + 尾部 "Please select..." |
| 评分脚本 | `eval_your_result.py` | `eval_daily_omni_result.py` |

## 文件结构

```
daily-omni/
├── eval_cpp_pipeline.py         # 主流水线
├── eval_cpp_config.py           # 配置（路径、参数）
├── eval_cpp_server_manager.py   # llama-server 生命周期管理
├── eval_cpp_http_client.py      # HTTP 客户端（omni streaming API）
├── eval_cpp_video_prep.py       # 视频帧采样
├── eval_cpp_audio_prep.py       # 音频切分（按帧时间戳）
├── rerun_failed.py              # 重跑失败题目（独立可用）
├── eval_daily_omni_result.py    # 评分脚本
├── .env                         # 本机路径配置（不入库）
├── output/                      # 评测结果输出
├── log/                         # llama-server 日志 + Pipeline 日志
├── result/                      # 历史评测结果备份
└── tmp_media/                   # 临时帧 JPG + 音频 WAV 片段（自动清理）
```
