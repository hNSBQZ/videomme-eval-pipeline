# Video-MME CPP Evaluation Pipeline

基于 llama.cpp-omni 的 Video-MME 视频理解评测流水线，支持多卡并行推理、自动重跑失败题目、自动评分。

## 关于 llama.cpp-omni 的修改

本仓库包含了一份修改后的 [llama.cpp-omni](https://github.com/anthropic-ai/llama.cpp-omni) 源码（`llama.cpp-omni/` 目录）。为了支持 Video-MME 视频评测流程，对以下文件做了修改：

- `tools/omni/omni.cpp` — 扩展 omni 推理接口，支持多帧视频 prefill
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
pip install pandas pyarrow requests python-dotenv decord Pillow
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
LLM_MODEL_PATH=/path/to/gguf-model-dir/MiniCPM-o-4_5-Q4_K_M.gguf

# Video-MME 数据集
PARQUET_PATH=/path/to/videomme/test-00000-of-00001.parquet
VIDEO_DATA_DIR=/path/to/videomme/data
```

其他参数（GPU 数量、端口、ctx_size 等）可通过环境变量或命令行参数覆盖，详见 `eval_cpp_config.py`。

## 使用

### 完整流水线（推理 + 重跑 + 评分）

```bash
python eval_cpp_pipeline.py --num-gpus 8 --base-port 9080
```

流水线会依次执行：
1. 加载数据集，按视频分组并分配到各 GPU
2. 启动 N 个 llama-server，多线程并发推理
3. 保存结果到 `output/output_videomme_cpp.json`
4. 自动扫描不合法的 response 并用单卡重跑
5. 调用官方评测脚本输出各维度准确率

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-gpus` | 使用 GPU 数量 | 8 |
| `--base-port` | Server 起始端口 | 8080 |
| `--limit N` | 只取前 N 条数据（测试用） | 0（全量） |
| `--skip-rerun` | 跳过重跑失败题目 | false |
| `--skip-scoring` | 跳过评分 | false |
| `--rerun-gpu` | 重跑使用的 GPU | 0 |
| `--rerun-port` | 重跑 server 端口 | 9080 |

### 快速测试（1 卡 + 6 条数据）

```bash
python eval_cpp_pipeline.py --num-gpus 1 --base-port 9080 --limit 6
```

### 单独重跑失败题目

```bash
python rerun_failed.py --gpu 0 --port 9080
```

### 单独评分

```bash
python eval_your_result.py \
  --results_file output/output_videomme_cpp.json \
  --video_duration_type "short,medium,long" \
  --return_categories_accuracy \
  --return_sub_categories_accuracy \
  --return_task_types_accuracy
```

## 文件结构

```
cpp-eval/
├── llama.cpp-omni/            # 修改后的 llama.cpp-omni 源码（含视频评测接口改动）
├── eval_cpp_pipeline.py       # 主流水线
├── eval_cpp_config.py         # 配置（路径、参数）
├── eval_cpp_server_manager.py # llama-server 生命周期管理
├── eval_cpp_http_client.py    # HTTP 客户端（omni streaming API）
├── eval_cpp_video_prep.py     # 视频帧采样
├── rerun_failed.py            # 重跑失败题目（独立可用）
├── eval_your_result.py        # 官方评测脚本
├── clean_response.py          # 清洗已有 output 中的 response
├── .env                       # 本机路径配置（不入库）
├── output/                    # 评测结果输出
└── log/                       # llama-server 日志
```
