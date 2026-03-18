# Daily-Omni CPP 评测 Python 端实现计划

## 复用评估

| 组件 | 复用程度 | 说明 |
|------|---------|------|
| `eval_cpp_server_manager.py` | **100%** | 完全复用，仅改 import 源 |
| `eval_cpp_video_prep.py` | **60%** | 复用解码后端（decord/ffmpeg/torchvision）和帧保存逻辑；**采样算法需重写**（Daily-Omni 分长短视频两种策略，且需返回时间戳供音频切分） |
| `eval_cpp_http_client.py` | **70%** | 复用 `_post_json`、`reset`、`decode`、SSE 收集；**新增** `prefill_audio`、`prefill_interleaved`（交错 prefill 图片+音频） |
| `eval_cpp_config.py` | **30%** | Server 相关配置相同，但数据集路径、prompt 模板、评测参数不同 |
| `eval_cpp_pipeline.py` | **30%** | 复用 server 启动/停止、线程池并发框架；数据集加载（JSONL vs parquet）、处理流程（需音频交错）、结果格式均不同 |
| `eval_your_result.py` | **20%** | 答案提取逻辑可复用；分组维度完全不同（qa_type/video_category/video_duration 等） |
| `rerun_failed.py` | **40%** | 框架可复用；数据源和处理逻辑需改为 JSONL + 音频交错 |

## 新增模块

| 模块 | 说明 |
|------|------|
| `eval_cpp_audio_prep.py` | **全新**。音频加载（librosa 重采样到 16kHz）、按帧时间戳切分音频段、保存 WAV 文件、变速/截断（OOM 重试用） |

## Daily-Omni vs VideoMME 关键差异

| 维度 | VideoMME | Daily-Omni |
|------|----------|------------|
| 数据格式 | Parquet | JSONL |
| 每视频题数 | 3 题/视频 | 1 题/视频 |
| 音频 | 无 | 有独立 WAV，需交错 prefill |
| 帧采样算法 | `range(0, total, round(fps))` | 长视频：0.1s 粒度 → uniform_sample；短视频：每秒1帧 |
| 需要时间戳 | 否 | 是（用于切分音频段） |
| Prompt 模板 | 无 `{media}` 占位符 | 有 `{media}` 占位符（替换后移除） |
| 分组评测维度 | domain/sub_category/task_type/duration | qa_type/content_parent_category/video_category/video_duration |
| 解码策略 | greedy + repeat_penalty=1.02 | greedy（无 repeat_penalty） |
| max_tokens | 100 | 128 |

## 文件结构

```
daily-omni/
├── eval_cpp_config.py          # 配置
├── eval_cpp_video_prep.py      # 视频帧采样（Daily-Omni 算法）
├── eval_cpp_audio_prep.py      # 音频加载/切分/保存（全新）
├── eval_cpp_http_client.py     # HTTP 客户端（含音频 prefill）
├── eval_cpp_pipeline.py        # 主流水线
├── eval_cpp_server_manager.py  # Server 管理（复用 videomme）
├── eval_daily_omni_result.py   # Daily-Omni 评测评分
├── rerun_failed.py             # 重跑失败题目
├── CPP_CHANGES.md              # C++ 端需要的改动记录
├── README.md                   # 使用文档
├── .env                        # 本机路径配置
├── output/                     # 评测结果
└── log/                        # Server 日志
```

## 推理链路（单样本）

```
1. reset KV cache
2. 交错 prefill（对齐 Python "\n".join 语义）:
   cnt=0:  prefill_image(frame_0, prompt="\n")  [skip_system_prompt=True]
   cnt=1:  prefill_audio(audio_seg_0, prompt="\n")
   cnt=2:  prefill_image(frame_1, prompt="\n")
   cnt=3:  prefill_audio(audio_seg_1, prompt="\n")
   ...
   cnt=2N-2: prefill_image(frame_{N-1}, prompt="\n")
   cnt=2N-1: prefill_audio(audio_seg_{N-1}, prompt="\n")
3. prefill_text(question_prompt, cnt=2N)
4. decode → 提取答案字母
```

## 依赖

```
# 在 videomme 基础上额外需要:
pip install librosa soundfile
```
