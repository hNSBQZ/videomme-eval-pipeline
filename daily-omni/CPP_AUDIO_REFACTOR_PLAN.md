# C++ 推理端音频处理重构方案（v3）

> 日期: 2026-03-25
> 基于: AUDIO_PIPELINE_DEEP_DIVE.md（含 Transformer 层等价性证明）
> 目标: 让 C++ 推理端的 Whisper 音频编码行为与 Python 端 (o45-py) **完全对齐**，消除 mel 归一化差异和 conv 边界差异。

---

## 〇、核查结论摘要（已确认事实）

### 0.1 已确认对齐的项（不需要改动）

| # | 项目 | 结论 |
|---|------|------|
| 1 | 采样参数 | ✅ 两端一致: temp=0.7, top_p=0.8, top_k=100, repeat_penalty=1.02 |
| 2 | Assistant prompt 格式 | ✅ `<think>\n\n</think>\n\n<\|tts_bos\|>` 两端一致 |
| 3 | Prompt 选项格式 | ✅ `f"{key}. {choice}\n"` + `.rstrip()` 两端一致 |
| 4 | 换行符处理 | ✅ 已修复媒体→文本边界双 `\n` 问题 |
| 5 | Decode 路径 | ✅ `stream_decode`，`use_tts=false` 时 TTS 完全旁路 |
| 6 | 异步 Prefill | ✅ FIFO 保序严格，decode 正确等待所有 prefill 完成 |
| 7 | Whisper 跨样本 KV cache | ✅ `reset()` 清零，无跨样本污染 |
| **8** | **Transformer 层 Attention 模式** | **✅ 已证明等价：Python chunk mask ≡ C++ KV cache streaming（详见 0.4）** |

### 0.2 已确认的关键数值

| 参数 | 值 | 来源 |
|------|-----|------|
| 1s 音频 → 最终 LLM token 数 | **10 tokens** | mel 100 → conv2(stride=2) 50 → AvgPool(k=5,s=5) 10 |
| PE 上限 (n_audio_ctx) | **1500** tokens (conv2 后) | GGUF `encoder.positional_embedding` shape [1500, 1024] |
| PE 上限对应音频时长 | **30 秒** | 30s → mel 3000 → conv2 1500 = PE 上限 |
| chunk_size | **50** tokens (= 1s conv 后) | `audio_chunk_length=1.0`, `int(1.0 * 50) = 50` |
| Daily-Omni 实测 max_mel_frames | **3000** | 1197 条全量验证，不超限 |

### 0.3 需要修复的差异

| 优先级 | 差异 | C++ 当前行为 | Python 参考行为 | 影响程度 |
|--------|------|-------------|----------------|---------|
| **P0** | **Mel 归一化基准** | 逐段独立(~1s)的局部 max 归一化 | 合并后按 ≤30s chunk 的全局 max 归一化 | **最大差异源**：所有音频帧的 mel 数值系统性漂移 |
| **P1** | **Conv 边界效应** | 每段 ~1s 独立过 conv1/conv2，边界处只看到 zero-padding | 整个 ≤30s chunk 的 mel 一起过 conv，边界处看到相邻段的真实 mel 帧 | 每段边界 ~2 帧受影响 |
| P2 | STFT 边界 | 每段首尾独立反射 padding | 合并波形上连续 STFT | 很小 |

### 0.4 已排除的差异：Transformer 层 Attention 等价性（重要）

**结论：Python 的 chunk attention mask 与 C++ 的 KV cache streaming 在 Transformer encoder 层产生完全相同的计算结果。**

等价性证明摘要（详见 AUDIO_PIPELINE_DEEP_DIVE.md §4.2）：

```
单层等价（以 2 个 chunk c0, c1 为例）：

C++ streaming:
  第 1 次 forward c0: Q_0 @ [K_0, V_0] → output_0, 存 K_0,V_0 到 cache
  第 2 次 forward c1: Q_1 @ [K_0_cached, K_1], [V_0_cached, V_1] → output_1

Python chunk mask:
  一次 forward [c0, c1], mask 令 c0 只看自己:
    c0: Q_0 @ [K_0, V_0] (mask 挡住 c1) → output_0
    c1: Q_1 @ [K_0, K_1], [V_0, V_1] (mask 放行) → output_1

K_0, V_0 在两边完全一样 → 单层输出完全一致。
多层归纳：FFN 逐 token、LayerNorm 逐 token，不跨 token 混合信息 → 所有层输出一致。
```

> **v2 文档勘误**：v2 方案声称"Python chunk attention 严格优于 C++ KV cache 流式，因为多层叠加允许信息间接跨 chunk 传播"。这是错误的。chunk mask 在每一层都阻断未来 chunk 的信息流，Transformer 中不存在能绕过 attention mask 的跨 token 操作，所以不存在"通过多层间接传播"的现象。**Attention 模式不是差异源，无需修复。**

### 0.5 差异总结

**全部差异来自 Transformer 之前的预处理步骤**（mel 归一化 + conv 边界），Transformer 层本身的 attention 行为已经等价。这大幅简化了重构方案。

---

## 一、Python 端完整处理链路（对齐基准）

```
Python evalkit 评测路径:
  ① 音频来源：交错模式下，每帧配一段 ~1s 音频
  ② 合并：同消息内所有音频段 np.hstack 拼为一条长波形
  ③ 30s 切分：超过 480000 samples (30s) 的波形按 30s 切 chunk
  ④ Mel 提取：每个 ≤30s chunk 独立算 mel → mel 归一化基于 chunk 内全局 max
  ⑤ Batch pad：所有 chunk pad 到最长，构成 batch tensor [n_chunks, 80, max_frames]
  ⑥ Chunk attention mask：chunk_size=50, num_left_chunks=-1（块级因果）
  ⑦ APM 一次 forward：所有 chunks batch 输入，无 KV cache
  ⑧ Projection + AvgPool1d(k=5, s=5)
  ⑨ 按各段长度拆分 embedding
```

其中步骤 ⑥⑦ 的 chunk attention mask + 一次 forward 已被证明与 C++ 的 KV cache streaming 等价。需要修复的是步骤 ②③④（mel 归一化）和步骤 ④→⑦ 中 conv 接收完整 chunk 输入的行为。

---

## 二、现有 C++ 代码架构

### 2.1 数据流（当前）

```
[Python 前端 (cpp-eval)]
  prepare_audio_segments()         # 音频切 N 段 → audio_seg_000.wav ~ audio_seg_N.wav
  prefill_interleaved()            # 逐个 POST /v1/stream/prefill
    └── for i in range(N):
          POST prefill(img=frame_i)    →  server 做 VPM encode
          POST prefill(audio=seg_i)    →  server 做 Whisper encode  ← 每次独立！
  POST prefill(text=question)
  POST decode()

[C++ Server (llama.cpp-omni)]
  /v1/stream/prefill handler
    → stream_prefill() [omni.cpp]
      → omni_audio_embed_make_with_filename() [omni.cpp]
        → audition_audio_preprocess()  [audition.cpp]   # wav→pcm→mel（单段 ~1s）
        → audition_audio_encode()      [audition.cpp]   # mel→Whisper encoder（单段，KV cache 累积）
      → prefill_with_emb()                              # embedding→LLM KV cache
```

**问题所在**：每段 ~1s 音频独立走完整个 Whisper 流水线（mel 计算 → conv → Transformer）。mel 归一化是局部的，conv 在段边界只看到 zero-padding。

### 2.2 关键代码位置

| 模块 | 文件 | 核心函数/结构 |
|------|------|-------------|
| Python 前端 - HTTP 客户端 | `cpp-eval/daily-omni/eval_cpp_http_client.py` | `prefill_interleaved()`, `prefill_audio()` |
| Python 前端 - 音频准备 | `cpp-eval/daily-omni/eval_cpp_audio_prep.py` | `prepare_audio_segments()`, `segment_audio_by_timestamps()` |
| Python 前端 - Pipeline | `cpp-eval/daily-omni/eval_cpp_pipeline.py` | `process_sample()` |
| C++ Server - HTTP handler | `llama.cpp-omni/tools/server/server.cpp` | `handle_stream_prefill_impl` |
| C++ 推理 - 入口 | `llama.cpp-omni/tools/omni/omni.cpp` | `stream_prefill()`, `omni_audio_embed_make_with_filename()` |
| C++ 推理 - 音频头文件 | `llama.cpp-omni/tools/omni/audition.h` | `audition_audio_preprocess()`, `audition_audio_batch_encode()`, `whisper_kv_cache` |
| C++ 推理 - 音频实现 | `llama.cpp-omni/tools/omni/audition.cpp` | `log_mel_spectrogram()`, `preprocess_audio()`, `audition_audio_batch_encode()` |

---

## 三、重构方案

### 3.1 总体策略

**改 C++ 端对齐 Python 端**，原因：
- Python 端是"参考实现"，准确率以 Python 为基准
- 合并处理在语义上更合理（全局归一化 + conv 跨段连续）

**核心改动**：让 C++ 端支持"先合并所有音频 PCM → 按 30s 切 chunk → 每个 chunk 整体算 mel + 整体过一次 Whisper encoder → 拆回各段 embedding"。

这一个改动同时解决 P0 (mel 归一化) + P1 (conv 边界) + P2 (STFT 边界)。**不需要实现 chunk attention mask**——因为已证明 Transformer 层的 attention 行为等价，重构后可以沿用与 KV cache streaming 等价的处理方式。

### 3.2 新数据流（改动后）

```
[Python 前端 (cpp-eval)]  ← 需要改
  prepare_audio_segments()             # 不变：切分 N 段 → 独立 wav（保留调试用）
  prefill_interleaved_v2()             # 【新增】新的交错 prefill 逻辑
    ├── POST /v1/stream/prefill_audio_batch  ← 【新增 API】一次性送入所有音频路径
    │     body: { audio_paths: [...] }
    │     server 处理:
    │       ① 读取所有 wav → PCM → 拼接为一条长波形     ← 对齐 Python 的 np.hstack
    │       ② 按 30s 切 chunk                          ← 对齐 Python 的 30s 切分
    │       ③ 每个 chunk 独立算 mel（chunk 内全局归一化）  ← 解决 P0 mel
    │       ④ 每个 chunk 整体送入 Whisper encoder         ← 解决 P1 conv 边界
    │           - 关闭 KV cache streaming（单次完整 forward）
    │           - 使用 chunk attention mask 保持与 Python 完全一致
    │       ⑤ 按原始段边界拆分 embedding → 缓存
    │     返回: { n_segments, tokens_per_segment, total_tokens }
    │
    └── for i in range(N):
          POST /v1/stream/prefill(img=frame_i)                          # 不变
          POST /v1/stream/prefill(audio_segment_idx=i)                  # 【改】从缓存取 embedding
  POST prefill(text=question)
  POST decode()
```

### 3.3 关于 Attention 模式的选择

既然 chunk attention mask ≡ KV cache streaming（数学等价），在重构后的整体 forward 中有三种选择：

| 方案 | 与 Python 是否等价 | 实现复杂度 | 说明 |
|------|-------------------|-----------|------|
| **A. Chunk attention mask** | ✅ 完全等价 | 中（需构建 mask） | 与 Python 代码逻辑一一对应，最易验证 |
| B. Full attention（无 mask） | ❌ 不等价 | 低 | 每个位置可见全部上下文，与 Python/streaming 均不同 |
| C. 拆回 1s 做 KV cache streaming | ✅ 等价 | 高（架构割裂） | conv 要整体过但 Transformer 要拆开，不自然 |

**推荐方案 A**：虽然 Transformer 层本身 chunk mask ≡ KV cache streaming，但既然我们已经要整体 forward（为了修 conv 边界），实现 chunk attention mask 是保持等价性的最自然方式。实现简单（一个二重循环构建 mask tensor），且方便与 Python 逐层对比验证。

### 3.4 方案比选（整体架构）

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A. 新增 batch API（推荐）** | 新 API 预处理所有音频，原 prefill API 引用缓存 | 改动最小，向后兼容 | 需要 server 端缓存 embedding |
| B. 前端合并+单次 prefill | 合并成一个大 wav，一次 POST | 前端简单 | 无法交错 img/audio |
| C. 前端算 mel 传给 server | numpy 算好 mel POST 给 server | 最灵活 | 传输量大，侵入性大 |

**选择方案 A**。

---

## 四、详细改动清单

### 4.1 C++ 端改动 (llama.cpp-omni)

#### 4.1.1 `audition.h` — 新增数据结构和接口

```cpp
struct audition_merged_audio_info {
    std::vector<float> merged_pcm;           // 合并后的完整波形
    std::vector<size_t> segment_boundaries;  // 各段在 merged_pcm 中的起始采样点索引
    int sr;                                  // 采样率 (16000)
};

struct audition_audio_chunk_mel {
    audition_audio_f32 mel;                  // 该 chunk 的 mel 特征
    int start_sample;                        // 该 chunk 在 merged_pcm 中的起始采样点
    int end_sample;                          // 结束采样点
};

// 批量音频预处理：合并 → 切 chunk → 计算 mel
bool audition_audio_preprocess_merged(
    struct audition_ctx * ctx,
    const std::vector<std::string> & audio_paths,
    std::vector<audition_audio_chunk_mel> & chunks,
    audition_merged_audio_info & merge_info
);

// 批量 Whisper encoder forward（无 KV cache，带 chunk attention mask）
bool audition_audio_batch_encode_merged(
    struct audition_ctx * ctx,
    int n_threads,
    const std::vector<audition_audio_chunk_mel> & chunks,
    const audition_merged_audio_info & merge_info,
    std::vector<std::vector<float>> & segment_embeddings,
    std::vector<int> & segment_n_tokens
);
```

#### 4.1.2 `audition.cpp` — 实现新接口

**a) `audition_audio_preprocess_merged()`**

```
伪代码:
1. for each audio_path:
     decode_audio_from_buf() → pcm_i
     merge_info.segment_boundaries.push_back(total_samples)
     merge_info.merged_pcm.insert(end, pcm_i)
     total_samples += pcm_i.size()

2. pad merged_pcm 到 100ms 边界

3. 按 30s (480000 samples) 为上限切 chunk:
     for offset = 0; offset < total_samples; offset += 480000:
       chunk_pcm = merged_pcm[offset : min(offset+480000, total_samples)]
       log_mel_spectrogram(chunk_pcm, ..., mel)   ← 复用现有函数
       chunks.push_back({mel, offset, end})
```

关键点：
- `log_mel_spectrogram()` 已实现 `mmax-8dB` 全局归一化 → 传入 ≤30s PCM 后归一化自然变为 chunk 内全局，与 Python 一致 → **解决 P0**
- STFT 在合并波形上连续计算 → **解决 P2**
- **不需要改动 `log_mel_spectrogram()` 函数本身**

**b) `audition_audio_batch_encode_merged()` — 核心改动**

```
伪代码:
1. 关闭 streaming:
     ctx->whisper_streaming_mode = false
     audition_whisper_clear_kv_cache(ctx)

2. for each chunk in chunks:
     a. 将整个 chunk 的 mel 作为一次输入送入 Whisper encoder
        - conv1/conv2 处理完整 chunk → 解决 P1 conv 边界
        - 构建 chunk attention mask (chunk_size=50) → 保持 Transformer 等价
     b. Projection + AvgPool1d(k=5, s=5)

3. 按段边界拆分 encoder 输出 → segment_embeddings

4. 恢复 streaming mode
```

**Chunk Attention Mask 构建**（在 `build_whisper()` 中，非 streaming 模式时启用）：

```c
// chunk_size = 50 (1s after conv2 stride=2)
for (int i = 0; i < seq_len; i++) {
    int chunk_end = ((i / 50) + 1) * 50;
    for (int j = 0; j < seq_len; j++) {
        mask[i][j] = (j < chunk_end && j < actual_len) ? 0.0f : -INFINITY;
    }
}
```

> 注：chunk attention mask 不是在修复差异（Transformer 层已等价），而是在从 KV cache streaming 切换到整体 forward 时**保持已有的等价性**。如果不加 mask（使用 full attention），Transformer 层的行为反而会偏离 Python。

#### 4.1.3 `omni.h` / `omni.cpp` — 新增缓存管理

```cpp
// omni.h
struct omni_audio_batch_cache {
    std::vector<omni_embed*> segment_embeds;
    bool valid = false;
};

bool omni_audio_batch_precompute(
    struct omni_context * ctx_omni,
    const std::vector<std::string> & audio_paths
);

struct omni_embed * omni_audio_batch_get_segment(
    struct omni_context * ctx_omni, int segment_idx
);

void omni_audio_batch_cache_clear(struct omni_context * ctx_omni);
```

`omni_context` 中新增字段：
```cpp
struct omni_context {
    // ... existing fields ...
    omni_audio_batch_cache audio_batch_cache;  // 【新增】
};
```

实现逻辑：
```
omni_audio_batch_precompute():
  1. audition_audio_preprocess_merged() → 合并 + mel
  2. audition_audio_batch_encode_merged() → 整体 forward + 拆分
  3. 每段 embedding 封装为 omni_embed → 存入 cache
  4. cache.valid = true

omni_audio_batch_get_segment(i):
  return cache.segment_embeds[i]

omni_audio_batch_cache_clear():
  for embed in cache: omni_embed_free(embed)
  cache.clear(); cache.valid = false
```

#### 4.1.4 `server.cpp` — 新增 HTTP API

```
新增路由: POST /v1/stream/prefill_audio_batch

请求:
{
    "audio_paths": ["/path/to/seg_000.wav", "/path/to/seg_001.wav", ...]
}

响应:
{
    "status": "ok",
    "n_segments": 30,
    "tokens_per_segment": [10, 10, 10, ...],
    "total_tokens": 300
}

处理:
  lock(ctx_server.octx_mutex)
  omni_audio_batch_precompute(ctx_server.octx, audio_paths)
```

#### 4.1.5 `stream_prefill()` — 修改现有逻辑

增加 `audio_segment_idx` 参数（默认 -1 走原有逻辑）：

```
修改签名:
  bool stream_prefill(..., int audio_segment_idx = -1);

audio prefill 逻辑:
  if (audio_segment_idx >= 0 && ctx_omni->audio_batch_cache.valid) {
      embeds = omni_audio_batch_get_segment(ctx_omni, audio_segment_idx);
  } else if (aud_fname is not empty) {
      embeds = omni_audio_embed_make_with_filename(...);  // 原有逻辑（向后兼容）
  }
```

同步修改 `handle_stream_prefill_impl` 解析新参数。

#### 4.1.6 `handle_stream_reset_impl` — 增加缓存清理

在 reset 逻辑末尾追加 `omni_audio_batch_cache_clear()`。

---

### 4.2 Python 前端改动 (cpp-eval/daily-omni)

#### 4.2.1 `eval_cpp_http_client.py` — 新增 API 调用

```python
class OmniServerClient:

    def prefill_audio_batch(self, audio_paths: list) -> dict:
        payload = {"audio_paths": audio_paths}
        resp = self._post_json("/v1/stream/prefill_audio_batch", payload, timeout=HTTP_TIMEOUT)
        return resp.json()

    def prefill_audio_from_cache(self, audio_segment_idx: int, cnt: int, audio_prompt: str = "\n") -> dict:
        payload = {
            "audio_path_prefix": "",
            "img_path_prefix": "",
            "cnt": cnt,
            "audio_segment_idx": audio_segment_idx,
            "prompt": audio_prompt,
        }
        resp = self._post_json("/v1/stream/prefill", payload, timeout=HTTP_TIMEOUT)
        return resp.json()

    def prefill_interleaved_v2(self, frame_paths, audio_paths, ...) -> int:
        # Step 1: 批量预计算所有音频
        if audio_paths:
            batch_result = self.prefill_audio_batch(audio_paths)

        # Step 2: 交错 prefill，音频用 segment_idx 引用缓存
        cnt = 0
        for i in range(num_pairs):
            self.prefill_image(img_path=frame_paths[i], cnt=cnt, ...)
            cnt += 1
            if audio_paths and i < len(audio_paths):
                self.prefill_audio_from_cache(audio_segment_idx=i, cnt=cnt, ...)
                cnt += 1
        # ... 处理多余 frames ...
        return cnt
```

#### 4.2.2 `eval_cpp_pipeline.py` — 切换到 v2 接口

```python
# process_sample() 中:
total_cnt = client.prefill_interleaved_v2(frame_paths, audio_seg_paths, ...)
```

`eval_cpp_audio_prep.py` **不需要改动**。

---

## 五、技术细节

### 5.1 Whisper Encoder 序列长度

| 音频时长 | mel frames | conv2 后 tokens | PE 消耗 | 最终 LLM tokens |
|----------|-----------|-----------------|---------|----------------|
| 1s | 100 | 50 | 50 | 10 |
| 10s | 1000 | 500 | 500 | 99 |
| 30s | 3000 | 1500 | 1500 (上限) | 299 |

- PE 容量 1500，来自 GGUF `encoder.positional_embedding` shape [1500, 1024]
- Daily-Omni 实测 max_mel_frames=3000, max_enc_seq_len=1500，恰好触及上限不超限
- **30s 切 chunk 是必须的**，与 Python 一致

### 5.2 Embedding 拆分策略

| 策略 | 描述 | 推荐？ |
|------|------|--------|
| **A. 逐 chunk forward + 后拆分** | 每个 ≤30s chunk 独立 forward，输出按段边界切分 | ✅ 推荐 |
| B. 真 batch forward (batch_size>1) | 修改计算图支持 batch 维度 | ❌ 改动太大 |

**选择策略 A**：
- Daily-Omni ≤30s 视频只有 1 个 chunk，策略 A 与 Python 完全等价
- 即使 >30s，Python 也是每个 ≤30s chunk 独立 encoder forward，chunk 间无信息交换
- 不需要修改计算图 batch 维度

### 5.3 拆分公式

Python 端精确公式（`_get_feat_extract_output_lengths()`）：
```python
conv_len = (input_lengths - 1) // 2 + 1      # conv2 stride=2
pool_len = (conv_len - pool_step) // pool_step + 1  # AvgPool stride=5
```

C++ 端合并后在一个长序列上做 forward，输出是连续的 token 序列。拆分需要：
1. 知道各段在 merged_pcm 中的 sample 边界
2. 推算各段在 mel frame 中的范围
3. 注意：**conv 在合并序列上连续计算，边界处有 kernel 重叠**，不能简单用逐段独立的公式

> ⚠️ 这里有一个关键的实现细节：Python 端每段是 batch 中的独立 sample（conv 独立处理），而 C++ 合并方案是一个长序列（conv 连续处理）。conv 边界处的输出会略有不同——这恰恰是我们要修复的 P1 差异。拆分时需要精确计算每段在 conv 输出中的 token 位置。

### 5.4 PE 溢出处理

重构后每个 chunk ≤30s → conv2 后 ≤1500 tokens ≤ PE 容量。不会出现 PE 溢出。

### 5.5 GPU 显存

| 模式 | 单次 forward 序列长度 | 显存需求 |
|------|---------------------|---------|
| 当前（逐段 ~1s） | ~50 tokens | 基准 |
| 重构后（≤30s chunk） | ≤1500 tokens | ~30x 基准 |

Whisper encoder 较小（~300M），1500 tokens 的 self-attention 在 80GB A100 上不成问题。

---

## 六、实施步骤

### Step 1: Mel 归一化对齐 + 基础 batch 框架

**预计改动量：~250 行 C++ / ~60 行 Python**

| 改动 | 文件 | 说明 |
|------|------|------|
| 新增 `audition_audio_preprocess_merged()` | audition.h/cpp | 合并 PCM → 30s 切 chunk → 复用 `log_mel_spectrogram()` |
| 新增 `omni_audio_batch_precompute()` + cache | omni.h/cpp | 逐 chunk 整体 forward（`streaming=false`，无 KV cache，**暂时 full attention**） |
| 新增 `/v1/stream/prefill_audio_batch` | server.cpp | HTTP 路由 + handler |
| 修改 `stream_prefill()` | omni.cpp | `audio_segment_idx` 参数，支持从 cache 取 embedding |
| 修改 `handle_stream_prefill_impl` | server.cpp | 解析新参数 |
| 修改 `handle_stream_reset_impl` | server.cpp | 追加 cache 清理 |
| 新增 `prefill_audio_batch()` + `prefill_interleaved_v2()` | eval_cpp_http_client.py | Python 前端调用新 API |
| 切换到 v2 接口 | eval_cpp_pipeline.py | 一行改动 |

**Step 1 效果**:
- ✅ **P0 mel 归一化对齐**（最大差异源消除）
- ✅ **P1 conv 边界对齐**（整个 chunk 的 mel 一起过 conv）
- ✅ **P2 STFT 边界对齐**
- ⚠️ Transformer 层暂时用 full attention（与 chunk mask / KV cache streaming 略有差异，但这个差异**远小于** P0 mel 修复带来的改善）

### Step 2: 添加 Chunk Attention Mask（精确对齐）

**预计改动量：~50-80 行 C++**

| 改动 | 文件 | 说明 |
|------|------|------|
| 修改 `build_whisper()` | audition.cpp | 非 streaming 模式下构建 chunk-causal mask (chunk_size=50) |

**Step 2 效果**:
- ✅ Transformer 层精确等价（chunk mask ≡ KV cache streaming ≡ Python）
- 此时 C++ 与 Python 端的 Whisper encoder 全流程完全对齐

### Step 3: 验证

| 任务 | 说明 |
|------|------|
| **特征数值验证** | 选 3-5 个代表性视频，dump C++ 和 Python 的 mel + encoder 输出，计算 MSE / cosine similarity |
| **端到端准确率** | Daily-Omni 全量 1197 条，对比重构前后准确率 |
| **AB 实验** | Step 1（无 mask）vs Step 2（有 mask）的准确率差异，量化 mask 的实际影响 |
| **回归测试** | 确保 VideoMME（无音频）pipeline 不受影响 |
| **性能 benchmark** | 对比推理速度和显存占用 |

---

## 七、文件改动汇总

| 文件 | 改动类型 | Step | 改动量估计 |
|------|---------|------|-----------|
| `llama.cpp-omni/tools/omni/audition.h` | 新增接口声明 | 1 | +30 行 |
| `llama.cpp-omni/tools/omni/audition.cpp` | 新增 `preprocess_merged` + `batch_encode_merged` | 1 | +150 行 |
| `llama.cpp-omni/tools/omni/audition.cpp` | 修改 `build_whisper` 加 chunk mask | 2 | +50 行 |
| `llama.cpp-omni/tools/omni/omni.h` | 新增 cache 结构 + 函数声明 | 1 | +30 行 |
| `llama.cpp-omni/tools/omni/omni.cpp` | 新增 `batch_precompute` + cache + 修改 `stream_prefill` | 1 | +100 行 |
| `llama.cpp-omni/tools/server/server.cpp` | 新增 batch API + 修改 prefill handler + reset 清理 | 1 | +60 行 |
| `cpp-eval/daily-omni/eval_cpp_http_client.py` | 新增 batch API + v2 prefill | 1 | +60 行 |
| `cpp-eval/daily-omni/eval_cpp_pipeline.py` | 切换到 v2 接口 | 1 | +5 行 |

**总计: ~485 行新增/修改，分布在 7 个文件中。**

---

## 八、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 30s chunk forward GPU OOM | 低 | Whisper encoder 较小 | 监控显存；极端情况回退逐段模式 |
| Embedding 拆分边界不精确 | 中 | conv 在合并序列上连续计算，边界推算需精确 | 小样本 dump 对比 Python 的 per-sample 独立输出 |
| build_whisper 改动引入 bug | 低 | Step 1 先不加 mask，Step 2 单独加 | AB 实验隔离 mask 的影响 |
| 向后兼容性 | 低 | 新增 API 不影响旧接口 | `audio_segment_idx` 默认 -1 走原逻辑 |

---

## 附录 A: Transformer 层等价性证明摘要

**结论：Python chunk attention mask 与 C++ KV cache streaming 在 Transformer encoder 层产生完全相同的计算结果。**

证明要点：
1. **单层等价**：对于任意两个 chunk (c0, c1)，C++ streaming 先 forward c0 存 KV，再 forward c1 attend 到 [c0_cached, c1]；Python mask 一次 forward 但 mask 令 c0 只看自己、c1 看 c0+c1。K/V 相同 → 输出相同。
2. **多层归纳**：Layer L 输入 = Layer L-1 输出（归纳假设相同）。FFN 逐 token，LayerNorm 逐 token，不跨 token 混合信息 → Layer L 输出相同。
3. **不存在"多层间接传播"**：chunk mask 在每一层都阻断未来 chunk 信息。Transformer 中没有能绕过 attention mask 的跨 token 操作。

> 勘误：v2 方案的附录 A 声称"Python chunk attention 严格优于 C++ KV cache 流式"，此结论**错误**，两者数学等价。

**差异全部来自 Transformer 之前**：mel 归一化范围不同（P0）、conv 边界 zero-padding vs 真实帧（P1）、STFT 反射 padding（P2）。

---

## 附录 B: 临时插桩代码（可还原）

在 `/cache/hanqingzhe/o45-py/modeling_minicpmo.py` 中添加了 `[NFRAMES]` 运行时追踪代码，数据收集完毕后可还原：
```bash
cd ~/o45-py && git checkout modeling_minicpmo.py
```
