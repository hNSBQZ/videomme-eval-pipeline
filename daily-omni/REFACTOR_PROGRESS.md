# 音频重构实施进度

> 基于: CPP_AUDIO_REFACTOR_PLAN.md (v3)
> 目标仓库: ~/Video-MME/llama.cpp-omni (分支: eval_daily_omni)
> 上次更新: 2026-03-25

---

## 已完成的改动

### ✅ 1. `audition.h` — 新增数据结构和接口声明

**文件**: `llama.cpp-omni/tools/omni/audition.h`

在文件末尾（`audition_whisper_clear_kv_cache` 声明之后）新增：

```cpp
struct audition_merged_audio_info {
    std::vector<float> merged_pcm;
    std::vector<size_t> segment_sample_boundaries;
    std::vector<size_t> segment_sample_lengths;
    int sr = 16000;
};

struct audition_audio_chunk_mel {
    audition_audio_f32 mel;
    int start_sample;
    int end_sample;
};

bool audition_audio_preprocess_merged(...);
bool audition_audio_batch_encode_merged(...);
```

补充声明 `audition_read_binary_file`（定义在 omni.cpp，audition.cpp 需要调用）：
```cpp
bool audition_read_binary_file(const char * fname, std::vector<uint8_t> * buf_res);
```

### ✅ 2. `audition.cpp` — 实现 `audition_audio_preprocess_merged()`

**文件**: `llama.cpp-omni/tools/omni/audition.cpp`

在文件末尾新增函数。逻辑：
1. 遍历所有 audio_paths，decode 为 PCM，拼接到 merged_pcm
2. 记录每段的 sample boundary 和 length
3. Pad 到 100ms 边界
4. 按 30s (480000 samples) 切 chunk
5. 每个 chunk 调用 `whisper_preprocessor::preprocess_audio()` 计算 mel
   - mel 归一化自然变为 chunk 内全局 → **解决 P0**
   - STFT 在合并波形上连续 → **解决 P2**

### ✅ 3. `audition.cpp` — 实现 `audition_audio_batch_encode_merged()`

同文件末尾新增。逻辑：
1. 关闭 streaming mode，清空 KV cache
2. 逐 chunk 构建 batch，调用 `audition_audio_batch_encode()` 做完整 forward
   - conv1/conv2 处理完整 chunk mel → **解决 P1**
3. 按段边界比例拆分 encoder 输出到各 segment
4. 恢复 streaming mode

**注意**：当前 Step 1 阶段使用 full attention（不加 chunk mask），Transformer 层暂时与 Python 有微小差异，但远小于 P0 mel 修复的改善。Step 2 再加 chunk attention mask。

### ✅ 4. `omni.h` — 新增 cache 结构 + 函数声明

**文件**: `llama.cpp-omni/tools/omni/omni.h`

在 `omni_context` 结构体中新增 `audio_batch_cache` 字段：
```cpp
struct audio_batch_cache_t {
    std::vector<omni_embed*> segment_embeds;
    bool valid = false;
} audio_batch_cache;
```

新增 3 个函数声明（在 `omni_clear_audio_kv_cache` 之后）：
```cpp
bool omni_audio_batch_precompute(struct omni_context * ctx_omni, const std::vector<std::string> & audio_paths);
struct omni_embed * omni_audio_batch_get_segment(struct omni_context * ctx_omni, int segment_idx);
void omni_audio_batch_cache_clear(struct omni_context * ctx_omni);
```

修改 `stream_prefill` 签名，增加 `int audio_segment_idx = -1` 参数。

### ✅ 5. `omni.cpp` — 实现 batch_precompute + cache + 修改 stream_prefill

**文件**: `llama.cpp-omni/tools/omni/omni.cpp`

**5a. 新增 3 个函数**（在 `omni_clear_audio_kv_cache` 之后）：

- `omni_audio_batch_cache_clear()`: 释放所有 cached embed，`cache.valid = false`
- `omni_audio_batch_get_segment(i)`: 返回 cache 中第 i 段的 embed 指针
- `omni_audio_batch_precompute()`:
  1. 调用 `audition_audio_preprocess_merged()` → 合并 + mel
  2. 调用 `audition_audio_batch_encode_merged()` → 整体 forward + 拆分
  3. 每段 embedding 封装为 `omni_embed`（malloc）存入 cache
  4. `cache.valid = true`

**5b. 修改 `stream_prefill()` 签名和音频处理段（sync + async 两个分支都要改）**：

- 签名增加 `int audio_segment_idx = -1`
- **sync 分支** (`if (!ctx_omni->async)`)：音频处理段增加优先分支，如果 `audio_segment_idx >= 0 && cache.valid`，从 cache 取 embedding 直接 eval（不 free，cache 拥有内存）
- **async 分支** (`else`)：同样增加优先分支，如果 `audio_segment_idx >= 0 && cache.valid`，从 cache **拷贝** embedding 到 `omni_embeds->audio_embed` 队列中
- 原有逻辑作为 `else if (aud_fname.length() > 0)` 分支保留（向后兼容）

> ⚠️ **踩坑记录**：首次实现时只改了 sync 分支，遗漏了 async 分支。Daily-Omni 评测实际运行在 async 模式下（`ctx_omni->async = true`，index=0 时会 create LLM thread），导致音频 embedding 完全没有注入，只 eval 了 prompt `"\n"`。表现为准确率反而下降（61.5%，比重构前低），server log 中帧间没有 `<|audio_start|>` / `<|audio_end|>` 标记。修复后 async 分支也支持从 cache 取 embedding。

### ✅ 6. `server.cpp` — 新增 batch API 路由 + 修改 prefill/reset handler

**文件**: `llama.cpp-omni/tools/server/server.cpp`

**6a. 新增 `handle_stream_prefill_audio_batch` handler**：
- 解析 `audio_paths` 数组
- 加锁调用 `omni_audio_batch_precompute()`
- 返回 `{status, n_segments, tokens_per_segment, total_tokens}`

**6b. 修改 `handle_stream_prefill_impl`**：
- 解析新的可选参数 `audio_segment_idx`（默认 -1）
- 传递给 `stream_prefill()`

**6c. 修改 `handle_stream_reset_impl`**：
- 在 reset 逻辑末尾追加 `omni_audio_batch_cache_clear(ctx_server.octx)`

**6d. 注册新路由**：
```cpp
svr->Post(params.api_prefix + "/v1/stream/prefill_audio_batch", handle_stream_prefill_audio_batch);
```

### ✅ 7. `eval_cpp_http_client.py` — 新增 batch API + v2 prefill

**文件**: `cpp-eval/daily-omni/eval_cpp_http_client.py`

新增 3 个方法：
- `prefill_audio_batch(audio_paths)` → POST /v1/stream/prefill_audio_batch
- `prefill_audio_from_cache(audio_segment_idx, cnt)` → POST /v1/stream/prefill（带 audio_segment_idx）
- `prefill_interleaved_v2(frame_paths, audio_paths, ...)` → 先调 batch，再交错 prefill（音频走 segment_idx 引用缓存）

原有 `prefill_interleaved()` 保留不动（向后兼容）。

### ✅ 8. `eval_cpp_pipeline.py` — 切换到 v2 接口

**文件**: `cpp-eval/daily-omni/eval_cpp_pipeline.py`

修改 `process_sample()` 中的调用：
```python
# 改前
total_cnt = client.prefill_interleaved(frame_paths, audio_seg_paths, ...)
# 改后
total_cnt = client.prefill_interleaved_v2(frame_paths, audio_seg_paths, ...)
```

---

## 已修复的编译/运行时问题

### 🔧 编译错误：`audition_read_binary_file` 未声明

`audition_audio_preprocess_merged()` 调用了 `audition_read_binary_file()`，该函数定义在 `omni.cpp` 中但没有在任何头文件中声明，导致 `audition.cpp` 编译报错 `was not declared in this scope`。

**修复**：在 `audition.h` 中补充声明。

### 🔧 运行时问题：async 分支遗漏 audio cache 逻辑

`stream_prefill()` 有 sync (`if (!ctx_omni->async)`) 和 async (`else`) 两个分支。首次实现只在 sync 分支添加了 `audio_segment_idx` / cache 逻辑，但 Daily-Omni 评测实际运行在 async 模式下（`ctx_omni->async = true`）。

**表现**：
- `prefill_audio_batch` 成功预计算并缓存了所有音频 embedding
- 但每次 `prefill_audio_from_cache` 调用走 async 分支时，只 eval 了 prompt `"\n"`，audio embedding 完全没有注入
- 模型在没有音频上下文的情况下答题，准确率从重构前的水平下降到 61.5%
- Server log 特征：帧间只有两个 `\n` token（image 的 frame_prompt + audio 的 audio_prompt），没有 `<|audio_start|>` / `<|audio_end|>` 标记

**修复**：在 async 分支同样添加 `audio_segment_idx >= 0 && cache.valid` 优先分支，从 cache 拷贝 embedding 到 `omni_embeds->audio_embed` 队列。

---

## 未完成的改动 (TODO)

### 🔲 重新验证（编译 + 全量评测）

修复 async 分支后需要重新编译和测试：
1. `cmake --build build` 编译
2. 小样本端到端测试：确认 server log 中出现 `<|audio_start|>` / `<|audio_end|>` 标记和正确的 audio token 数
3. 全量评测：Daily-Omni 1197 条对比重构前后准确率

### 🔲 Step 2: 添加 Chunk Attention Mask（精确对齐）

在 `build_whisper()` 中，非 streaming 模式下构建 chunk-causal mask (chunk_size=50)。
当前 Step 1 使用 full attention，与 Python 有微小差异（远小于 P0 mel 修复的改善）。

---

## 关键实现细节备忘

### sync vs async 两个分支
`stream_prefill()` 中 index >= 1 的处理有两条路径：
- **sync** (`!ctx_omni->async`)：直接 `eval_string` + `prefill_with_emb`，即时修改 `n_past`
- **async** (`ctx_omni->async`)：将 embed 数据拷贝到 `omni_embeds` 结构并推入队列，由 LLM thread 异步处理

**任何涉及 `stream_prefill` 音频/视频处理的改动都必须同时修改两个分支！**

Daily-Omni 评测模式下 `ctx_omni->async = true`（index=0 时创建 LLM thread），所有 index >= 1 的 prefill 走 async 分支。

### Token 拆分策略
当前用**比例拆分**（按各段占 chunk 的 sample 比例分配 token）。这不是精确公式，但对 ~1s 均匀切分的场景误差极小。如果后续 dump 对比发现问题，可改用精确公式：
```
mel_frames = n_samples / 160 + 1
conv_out = (mel_frames - 1) / 2 + 1  (conv2 stride=2, ceil-like for conv1d_ph)
pool_out = (conv_out - 5) / 5 + 1    (AvgPool k=5 s=5)
```

### Cache 内存管理
- `omni_audio_batch_precompute()` 分配 `omni_embed` 结构（malloc）
- sync 分支：`stream_prefill()` 从 cache 取 embed 指针直接使用，**不 free**（cache 拥有内存）
- async 分支：`stream_prefill()` 从 cache **拷贝**数据到 `omni_embeds->audio_embed` vector（LLM thread 消费后 delete omni_embeds 即释放 vector）
- `omni_audio_batch_cache_clear()` 负责释放所有 cached embed
- `handle_stream_reset_impl` 调用 cache clear 确保每条样本干净

---

## 文件改动清单

| 文件 | 状态 | 改动概述 |
|------|------|---------|
| `llama.cpp-omni/tools/omni/audition.h` | ✅ 已改 | +30 行，新增 2 个 struct + 2 个函数声明 + `audition_read_binary_file` 声明 |
| `llama.cpp-omni/tools/omni/audition.cpp` | ✅ 已改 | +200 行，新增 `preprocess_merged` + `batch_encode_merged` |
| `llama.cpp-omni/tools/omni/omni.h` | ✅ 已改 | 新增 cache struct + 3 个函数声明 + stream_prefill 签名改 |
| `llama.cpp-omni/tools/omni/omni.cpp` | ✅ 已改 | +120 行，新增 cache 管理函数 + 修改 stream_prefill **sync+async 两个分支** |
| `llama.cpp-omni/tools/server/server.cpp` | ✅ 已改 | +60 行，新增 batch API handler + 路由 + 修改 prefill/reset handler |
| `cpp-eval/daily-omni/eval_cpp_http_client.py` | ✅ 已改 | +90 行，新增 3 个方法（batch + from_cache + interleaved_v2） |
| `cpp-eval/daily-omni/eval_cpp_pipeline.py` | ✅ 已改 | 1 行改动，切换到 v2 接口 |

---

## 下一步

1. **重新编译**: `cd ~/Video-MME/llama.cpp-omni && cmake --build build`
2. **小样本验证**: 启动 server → 跑 1-2 个样本 → 确认 log 中有 `audio from batch cache segment` 和正确的 audio token 数
3. **全量评测**: Daily-Omni 1197 条对比重构前后准确率
4. **Step 2 (可选)**: 添加 chunk attention mask 实现精确对齐
