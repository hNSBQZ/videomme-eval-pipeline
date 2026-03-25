# 音频处理流水线深度核查报告

> 日期: 2025-03-25
> 背景: 对 C++ 推理端与 Python evalkit 评测端的音频处理链路进行逐行代码核查，澄清此前文档中的模糊或错误描述。

---

## 零、代码位置速查

| 角色 | 路径 |
|------|------|
| Python 推理代码（evalkit） | `~/Video-MME/evalkit` |
| Python 侧加载的模型代码（o45-py） | `~/o45-py` |
| C++ 推理前端代码/入口（本仓库） | `~/Video-MME/cpp-eval` |
| C++ 推理端（llama.cpp-omni） | `~/Video-MME/llama.cpp-omni` |

---

## 一、1秒音频 → 最终 token 数量的精确计算

### 结论：1秒 = 10 个 audio token

```
1s 音频 (16000 samples, sr=16kHz)
  │
  ├─ Mel Spectrogram: 80 × 100
  │    n_mels=80, hop_length=160 → 16000/160 = 100 帧
  │    mel 维度: [n_mels=80, n_frames=100]
  │
  ├─ Conv1 (kernel=3, stride=1, padding=1): 100 → 100 帧
  │
  ├─ Conv2 (kernel=3, stride=2, padding=1): 100 → 50 帧
  │    位置编码在这一步之后加（PE 维度=1500）
  │
  ├─ Transformer Encoder Layers（在 50 维序列上做 self-attention）
  │
  ├─ Projection Layer
  │
  └─ AvgPool1d (kernel=5, stride=5): 50 → (50-5)/5+1 = 10 tokens
      最终送入 LLM 的 audio token 数
```

### 代码证据

**C++ 侧** — `audition.cpp` token 数量计算：

```c++
// audition.cpp:1374-1384
const int n_tokens_after_conv = audio->nx / 2;  // conv2 stride=2 下采样
const int pool_k = 5;
const int pool_s = 5;
n_patches = (n_tokens_after_conv - pool_k) / pool_s + 1;
// 对于 nx=100: n_tokens_after_conv=50, n_patches=(50-5)/5+1=10
```

**Python 侧** — `modeling_minicpmo.py` 长度公式：

```python
# o45-py/modeling_minicpmo.py:419-427
def _get_feat_extract_output_lengths(self, input_lengths):
    input_lengths_after_cnn = (input_lengths - 1) // 2 + 1     # 100 → 50
    input_lengths_after_pooling = (
        input_lengths_after_cnn - self.config.audio_pool_step   # audio_pool_step=5
    ) // self.config.audio_pool_step + 1                        # (50-5)//5+1 = 10
    return input_lengths_after_cnn, input_lengths_after_pooling
```

### 常见时长对照表

| 音频时长 | mel 帧数 (80×T) | conv 后 | pool 后 (最终 token) | PE 消耗 (上限 1500) |
|---------|-----------------|---------|---------------------|-------------------|
| 1s      | 100             | 50      | **10**              | 50                |
| 5s      | 500             | 250     | **49**              | 250               |
| 10s     | 1000            | 500     | **99**              | 500               |
| 30s     | 3000            | 1500    | **299**             | 1500 (上限)       |

> **此前文档中的勘误**：`CPP_AUDIO_REFACTOR_PLAN.md` 第 235 行写 "30s → ~1500 tokens" 指的是 **conv 后的中间序列长度**（PE 覆盖范围），不是最终送入 LLM 的 token 数。最终 pool 后是 ~300 tokens。

---

## 二、C++ `stream_prefill` 能接受多长音频

### 结论

- **单次可接受任意长度音频**（函数入口无秒数限制）
- **硬限制来自位置编码**：`n_audio_ctx = 1500`（conv 后 token 数上限）
- **30 秒 = 3000 mel → 1500 conv tokens = PE 上限**
- **当前设计按 1 秒/次调用**，30 次后 PE 耗尽

### 关键约束

#### 2.1 位置编码长度 = 1500

```c++
// audition.cpp:862-865
get_u32("max_source_positions", hparams.n_ctx, false);
if (hparams.n_ctx == 0) {
    hparams.n_ctx = 1500; // whisper default audio context length
}
```

#### 2.2 Mel 计算无上限，30 秒 padding 已关闭

```c++
// audition.h:68-71
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

// audition.cpp:1889-1890 — 30s padding 已关闭
int64_t stage_1_pad = 0;  // 原始：WHISPER_SAMPLE_RATE * 30
```

mel 帧数 = `(n_samples + stage_2_pad*2 - n_fft) / hop_length`，随音频长度线性增长，无硬上限。

#### 2.3 PE 溢出时的行为

每次 forward 消耗 `n_tokens` 个 PE 位置，累积迭代次数 `kv_cache.iter`：

```c++
// audition.cpp:395-396
const int n_audio_ctx = hparams.n_ctx;  // 1500
const int n_iter = n_audio_ctx / n_tokens;  // 1500 / 50 = 30 (对于1s chunk)

// audition.cpp:412-417 — PE 耗尽时直接重置 KV cache（丢失所有历史！）
if (effective_iter >= n_iter) {
    LOG_WRN("Position encoding buffer exhausted");
    audition_whisper_clear_kv_cache(ctx);
    e_pe_offset = 0;
}
```

#### 2.4 PE 偏移隐含假设：每次 chunk 大小一致

```c++
// audition.cpp:419
e_pe_offset = embd_size * n_tokens * effective_iter;
```

偏移量用**当前这次**的 `n_tokens` 乘以 `effective_iter` 算。如果前几次送 1 秒（50 tokens），后面突然送 2 秒（100 tokens），偏移会跳跃错位——当前设计隐含假设每次 chunk 大小一致。

#### 2.5 总结表

| 场景 | mel 形状 | conv 后 tokens | 可否接受 | 备注 |
|------|---------|---------------|---------|------|
| 1s × 1 次   | 80×100  | 50   | ✅ | 标准工作模式 |
| 1s × 30 次  | 每次 80×100 | 累计 1500 | ✅ | PE 恰好填满 |
| 1s × 31 次  | — | 累计 > 1500 | ⚠️ | KV cache 被重置，丢失历史 |
| 30s × 1 次  | 80×3000 | 1500 | ✅ | 边界值，一次填满 |
| >30s × 1 次 | 80×3000+ | >1500 | ❌ | PE 溢出，抛异常或重置 |
| 混合长度     | — | — | ⚠️ | PE 偏移错位 |

---

## 三、Python evalkit 评测的完整音频处理链路

### 3.1 调用链总览

```
evalkit 评测入口 (eval_main.py)
  └─ minicpmo_ou._generate_chat()
       └─ model.chat(msgs, omni_input=True, merge_audio_from_same_content=True)
            │                       ↑ o45-py/modeling_minicpmo.py
            │
            ├─ ① 解析 msgs → images[], audios[] (np.ndarray), audio_parts[]
            │     每个 np.ndarray 记为一段独立音频，记录所属消息索引
            │
            ├─ ② self.processor() → mel 特征提取
            │     ├─ 合并同消息音频 (np.hstack)
            │     ├─ 30s 切分
            │     └─ batch mel 计算
            │
            ├─ ③ self.generate(**inputs)
            │     ├─ get_vllm_embedding()  → text + vision embeddings
            │     └─ get_omni_embedding(stream_input=False)
            │           └─ get_audio_embedding(chunk_length=1.0)  ← 离线模式
            │                 ├─ 构建 chunk attention mask
            │                 ├─ APM 一次性 forward（全部 segments batch）
            │                 ├─ projection
            │                 └─ AvgPool1d(k=5, s=5)
            │
            └─ ④ llm.generate() → 文本输出
```

**关键结论：evalkit 评测走 `get_audio_embedding()`（离线一次性 forward），不走 `streaming_prefill`。**

### 3.2 音频来源与切分

**交错模式**（`interleave_fps > 0`，Daily-Omni 评测默认）：

```python
# evalkit/o_e_Kit/utils/utils.py:781-784
media: List[Any] = []
for f, a in zip(frames, audio_segments):
    media.append(f)       # PIL.Image
    media.append(a)       # np.ndarray (按帧时间戳切的音频段)
return media
# 输出: [frame_0, audio_seg_0, frame_1, audio_seg_1, ..., prompt_text]
```

每帧配一段音频（通常约1秒），由 `get_video_frame_audio_segments()` 按帧时间戳 `[t_i, t_{i+1})` 切分。

**非交错模式**（`interleave_fps == 0`）：

```python
# minicpmo_ou.py:305-317
frames, waveform = load_video_and_audio(...)
content.extend(frames)         # 所有帧在前
content.append(waveform)       # 整段波形作为一个 ndarray 在后
```

### 3.3 processor 中的合并与30秒切分

**合并同消息音频**（`merge_audio_from_same_content=True` 时生效）：

```python
# o45-py/processing_minicpmo.py:1412-1424
if audio_parts is not None:
    # same audio part merge
    audio_part = audio_parts[idx]
    merge_audio = []
    cur_audio = []
    for aid, (part, audio) in enumerate(zip(audio_part, audios)):
        if aid == 0 or audio_part[aid] == audio_part[aid - 1]:
            cur_audio.append(audio)          # 同消息 → 累积
        else:
            merge_audio.append(np.hstack(cur_audio))  # 跨消息 → 拼接并断开
            cur_audio = [audio]
    if cur_audio:
        merge_audio.append(np.hstack(cur_audio))
```

> **交错模式下**：视频的所有帧+音频段都在同一条 user message 中 → `audio_parts` 索引全部相同 → **所有音频段被 `np.hstack` 拼成一条长波形**。

**超过30秒则按30秒切分**：

```python
# o45-py/processing_minicpmo.py:1428-1436
max_audio_inp_len = 30 * sampling_rate  # 480000 samples
for audio in merge_audio:
    if len(audio) <= max_audio_inp_len:
        final_merge_audio.append(audio)
    else:
        for i in range(math.ceil(len(audio) / max_audio_inp_len)):
            final_merge_audio.append(audio[i * max_audio_inp_len : (i + 1) * max_audio_inp_len])
```

**每段独立计算 mel，然后 batch pad**：

```python
# o45-py/processing_minicpmo.py:1454-1468
audio_inputs = self.audio_processor(
    final_merge_audio, sampling_rate=sampling_rate,
    return_attention_mask=True, padding="max_length", return_tensors="pt",
)
# 输出: audio_features shape = [num_segments, 80, max_mel_frames]
```

### 3.4 APM forward 中的 Chunk Attention（核心）

`get_audio_embedding()` 调用 APM 时，**不使用 KV cache**，但使用 **chunk attention mask**：

```python
# o45-py/modeling_minicpmo.py:704-716
if chunk_length > 0:                              # config: audio_chunk_length=1.0
    chunk_num_frame = int(chunk_length * 50)       # 1.0 * 50 = 50 (conv后的1秒)
    chunk_mask = self.subsequent_chunk_mask(
        size=max_seq_len, chunk_size=chunk_num_frame, num_left_chunks=-1,
    )
    audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

audio_attention_mask[audio_attention_mask_] = float("-inf")
audio_states = self.apm(
    wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
).hidden_states[self.audio_encoder_layer]
```

`subsequent_chunk_mask` 生成的是**下三角 block mask**（不是全双向，也不是 causal）：

```python
# o45-py/modeling_minicpmo.py:409-417
for i in range(size):
    if num_left_chunks < 0:
        start = 0                                          # 可看到所有历史
    else:
        start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
    ending = min((i // chunk_size + 1) * chunk_size, size) # 只看到当前 chunk 末尾
    ret[i, start:ending] = True
```

**mask 可视范围示意**（chunk_size=50, 3秒音频=150 conv tokens）：

```
Token 位置    可见范围
[0..49]       [0, 50)          ← 第0秒 chunk：只看自己
[50..99]      [0, 100)         ← 第1秒 chunk：看前2秒
[100..149]    [0, 150)         ← 第2秒 chunk：看全部3秒
```

---

## 四、Python Chunk Attention vs C++ KV Cache 流式：核心区别

### 4.1 关键结论：Transformer 层数学等价

**对于 Transformer encoder 层本身（self-attention + FFN + LayerNorm），Python 的 chunk attention mask 与 C++ 的 KV cache 流式产生完全相同的计算结果。** mask 精确模拟了 streaming 的因果关系。

### 4.2 等价性证明

#### 单层等价

以 2 个 chunk (c0, c1) 为例：

**C++ streaming：**
- 第 1 次 forward chunk 0：`Q_0 @ [K_0, V_0]` → `output_0`，存 K_0, V_0 到 cache
- 第 2 次 forward chunk 1：`Q_1 @ [K_0_cached, K_1], [V_0_cached, V_1]` → `output_1`

**Python chunk mask：**
- 一次 forward [chunk 0, chunk 1]，mask 令 chunk 0 只看自己：
  - chunk 0 位置：`Q_0 @ [K_0, V_0]`（mask 挡住 chunk 1） → `output_0`
  - chunk 1 位置：`Q_1 @ [K_0, K_1], [V_0, V_1]`（mask 放行） → `output_1`

K_0, V_0 在两边完全一样（同样的输入、同样的权重）→ **单层输出完全一致。**

#### 多层归纳

- **Layer 0 输入**：conv 后的 embedding + PE，两边一样（暂不考虑 conv 边界效应）
- **Layer 0 输出**：根据单层等价性 → 两边一样
- **Layer L 输入** = Layer L-1 输出（归纳假设相同）
- **Layer L 中**：
  - `K_i^L = W_K * (Layer L-1 output for chunk i)` — 两边一样
  - Attention pattern 一样（mask = streaming 的因果关系完全一致）
  - FFN 是 **position-wise** 的，不跨 token 混合信息
  - LayerNorm 是 **per-token** 的（沿 hidden dim 归一化），不跨 token
- → **Layer L 输出两边一样**

归纳完毕：**所有层的输出在 Transformer 部分完全一致**。

> ⚠️ **勘误**：本文档早期版本声称 "Python 的 chunk attention 在质量上严格优于 C++ 的 KV cache 流式，因为多层叠加允许信息间接跨 chunk 传播"。这是错误的。chunk mask 在**每一层**都阻断了未来 chunk 的信息流，Transformer 中不存在能绕过 attention mask 的跨 token 操作（FFN 逐 token、LayerNorm 逐 token），所以不存在 "通过多层间接传播" 的现象。

### 4.3 真正的差异来源：Transformer 之前的步骤

既然 Transformer 层数学等价，Python 和 C++ 的全部差异来自 **Transformer 之前**的预处理步骤：

| 差异来源 | 严重程度 | 说明 |
|---------|---------|------|
| **Mel 归一化范围** | **P0（最大）** | Python: ≤30s 段内全局 `max-8dB`；C++: ~1s 段内局部 `max-8dB`。音量波动大的视频中 mel 数值系统性偏移 |
| **Conv 边界效应** | P2（较小） | Python 的 conv1/conv2 在 chunk 边界处能看到相邻 chunk 的真实 mel 帧；C++ 每段独立 conv，边界处只看到 zero-padding。影响每个 chunk 边界的 ~2 帧 |
| **STFT 边界** | P2（很小） | mel 频谱计算时首尾的反射 padding 不同 |

### 4.4 Mel 归一化差异（已确认，最主要差异源）

| 维度 | Python | C++ |
|------|--------|-----|
| **合并策略** | 同消息的所有音频段 `np.hstack` 合并为一条长波形，再按 30s 切分 | 逐段独立（每段约1秒） |
| **mel 计算** | 每段（≤30s）独立算 mel | 每段（~1s）独立算 mel |
| **归一化基准** | 基于当前段（≤30s）的全局 max | 基于当前段（~1s）的局部 max |
| **影响** | 长段内归一化更稳定 | 短段间归一化基准跳跃大 |

Python 中 mel 归一化发生在 `WhisperFeatureExtractor` 内部（`log_spec = np.maximum(log_spec, log_spec.max() - 8.0)`），每次 `self.audio_processor(final_merge_audio)` 调用时，**每个 ≤30s 段独立归一化**。

C++ 侧 `audition.cpp:1929` 同样做 `clamp(mel, mmax-8.0)` 的归一化，但每段只有约1秒。

**关键差异**：Python 合并后段长可达30秒，归一化基于30秒内的全局 max；C++ 每段约1秒，归一化基于1秒内的局部 max。在音量变化剧烈的视频中，这会导致 mel 特征数值的系统性偏移。

---

## 五、Python PE 溢出处理 vs C++ PE 溢出处理

**Python** — 超过1500时 repeat 最后一个 PE（不crash，但有 warning）：

```python
# o45-py/modeling_minicpmo.py:3977-3989
if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
    logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")
    embed_pos_front = embed_pos[past_key_values_length:, :]
    embed_pos = torch.cat((
        embed_pos_front,
        torch.repeat_interleave(embed_pos[-1, :].unsqueeze(0), overflow_count, dim=0),
    ))
```

**C++** — 超过 n_iter 时直接重置 KV cache（丢失所有历史）：

```c++
// audition.cpp:412-417
if (effective_iter >= n_iter) {
    LOG_WRN("Position encoding buffer exhausted");
    audition_whisper_clear_kv_cache(ctx);
    e_pe_offset = 0;
}
```

---

## 六、待验证清单（更新）

基于以上核查，Transformer 层的 attention 行为已确认等价，差异全部来自 Transformer 之前的预处理。优先级如下：

- [ ] **[P0] Mel 归一化对齐（唯一显著差异源）**：让 C++ 端先合并所有音频段的 PCM → 按30s切分 → 每段独立算mel（与 Python processor 行为一致）
- [ ] **[P1] Conv 边界效应对齐**：让 C++ 端合并 PCM 后再过 conv，消除 chunk 边界处的 zero-padding 差异（影响每段边界 ~2 帧）
- [ ] **[P1] 特征数值验证**：dump Python 和 C++ 在同一视频上的 mel 特征，计算 MSE 以量化 mel 归一化差异的实际影响
- [x] ~~**[已排除] Attention 模式差异**~~：chunk mask 与 KV cache streaming 在 Transformer 层数学等价，无需对齐
- [x] **[已确认] `merge_audio_from_same_content=True` 在 Python chat 路径生效**：processor 中 `audio_parts` 索引相同时会 `np.hstack` 合并
- [x] **[已确认] Python 评测走离线模式**：`get_audio_embedding(chunk_length=1.0)`，非 streaming
- [x] **[已确认] 1s = 10 audio tokens**：mel 100 → conv 50 → pool 10
