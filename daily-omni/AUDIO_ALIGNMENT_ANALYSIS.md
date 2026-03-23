# Daily-Omni C++ vs Python 音频对齐差异分析

> 日期: 2026-03-19
> 背景: Daily-Omni 测评中 C++ 推理端与 Python (o45-py) 推理端存在约 7 分的准确率差距。
> 本文档按可能造成影响的严重程度排序，列出所有已发现的差异点。

---

## P0 级：最可能造成大分差的差异

### 1. 音频合并策略：Python 先拼后算 vs C++ 逐段独立

**Python 路径** (`merge_audio_from_same_content=True`):

```
原始音频: [seg_0(1s), seg_1(1s), ..., seg_N(1s)]
         ↓ np.hstack()
合并波形: [一个 N 秒的完整波形]
         ↓ WhisperFeatureExtractor
一份 mel: [80, ~N*100 帧], 全局归一化
         ↓ conv1 → conv2(stride=2) → Transformer(chunk_attn) → proj → pool
一批嵌入: [N*10 tokens, 4096]
```

**C++ 路径** (逐段 stream_prefill):

```
原始音频: [seg_0(1s), seg_1(1s), ..., seg_N(1s)]
         ↓ 每段独立处理
N 份 mel: 每份 [80, ~100 帧], 各自独立归一化
         ↓ 每段独立: conv1 → conv2 → Transformer(KV cache) → proj → pool
N 批嵌入: 每批 [10 tokens, 4096]
```

**差异影响**:

| 环节 | Python (合并) | C++ (逐段) | 影响范围 |
|------|-------------|-----------|---------|
| STFT | 跨段边界连续 | 每段首尾有 reflect/zero padding | 每段首尾 2-3 帧 |
| mel 归一化 | `global_max - 8dB` 全局 | `local_max_i - 8dB` 每段独立 | **所有帧** |
| conv1/conv2 | 跨段边界看到相邻段数据 | 每段边界看到 padding | 每段首尾 1-2 帧 CNN 输出 |
| Transformer | chunk_attn 一次 forward | KV cache 模拟等价 attention | 等价（已验证） |

**mel 归一化是最大差异**：假设 10 段音频中，段 3 很安静 (local_max=-5)，段 7 很响 (local_max=-1)：
- Python global: `floor = -1 - 8 = -9` → 段 3 的值被截到 -9
- C++ per-seg: 段 3 `floor = -5 - 8 = -13` → 保留更多低能量细节

所有段的 mel 值都不同，Whisper 编码器的输入整体漂移。

**验证方法**: 对同一视频样本，Python 端 dump 合并后的 mel 频谱 (全局归一化后)，
C++ 端 dump 逐段 mel 频谱拼接结果，计算 MSE。如果 MSE 显著 → 这是主因。

---

### 2. omni_input / omni_mode 参数名不匹配（需验证实际行为）

evalkit `minicpmo_ou.py` 调用:

```python
response = self.model.chat(
    omni_input=omni_input,   # ← 传的是 omni_input
    merge_audio_from_same_content=True,
    ...
)
```

o45-py `modeling_minicpmo.py` 定义:

```python
def chat(self, ..., omni_mode=False, ..., **kwargs):
```

**参数名不匹配**: `omni_input` 会落入 `**kwargs` 被忽略，`omni_mode` 保持默认值 `False`。

这影响的是 content 的拼接方式:
- `omni_mode=True`: `"".join(cur_msgs)` → 媒体项之间**无换行**
- `omni_mode=False`: `"\n".join(cur_msgs)` → 媒体项之间**有换行**

C++ 的 `frame_prompt="\n"` 和 `audio_prompt="\n"` 实现的是有换行的方式，
所以如果 `omni_mode` 实际为 `False`（有换行），那 C++ 是**对齐**的。

**但需验证**: evalkit 使用的 model checkpoint 中的 `chat()` 函数是否与 o45-py 版本完全一致。
如果 checkpoint 中 `chat()` 的形参名是 `omni_input` 而非 `omni_mode`，那行为会完全不同。

**验证方法**: 直接检查 evalkit 实际加载的 model checkpoint 中 `modeling_minicpmo.py` 的 `chat` 签名。

---

### 3. merge_audio_from_same_content 在 Python 端可能存在 shape mismatch

如果音频确实被合并，存在一个潜在问题:

- `audio_ph_list` 有 N 个占位符（每个原始段一个），总计 N×10 个 `<unk>` token
- `audio_features` 只有 1 个合并的 mel 特征
- `audio_feature_lens_list = [[total_mel_len]]` 只有 1 个长度
- `get_audio_embedding` 输出 1 个 [N×10, hidden] 的 embedding tensor
- `audio_bounds` 有 N 个 bound（每个 10 token）
- `zip(audio_embs, bounds)` 只迭代 1 次 → 只有第一个 bound 被填充

**如果 assertion `embs.shape[0] == len(audio_indices)` 触发**: 100 ≠ 10 → crash

**可能的解释**:
1. evalkit 的 model checkpoint 中有不同的处理逻辑
2. `merge_audio_from_same_content` 参数在某些情况下被覆盖为 False
3. o45-py 版本已修复此问题但我们看到的代码还没有

**验证方法**: 在 evalkit 中对一个有多段音频的样本开 debug，print audio_feature_lens_raw 和 audio_bounds 的结构。

---

## P1 级：可能造成 1-3 分差的差异

### 4. Prompt 模板：多余的换行符

**evalkit 最终 token 序列** (最后一段音频后):

```
...audio_tokens <|audio_end|> \n Carefully read the following question...
```

其中 `\n` 来自 `"\n".join(cur_msgs)` 的分隔。

**C++ token 序列**:

```
...audio_tokens <|audio_end|> \n \n Carefully read the following question...
```

第一个 `\n` 来自 `audio_prompt="\n"`，第二个 `\n` 来自 `USER_PROMPT_TEMPLATE` 的开头:

```python
USER_PROMPT_TEMPLATE = (
    "\nCarefully read..."   # ← 开头有 \n
)
```

**多了一个 `\n` token**。虽然只是一个 token 的差别，但在 attention 中改变了
question text 相对于所有 media token 的位置关系。

**修复**: 去掉 `USER_PROMPT_TEMPLATE` 开头的 `\n`。

---

### 5. 空音频段处理差异

evalkit `get_video_frame_audio_segments`:
```python
if segment.size == 0:
    continue           # 跳过空段
```

C++ `segment_audio_by_timestamps`:
```python
segment = waveform[start_sample:end_sample]
segments.append(segment)   # 保留空段
```

**影响**: 如果某些时间戳产生空音频段:
- evalkit: 跳过 → 帧和音频对数减少，后续帧对应不同的音频段
- C++: 保留空段 → 空 WAV 被送入 server，可能产生异常或全零 embedding

**验证方法**: 检查是否有样本触发了空段（尤其是视频首帧 timestamp=0 且 duration 很短的情况）。

---

### 6. max_slice_nums 差异

| | evalkit | C++ |
|---|--------|-----|
| 视频场景 | `max_slice_nums=1` | `MAX_SLICE_NUMS=0` |

evalkit 中 `1` 表示"最多 1 个 slice"（整图不分割），
C++ 中 `0` 注释说"不分块"。

**需确认**: C++ server 对 `max_slice_nums=0` 的处理是否等同于 Python 的 `1`。
如果 `0` 被解释为"无限制"或导致不同的图片编码行为，会影响所有视觉 token。

---

## P2 级：影响较小但需注意

### 7. 采样参数 — ✅ 已验证：两端一致，非问题项

evalkit `minicpmo_ou.py` 调用 `model.chat(sampling=False, ...)`，
但 o45-py `chat()` 的形参名是 `do_sample`（非 `sampling`）：

```python
# o45-py modeling_minicpmo.py
def chat(self, ..., do_sample=True, ..., **kwargs):
```

`sampling=False` 落入 `**kwargs`，`do_sample` 保持默认值 `True`。
随后 `prepare_generation_config(do_sample=True)` 走 sampling 分支：

```python
if do_sample:
    generation_config.update({
        "top_p": 0.8, "top_k": 100,
        "temperature": 0.7, "do_sample": True,
        "repetition_penalty": 1.02,
    })
```

`kwargs` 中的 `sampling` 也不会被 `generation_config.update(... & kwargs.keys())` 拾取，
因为 `"sampling"` 不在 `generation_config` 的 key 集合中。

**最终实际参数对比**：

| 参数 | Python (evalkit + o45-py) | C++ config |
|------|--------------------------|------------|
| temperature | 0.7 | 0.7 |
| top_p | 0.8 | 0.8 |
| top_k | 100 | 100 |
| repetition_penalty | 1.02 | 1.02 |
| do_sample | True | True (sampling) |

**结论：两端采样参数完全一致。** 唯一区别是 Python 端通过 `torch.manual_seed(0)` 固定了随机种子，
C++ 端没有，但这不影响统计平均准确率。此项可从关注列表中划掉。

---

### 8. Mel 频谱计算的 n_fft 存储差异

Python `convert_apm.py`:
```python
filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
fout.add_uint32("n_fft", filters.shape[1])   # 201 = n_fft/2 + 1
```

C++ `audition.h`:
```c
#define WHISPER_N_FFT 400
```

但 C++ 从 GGUF 加载的 `n_fft` 实际是 `201`（mel filter bank 的列数，非 FFT 窗口大小）。
需确认 mel 计算时使用的实际窗口大小一致（应为 400）。

---

### 9. 音频变速 fallback 差异（仅在无 pyrubberband 时触发）

evalkit:
```python
target_len = int(len(waveform) / speed)
librosa.resample(waveform, orig_sr=len(waveform), target_sr=target_len)
```

C++:
```python
new_sr = int(sr * speed)
librosa.resample(waveform, orig_sr=new_sr, target_sr=sr)
```

数学上等价但 resampling 路径不同，数值结果可能有微小差异。
**仅在 OOM 重试触发变速时才有影响**（正常推理 speed=1.0 不触发）。

---

## 已验证对齐的部分

| 项目 | 说明 |
|------|------|
| 帧采样逻辑 | 长/短视频分支一致，timestamps 计算方式相同 |
| 音频采样率 | 均为 16kHz mono |
| 音频加载 | 均使用 `librosa.load(path, sr=16000, mono=True)` |
| 交错顺序 | 均为 `frame_0, audio_0, frame_1, audio_1, ...` |
| Whisper KV cache vs chunk attention | attention 模式等价（chunk_size=50, num_left_chunks=-1） |
| 位置编码 | C++ 递增位置编码 (0-49, 50-99, ...) 与 Python 合并序列的自然位置一致 |
| `<think>` 块注入 | C++ decode 时注入 `<think>\n\n</think>\n\n<|tts_bos|>`，对齐 Python |
| 答案提取 | 均提取 A/B/C/D 字母 |

---

## 建议验证优先级

1. **[P0]** 确认 evalkit 实际 model checkpoint 中 `chat()` 的 `merge_audio_from_same_content` 是否真正生效 → 如果不生效（crash 或被忽略），则 Python 端也是逐段独立处理，差异根源不在音频合并
2. **[P0]** 对同一样本 dump Python 和 C++ 的 mel 频谱数据做数值对比
3. **[P1]** 修复 prompt 模板多余 `\n`
4. ~~**[P1]** 检查 C++ decode 是否使用了 greedy 还是 sampling~~ → ✅ 已验证，两端一致
5. **[P1]** 检查空音频段是否有样本触发
6. **[P1]** 确认 `max_slice_nums=0` 在 C++ server 中的行为
7. **[P1]** 排查 C++ server audio 通路的内存泄漏（见下方 "C++ Server 稳定性问题"）

---

## 分差来源猜测（总结）

| 差异来源 | 估计影响 | 确定程度 |
|---------|---------|---------|
| mel 归一化 (全局 vs 逐段) | 2-4 分 | 高 |
| ~~采样参数 (sampling vs greedy)~~ | ~~0-3 分~~ | ✅ 已排除，两端一致 |
| prompt 多余 `\n` | 0.5-1 分 | 高 |
| merge_audio 行为差异 | 0-3 分 | 需验证 |
| 空段处理 / 对齐偏移 | 0-1 分 | 中 |
| max_slice_nums 差异 | 0-1 分 | 需验证 |
| C++ server 音频通路不稳定 | 间接影响 | 已确认 |

---

## C++ Server 稳定性问题（已确认）

> 日志来源: `daily_omni_20260319_104606.log`

### 现象

C++ 评测 1197 样本中出现 **242 条空预测**（占 20.2%），需 rerun 恢复。
而 Python evalkit 同数据集 **0 失败**，VideoMME（纯视频、无音频）C++ 评测也 **0 失败**。

### 各 GPU 失败分布

| GPU | 端口 | 失败数 | 说明 |
|-----|------|--------|------|
| GPU 0 | 9080 | **194** | 服务器进程崩溃，后续全部 Connection refused |
| GPU 1 | 9081 | 15 | 散发 500 错误 |
| GPU 2 | 9082 | 17 | 散发 500 错误 |
| GPU 3 | 9083 | 16 | 散发 500 错误 |

### GPU 0 崩溃时间线

```
样本 112 (02UvvE1oA1I): 500 Server Error on /v1/stream/prefill  ← 开始不稳定
样本 113 (03aJ_RcnBko): 成功 (Pred=A, 5.6s)                    ← 暂时恢复
样本 114 (02XbmweOzOY): 500 Server Error on /v1/stream/prefill  ← 再次出错
样本 115 (07rwFurzpw8): RemoteDisconnected                      ← 进程死亡
样本 116+ :              Connection refused (0.1~0.3s/条)        ← 全部失败
```

服务器日志 (`server_gpu0_20260319_104606.log`) 无 SIGSEGV/abort 信息，日志直接截断，
推测进程被系统 OOM killer 终止或遭遇未捕获的 segfault。

### 所有 500 错误均发生在 prefill 阶段

各 GPU 的散发性 500 错误全部指向 `/v1/stream/prefill`（非 decode），
说明是音频/视觉 prefill 处理时 server 端出错。

### Rerun 全部成功

Rerun 启动全新 llama-server 进程，在 GPU 0 上串行处理全部 242 条失败样本，
**无一失败**（包括原始"致命"样本 `07rwFurzpw8`，rerun 中 2 秒正常完成）。

### 对比证据：问题锁定在 audio 通路

| 测评 | 是否有音频 | server 错误 | 崩溃 |
|------|-----------|-------------|------|
| VideoMME (C++) | 无 | **0** | **0** |
| Daily-Omni (C++) | 有（逐段 prefill） | **~56 条 500** | **1 次进程死亡** |
| Daily-Omni (Python) | 有（合并处理） | **0** | **0** |

VideoMME 的 `server_gpu0.log`（95 万行）全程无异常，处理所有样本未出过一个错。
Daily-Omni 与 VideoMME 的唯一区别是每个样本额外 prefill 了 30-60 段音频。

### 根因分析

C++ llama-server 在处理音频时的资源管理存在问题：

1. **内存泄漏/碎片化**：每次 `audition_audio_preprocess` + `build_whisper` 后，
   中间 buffer（mel 频谱、Whisper encoder hidden states、projection 输出等）
   可能未完全释放。连续处理 ~100 样本（每样本 60 段音频 = ~6000 次音频编码）后累积到阈值。

2. **Reset 不彻底**：`audition_whisper_clear_kv_cache` 只清了 Whisper KV cache，
   但 audio encoder 内部的其他状态（计算图 buffer、临时分配等）可能未回收。

3. **Prefill 频次差异巨大**：
   - VideoMME：每样本 ~64 次 prefill（纯图片帧）
   - Daily-Omni：每样本 ~120 次 prefill（60 帧 + 60 段音频），分配释放频率翻倍

### 排查方向

重点排查 `llamacpp.omni` 中 audio 相关代码路径的资源生命周期：
- `audition_audio_preprocess`（mel 频谱计算）
- `build_whisper`（Whisper encoder forward）
- audio embedding 的 ggml_context / ggml_backend_buffer 是否在每次 prefill 后正确释放
- 连续运行 200+ 样本（含音频）的显存监控
