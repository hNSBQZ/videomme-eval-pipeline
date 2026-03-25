# Daily-Omni C++ vs Python 推理端差异与排查总结

> 日期: 2026-03-24
> 背景: 总结 Daily-Omni 测评中 C++ 推理端与 Python (o45-py) 推理端的差异点、稳定性排查结果以及后续需要关注的问题。

---

## 零、代码位置速查（重要）

- **Python 推理代码（evalkit）**: `~/Video-MME/evalkit`
- **Python 侧加载的模型代码（o45-py）**: `~/o45-py`
- **C++ 推理前端代码/入口（本仓库）**: `~/Video-MME/cpp-eval`（当前工作目录也可从 `@/cache/hanqingzhe/.cursor/projects/cache-hanqingzhe-Video-MME-cpp-eval/terminals/1.txt:7-8` 看到 `~/Video-MME/cpp-eval$`）
- **C++ 推理端（llama.cpp-omni）**: `~/Video-MME/llama.cpp-omni`

## 一、 已排查并解决的问题

### 1. C++ Server 稳定性问题与空路径报错（重点修复）
- **现象**: 之前在 C++ Server 测试中，发现大量散发的 `500 Server Error`，主要集中在 prefill 阶段，导致约 20.2% 的样本预测失败或崩溃（即原报告中 `GPU 0 崩溃时间线` 等现象）。
- **根本原因**: 并非 C++ Server 自身的音频显存泄漏，而是**原始的 Python 评测代码逻辑问题**。Python 代码会将同一个视频的不同任务分发给不同的推理端处理。在这种并发分发的过程中，视频和音频的切片文件路径有时会被其他进程/任务**提前删除**，导致 C++ Server 在读取时遇到**空路径错误**（File Not Found/Empty Path），从而引发 500 报错和后续的崩溃。
- **状态**: ✅ **已解决**。通过修正切片文件的生命周期和分发逻辑，排除了该稳定性干扰项。

### 2. Decode 实现路径与异步 Prefill
- **排查结果**: omni-cli 与 Server Pipeline 调用相同的 `stream_decode` 逻辑，在非双工 (`use_tts=false`) 模式下，文本生成的 token 不会被意外过滤。
- **异步同步机制**: 异步 Prefill 的 FIFO 队列保序逻辑严格，`decode` 会正确等待所有前置的 `image` 和 `audio` 的 prefill 完成，不会导致乱序。
- **状态**: ✅ **已排除**，行为与 Python 一致。

### 3. Prompt 模板与格式对齐
- **Assistant Prompt**: C++ 的 `<think>\n\n</think>\n\n<|tts_bos|>` 注入与 Python 开启 `use_tts_template=True` 时的行为一致。
- **选项格式与换行符**: 选项拼接（`A. xxx\n`）以及消除媒体与文本边界之间多余的双 `\n` 问题已排查修复，确保输入 token 序列结构对齐。
- **状态**: ✅ **已解决**。

### 4. 采样参数
- **采样参数**: 两端实际生效的参数一致（`temperature=0.7`, `top_p=0.8`, `top_k=100`, `repetition_penalty=1.02`）。Python 端的 `sampling=False` 参数因不匹配已被忽略，默认走 `do_sample=True`。
- **状态**: ✅ **已排除**。

---

## 二、 可能还存在的问题（待进一步验证的差异）

尽管排除了上述问题，C++ 与 Python 之间如果依然存在准确率分数差，大概率来源于以下核心差异：

### 1. [P0] Whisper 音频编码器整体行为差异（含 KV Cache、Attention 模式、Mel 归一化）

> ⚠️ 音频编码器两端的工作方式存在**根本性差异**，需要从头系统排查。以下分项列出已确认的差异点。

#### 1a. Whisper KV Cache / Attention 模式 → ✅ 已确认等价
- **跨样本清理已对齐**: C++ 端每条样本前通过 `client.reset()` → `omni_clear_audio_kv_cache` 清零 buffer 并重置 `iter=0`；Python 端 `chat()` 路径走 `get_audio_embedding()`（非 streaming），天然无状态。两端均无跨样本污染。
- **单样本内 Transformer 行为等价**:
  - **Python**: `get_audio_embedding(chunk_length=1.0)` 使用 `subsequent_chunk_mask`（下三角 block mask），每 1s chunk 只能 attend 到自己和所有历史 chunk。
  - **C++**: 逐段 prefill，KV cache 累积，每段 attend 到所有历史 KV + 当前段。
  - **数学等价**：chunk mask 精确模拟了 streaming 的因果关系。Transformer 中 FFN 逐 token、LayerNorm 逐 token，不存在能绕过 mask 的跨 token 操作，因此多层叠加也不会产生差异。详见 `AUDIO_PIPELINE_DEEP_DIVE.md` 第四章的归纳证明。
- **状态**: ✅ **已排除**，Transformer 层行为一致。

#### 1b. Mel 频谱归一化策略差异
- **Python 端**: 倾向于先使用 `np.hstack()` 将所有音频段拼接成一个完整的长波形，然后整体提取 Whisper 特征。**Mel 归一化是全局的**（基于整个长音频的最大值减去 8dB）。
- **C++ 端**: 逐段独立处理音频（逐段 prefill）。每段独立计算 Mel 频谱，**归一化是局部的**（基于当前小段的最大值）。
- **影响**: 两端传入 Transformer 的底层输入特征在绝对数值上会产生整体漂移（尤其音量起伏大的视频）。
- **建议验证**: dump 同一视频 Python 全局合并后 vs C++ 逐段的 mel 特征，计算 MSE；确认 Python 端 `merge_audio_from_same_content` 是否真正生效。

#### 1c. 待排查清单
- [ ] 确认 C++ streaming attention 与 Python full attention 对 encoder 输出特征的数值差距（小样本 cosine similarity）
- [ ] 确认 C++ 逐段 Mel 归一化 vs Python 全局归一化的特征 MSE
- [ ] 确认 `merge_audio_from_same_content=True` 在 Python chat 路径是否真正生效（processor 内部是否实际合并）
- [ ] 确认 C++ `whisper_streaming_mode` 的默认值来源，是否有配置可切换为非 streaming 模式以对齐 Python 行为

### 2. [P1] 特殊配置与异常参数的影响
- **参数名不匹配**: Python 代码中 `omni_input` 和 `omni_mode` 可能存在 kwargs 传参未能正确覆盖默认值的情况，需检查这是否导致了预期的合并逻辑失效。
- **空音频段与变速 Fallback**: 遇到空段时，Python 可能会 `continue` 跳过，而 C++ 可能会将其送入模型；在极少数触发变速的场景下，两边 `librosa.resample` 的实现细节有微小差异。

