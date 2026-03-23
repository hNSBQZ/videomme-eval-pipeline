# C++ 推理端需要确认/修改的事项

本文档记录 Daily-Omni 评测流水线中，C++ 推理端（llama-server / llama.cpp-omni）可能需要调整的部分。

---

## 0. omni_init（会话初始化）

### 请求
**POST** `/v1/stream/omni_init`

```json
{
    "media_type": 2,
    "use_tts": false,
    "n_predict": 128,
    "model_dir": "/path/to/gguf"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `media_type` | int | `2` = omni（audio + vision），VideoMME 中也是 2 |
| `use_tts` | bool | Daily-Omni 设为 `false`（不需要 TTS 输出） |
| `n_predict` | int | 最大生成 token 数，默认 128 |
| `model_dir` | str | GGUF 模型文件目录 |

### 调用时机
每个 llama-server 进程启动并通过 `/health` 健康检查后，**调用一次**即可。后续多条样本复用同一初始化。

### 需确认
- [ ] `media_type=2` 是否同时启用视觉和音频编码器
- [ ] `use_tts=false` 下 decode 是否仍可能输出 `<|tts_bos|>` / `<|tts_eos|>` token

---

## 0.5 reset（KV cache 重置）

### 请求
**POST** `/v1/stream/reset`

```json
{}
```

无任何参数，空 JSON body。

### 调用时机
**每条样本处理前**调用一次，清空上一条样本遗留的 KV cache，确保上下文干净。

### 需确认
- [ ] reset 是否完全清空 KV cache（包括 system prompt 部分）
- [ ] reset 后 prefill 的 `cnt` 是否需要从 0 重新开始（Python 端的行为是从 0 开始）

---

## 1. 音频 prefill 支持

### 当前状态
VideoMME 评测中 `audio_path_prefix` 始终为空字符串，未实际使用音频 prefill。

### Daily-Omni 需要
Python 端会通过 HTTP `/v1/stream/prefill` 发送音频段文件路径：

```json
{
    "audio_path_prefix": "/path/to/audio_seg_000.wav",
    "img_path_prefix": "",
    "cnt": 1,
    "prompt": "\n"
}
```

### 需确认
- [ ] 服务端是否已实现从 `audio_path_prefix` 加载 WAV 文件并 prefill 到 KV cache
- [ ] WAV 格式要求：16kHz mono float32（与 librosa 输出一致）
- [ ] 音频 prefill 后生成的 token 结构是否为 `<|audio_start|><unk>×N<|audio_end|>`

---

## 2. 交错 prefill 支持

### Daily-Omni 的 content 结构
```
[frame_0, audio_seg_0, frame_1, audio_seg_1, ..., frame_N-1, audio_seg_N-1, text_prompt]
```

Python 端按此顺序交替调用 `prefill_image` 和 `prefill_audio`，cnt 从 0 递增。

### 完整 prefill 序列示例（3 帧 + 3 音频段 + 文本）

| 步骤 | 类型 | cnt | payload 关键字段 |
|------|------|-----|------------------|
| 1 | 图片 | 0 | `img_path_prefix="/tmp/frame_000.jpg"`, `audio_path_prefix=""`, `max_slice_nums=0`, `skip_system_prompt=true` |
| 2 | 音频 | 1 | `audio_path_prefix="/tmp/audio_seg_000.wav"`, `img_path_prefix=""` |
| 3 | 图片 | 2 | `img_path_prefix="/tmp/frame_001.jpg"`, `audio_path_prefix=""`, `max_slice_nums=0` |
| 4 | 音频 | 3 | `audio_path_prefix="/tmp/audio_seg_001.wav"`, `img_path_prefix=""` |
| 5 | 图片 | 4 | `img_path_prefix="/tmp/frame_002.jpg"`, `audio_path_prefix=""`, `max_slice_nums=0` |
| 6 | 音频 | 5 | `audio_path_prefix="/tmp/audio_seg_002.wav"`, `img_path_prefix=""` |
| 7 | 文本 | 6 | `audio_path_prefix=""`, `img_path_prefix=""`, `prompt="Carefully read..."` |

- 所有 prefill 均为 **POST** `/v1/stream/prefill`，只是 payload 中 `img_path_prefix` / `audio_path_prefix` 不同
- 图片/音频 prefill 的 `prompt` 均为 `"\n"`（对齐 Python `"\n".join`）
- 第一次图片 prefill 携带 `skip_system_prompt=true`

### 需确认
- [ ] 服务端是否正确处理交替的 image/audio prefill（即 cnt=0 是图片，cnt=1 是音频，cnt=2 是图片...）
- [ ] 每次 prefill 的 `prompt="\n"` 是否会正确插入换行 token（对齐 Python `"\n".join`）
- [ ] 最终 decode 前的 KV cache 内容是否与 Python 端一致

---

## 3. 文本 prefill

### 请求
**POST** `/v1/stream/prefill`

```json
{
    "audio_path_prefix": "",
    "img_path_prefix": "",
    "cnt": 6,
    "prompt": "Carefully read the following question and select the letter corresponding to the correct answer.Highlight the applicable choices without giving explanations.\n{question}\nOptions:\n{options}Please select the correct answer from the options above. Only respond with the letter."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio_path_prefix` | str | 空字符串（纯文本 prefill） |
| `img_path_prefix` | str | 空字符串（纯文本 prefill） |
| `cnt` | int | 递增计数器，紧接交错 prefill 之后 |
| `prompt` | str | 包含 question + options 的完整评测 prompt |

### 调用时机
所有图片/音频交错 prefill 完成后，作为**最后一次 prefill** 调用。`cnt` 值等于前面交错 prefill 的总次数。

### 需确认
- [ ] 当 `img_path_prefix` 和 `audio_path_prefix` 均为空时，服务端是否只做纯文本 tokenize + KV cache 追加
- [ ] 长文本 prompt 是否有长度限制

---

## 4. decode（生成回答）

### 请求
**POST** `/v1/stream/decode`

```json
{
    "stream": true,
    "round_idx": 0
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `stream` | bool | 始终为 `true`，服务端以 SSE 流式返回 |
| `round_idx` | int | 对话轮次索引，单轮评测固定为 `0` |

### SSE 响应格式
服务端返回 `text/event-stream`，每行格式：

```
data: {"content": "A", "stop": false}
data: {"content": "", "stop": true}
data: [DONE]
```

Python 端逐行解析，拼接所有 `content` 字段直到 `stop=true` 或 `[DONE]`。

### 调用时机
所有 prefill（图片 + 音频 + 文本）完成后调用。连接超时 300s，SSE 读超时 120s。

### 需确认
- [ ] `round_idx=0` 在单轮场景下是否正确
- [ ] SSE 输出的 `content` 是否为逐 token 输出
- [ ] 生成结束时是否同时返回 `stop=true` 事件和 `[DONE]` 标记

---

## 5. Thinking 模式

### Python 端行为
Daily-Omni 使用 `enable_thinking=False`，模板自动注入空的 thinking 块：
```
<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
```

### 需确认
- [ ] C++ 端 decode 时是否自动注入 `<think>\n\n</think>\n\n`，或者需要在 prompt 尾部手动拼接
- [ ] `<|tts_bos|>` token 是否正确处理

---

## 6. 后处理

### Python 端
```python
response = response.replace("<|tts_eos|>", "").strip()
```

### C++ 端
- [ ] 确认 decode 输出是否可能包含 `<|tts_eos|>` token（如果有，Python 端已处理）

---

## 完整单样本处理流程

```
1. POST /v1/stream/reset                    ← 清空 KV cache
2. POST /v1/stream/prefill  (图片, cnt=0)   ← frame_0 + skip_system_prompt
3. POST /v1/stream/prefill  (音频, cnt=1)   ← audio_seg_0
4. POST /v1/stream/prefill  (图片, cnt=2)   ← frame_1
5. POST /v1/stream/prefill  (音频, cnt=3)   ← audio_seg_1
   ...                                       ← 交替进行
N. POST /v1/stream/prefill  (文本, cnt=K)   ← 评测 prompt
N+1. POST /v1/stream/decode (SSE)           ← 生成回答
```

首次使用前需调用 `omni_init` 一次（每个 server 生命周期仅一次）。

---

## 优先级排序

| 优先级 | 事项 | 说明 |
|--------|------|------|
| **P0** | 音频 prefill 支持 | 无此功能则 Daily-Omni 无法测评 |
| **P0** | 交错 prefill | 必须确保 image/audio 交替 prefill 正确 |
| **P1** | max_slice_nums=0 | Daily-Omni 不分块（cpp 参数 0 代表不分块） |
| **P2** | thinking 模式注入 | 可能影响生成质量 |
| **P3** | repeat_penalty 调整 | 启动参数修改即可 |
