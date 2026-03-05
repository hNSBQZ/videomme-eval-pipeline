# Omni Streaming Server API 配置指南

本文档说明 llama-server 中 omni streaming 相关 API 的参数配置，
以及在**单工、纯图片输入、纯文字输出**场景下的推荐设置与请求示例。

---

## 一、API 总览

| 端点 | 方法 | 用途 |
|---|---|---|
| `/v1/stream/omni_init` | POST | 初始化 omni 上下文（加载 ViT/Audio encoder，配置模式） |
| `/v1/stream/reset` | POST | 清空 KV cache，重置所有状态，为新一轮推理做准备 |
| `/v1/stream/prefill` | POST | 填充图片 embedding / 音频 embedding / 文本 prompt 到 KV cache |
| `/v1/stream/decode` | POST | 触发 LLM 生成，通过 SSE 流式返回文字 |

---

## 二、`/v1/stream/omni_init` — 初始化

### 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `media_type` | int | 是 | — | `1`=纯音频, `2`=omni（音频+视觉）。需要图片输入时**必须设为 `2`** |
| `use_tts` | bool | 否 | `true` | 是否启用 TTS 语音合成。**纯文字输出时设为 `false`** |
| `duplex_mode` | bool | 否 | `false` | 双工模式。**单工场景设为 `false`**（默认） |
| `n_predict` | int | 否 | `2048` | LLM 最大生成 token 数 |
| `model_dir` | string | 否 | `"./tools/omni/convert/gguf/"` | GGUF 模型目录 |
| `tts_bin_dir` | string | 否 | `model_dir + "token2wav-gguf"` | TTS token2wav 模型目录 |
| `tts_gpu_layers` | int | 否 | `99` | TTS 模型 GPU offload 层数 |
| `token2wav_device` | string | 否 | `"gpu:1"` | token2wav 使用的设备 |
| `output_dir` | string | 否 | `"./tools/omni/output"` | 输出目录 |
| `voice_audio` | string | 否 | `""` | 初始化时用于 voice cloning 的参考音频路径。不需要 TTS 时留空 |
| `vision_backend` | string | 否 | `"metal"` | 视觉编码器后端：`"metal"`(GPU) 或 `"coreml"`(ANE) |

### 纯图片 + 纯文字输出的推荐设置

```json
{
    "media_type": 2,
    "use_tts": false,
    "duplex_mode": false,
    "n_predict": 128
}
```

> **说明**：`media_type=2` 会加载 Vision encoder（ViT）和 Audio encoder。
> 虽然不使用音频，但 system prompt 初始化流程需要 audio encoder 来处理 ref_audio embedding。
> `use_tts=false` 不加载 TTS 模型，节省显存，decode 时直接输出纯文字。

---

## 三、`/v1/stream/reset` — 重置状态

### 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `duplex_mode` | bool | 否 | — | 如果提供，更新双工/单工模式 |

### 重置内容

调用后会：
1. 清空 LLM KV cache
2. 清空 TTS KV cache
3. 重置 `n_past = 0`
4. 重置 `system_prompt_initialized = false`（使下次 prefill 重新初始化 system prompt）
5. 重置其他状态变量（`break_event`, `speek_done` 等）

### 请求示例

```json
{}
```

> **注意**：每道独立的题目/每轮独立推理之前都应调用 reset，确保从干净状态开始。
> reset 不影响 ViT/Audio encoder 的状态（它们是无状态的前向推理，无需重置）。

---

## 四、`/v1/stream/prefill` — 填充 embedding / 文本

### 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `audio_path_prefix` | string | 是 | — | 音频文件路径。**无音频时传空字符串 `""`** |
| `img_path_prefix` | string | 否 | `""` | 图片文件路径。无图片时传空字符串 |
| `cnt` | int | 是 | — | 帧序号（从 0 开始），即 `stream_prefill` 的 `index` 参数 |
| `max_slice_nums` | int | 否 | `-1` | 图片切片数量控制（详见下方 HD 模式说明） |
| `prompt` | string | 否 | `""` | 文本 prompt，非空时在图片/音频 embedding 之后注入 KV cache |

### 关键行为

- **`cnt=0`（首次调用）**：触发 system prompt 初始化（voice clone prompt + ref_audio embedding + assistant prompt），然后处理图片/音频
- **`cnt>=1`（后续调用）**：直接处理图片/音频 embedding，跳过 system prompt
- **`prompt` 参数**：在图片和音频 embedding 之后、函数返回之前，将文本 tokenize 并 eval 到 KV cache（仅 sync 模式生效）

### 图片 HD 模式 — `max_slice_nums` 说明

| 值 | 含义 | 适用场景 |
|---|---|---|
| `-1`（默认） | 使用全局设置 | 一般情况 |
| `1` | **不切片**，只生成 overview chunk（~96 tokens） | 视频帧场景，每帧低分辨率足够 |
| `2` 或更大 | 高清切片，生成 1 个 overview + N 个 slice chunks | 需要高分辨率细节的单张图片 |

> **视频评测建议**：设 `max_slice_nums=1`（不切片），64 帧 × ~100 tokens/帧 ≈ 6400 tokens，ctx_size=8192 足够。
> 如果切片，每帧可能产生 300+ tokens，64 帧就会超出上下文限制。

---

## 五、`/v1/stream/decode` — 生成文字

### 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `stream` | bool | 否 | `true` | 是否 SSE 流式输出 |
| `debug_dir` | string | 否 | `"./"` | 调试输出目录 |
| `round_idx` | int | 否 | `-1` | 轮次索引（用于同步状态） |

### SSE 响应格式

每个 SSE event 的 data 为 JSON：

```json
{"content": "A", "stop": false, "is_listen": false, "end_of_turn": false}
```

结束时发送 `data: [DONE]`。

---

## 六、完整示例：单工 + 纯图片 + 纯文字输出

以 Video-MME 评测为例，对 1 个视频的 1 道题的完整请求序列：

### Step 0: 初始化（仅一次）

```bash
curl -X POST http://localhost:8080/v1/stream/omni_init \
  -H "Content-Type: application/json" \
  -d '{
    "media_type": 2,
    "use_tts": false,
    "duplex_mode": false,
    "n_predict": 128
  }'
```

### Step 1: 重置 KV cache（每道题前）

```bash
curl -X POST http://localhost:8080/v1/stream/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Step 2: 逐帧 prefill 图片（假设 3 帧）

```bash
# 帧 0（触发 system prompt 初始化）
curl -X POST http://localhost:8080/v1/stream/prefill \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path_prefix": "",
    "img_path_prefix": "/tmp/video001/frame_000.jpg",
    "cnt": 0,
    "max_slice_nums": 1
  }'

# 帧 1
curl -X POST http://localhost:8080/v1/stream/prefill \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path_prefix": "",
    "img_path_prefix": "/tmp/video001/frame_001.jpg",
    "cnt": 1,
    "max_slice_nums": 1
  }'

# 帧 2
curl -X POST http://localhost:8080/v1/stream/prefill \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path_prefix": "",
    "img_path_prefix": "/tmp/video001/frame_002.jpg",
    "cnt": 2,
    "max_slice_nums": 1
  }'
```

### Step 3: prefill 文本 prompt

```bash
curl -X POST http://localhost:8080/v1/stream/prefill \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path_prefix": "",
    "img_path_prefix": "",
    "cnt": 3,
    "prompt": "What is happening in the video?\nA. A cat is sleeping\nB. A dog is running\nC. A bird is flying\nD. A fish is swimming\nPlease select the correct answer from the options above. Only respond with the letter."
  }'
```

> `cnt` 接续帧序号（= 帧数量），`img_path_prefix` 为空（无图片），通过 `prompt` 注入文本。

### Step 4: decode 获取回答

```bash
curl -X POST http://localhost:8080/v1/stream/decode \
  -H "Content-Type: application/json" \
  -d '{"stream": true, "round_idx": 0}'
```

响应（SSE）：
```
data: {"content":"A","stop":false,"is_listen":false,"end_of_turn":false}
data: [DONE]
```

### 然后对下一道题，回到 Step 1 重新 reset。

---

## 七、Server 启动参数参考

解码策略通过 llama-server 命令行参数控制：

```bash
./build/bin/llama-server \
  --model ./tools/omni/convert/gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --temp 0 \
  --repeat-penalty 1.2
```

| 参数 | 说明 |
|---|---|
| `--temp 0` | 贪心解码（temperature=0），最接近 beam search 的单路近似 |
| `--repeat-penalty 1.2` | 重复惩罚，对齐 Python 版 `repetition_penalty=1.2` |
| `--ctx-size 8192` | 上下文窗口大小，64 帧 × ~100 tokens + prompt ≈ 6600 tokens |
