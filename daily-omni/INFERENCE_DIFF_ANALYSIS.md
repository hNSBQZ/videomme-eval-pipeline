# Daily-Omni 推理端差异分析：Server Pipeline vs omni-cli

对比两个推理路径在 Daily-Omni 评测上的差异，分析精度不对齐的原因。

- **Server Pipeline**：Python 处理音频图片 → HTTP 发送给 `llama-server` → 异步推理
- **omni-cli**：`~/llama.cpp-omni` 下的 `omni-cli.cpp --eval-daily-omni` → 同步推理

---

## 差异总览

| # | 差异项 | 严重程度 | Server Pipeline | omni-cli |
|---|--------|---------|----------------|----------|
| 1 | 采样参数 | **致命** | temp=0.7, repeat_penalty=1.02 | temp=0.0, top_k=1 (greedy) |
| 2 | assistant prompt 格式 | **高** | `<think>\n\n</think>\n\n<|tts_bos|>` | `<\|spk_bos\|><\|spk\|><\|spk_eos\|><\|tts_bos\|>` |
| 3 | decode 路径 | **高** | TTS pipeline 路径（hidden states 收集 + token 过滤） | 干净的 text_only_decode（逐 token sample） |
| 4 | Prompt 选项格式 | **中** | 双前缀 "A. A. xxx"（Python 额外加了 key 前缀） | 单前缀 "A. xxx"（直接拼接原始 choices） |
| 5 | 额外 `\n` token | **中** | 每帧/每段音频后注入 `prompt="\n"` | 无额外 `\n` 注入 |
| 6 | prefill 模式 | **中** | async=true（队列 + LLM 线程） | async=false（全同步串行） |
| 7 | Whisper KV Cache | **低** | reset API 中调用 omni_clear_audio_kv_cache ✓ | 每样本前手动 clear ✓ |

---

## 详细分析

### 差异 1（致命）：采样参数 — 随机采样 vs Greedy

**这是最可能导致精度差距的原因。**

omni-cli 在 `eval_daily_omni` 开始时显式覆盖采样参数为 greedy：

```cpp
// omni-cli.cpp L915-918
params.sampling.temp = 0.0f;
params.sampling.top_k = 1;
params.n_predict = 128;
if (ctx_omni->ctx_sampler) { common_sampler_free(ctx_omni->ctx_sampler); }
ctx_omni->ctx_sampler = common_sampler_init(ctx_omni->model, params.sampling);
```

Server Pipeline 的 `llama-server` 启动命令中传的是 `--temp 0.7 --repeat-penalty 1.02`（来自 `eval_cpp_config.py`），并且 **没有在 omni_init 或 decode 阶段覆盖为 greedy**。

> **影响**：MCQ 评测需要确定性输出，temp=0.7 的随机采样会导致每次运行结果不同，且选择概率分散，准确率自然低于 greedy decode。

### 差异 2（高）：assistant prompt 格式不同

omni-cli 的 `text_only_decode` 使用的 suffix：

```cpp
// omni-cli.cpp L213-215
std::string suffix = use_tts_template
    ? "<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"
    : "<|im_end|>\n<|im_start|>assistant\n";
```

评测时 `use_tts_template=true`，所以使用 `<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>` 格式。

Server Pipeline 的 `stream_decode` 使用的 prompt：

```cpp
// omni.cpp L9169
std::string prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>";
```

> **影响**：`<think>` 和 `<spk>` 系列 token 是不同的特殊标记，会改变模型进入不同的生成模式/概率分布，直接影响输出质量。

### 差异 3（高）：decode 实现路径完全不同

omni-cli 使用简洁的 `text_only_decode`：

```cpp
// omni-cli.cpp L219-227
for (int i = 0; i < max_tokens; ++i) {
    const char * tok = sample(ctx_omni->ctx_sampler, ctx_omni, &params, &ctx_omni->n_past);
    if (!tok) break;
    std::string piece(tok);
    if (piece == "</s>" || piece == "<|im_end|>" || ...) break;
    result += piece;
}
```

Server Pipeline 走的是完整 TTS pipeline 的 `stream_decode`（~600 行），包括：
- hidden states 收集（`llama_loop_with_hidden_and_token`）
- `is_valid_tts_token` 过滤
- step_size chunk 分块
- text_queue SSE 推送

即使 `use_tts=false`，这些逻辑也会被执行，可能引入非预期的 token 过滤或状态变化。

### 差异 4（中）：Prompt 选项双前缀

omni-cli 直接拼接原始 choices（已含 "A. xxx" 前缀）：

```cpp
options_str += c + "\n";  // 结果: "A. Read a book\n"
```

Python Pipeline 额外添加 key 前缀：

```python
options_prompt += f"{key}. {choice}\n"  # 结果: "A. A. Read a book\n"
```

> **影响**：选项格式不一致可能影响模型对选项的理解。

### 差异 5（中）：每帧/每段音频后注入额外 `\n`

Python HTTP client 在每次 `prefill_image` 和 `prefill_audio` 时传入 `prompt="\n"`：

```python
# eval_cpp_http_client.py L86, L111
frame_prompt: str = "\n"
audio_prompt: str = "\n"
```

Server 端收到后会 `eval_string` 这个换行符。omni-cli 的 `stream_prefill` 不传 prompt 参数，没有额外 `\n`。

> **影响**：多出的 `\n` token 会轻微改变 attention 分布和位置编码。

### 差异 6（中）：同步 vs 异步 prefill

- omni-cli：全程 `async=false`，所有 prefill 同步串行
- Server：`omni_init` 后 `async=true`，后续帧的 prefill 走 llm_thread queue

异步模式下 prefill 的实际执行顺序由 LLM 线程控制，理论上应当保持 FIFO，但增加了复杂度。

---

## 修复建议

### P0：修复采样参数（最高优先级）

在 `eval_cpp_config.py` 中为 Daily-Omni 评测使用 greedy decoding：

```python
# eval_cpp_config.py
TEMPERATURE = 0.0   # 改为 greedy
TOP_K = 1           # 改为 1
TOP_P = 1.0         # 不限制
REPEAT_PENALTY = 1.0  # 不惩罚
```

同时确保 `llama-server` 启动参数传递正确的温度值。

> **注意**：即使 server 启动时设了 `--temp 0.7`，如果 `omni_init` 时不覆盖 sampler 参数，整个 session 内都是 temp=0.7。需要确认 server 端是否有覆盖 sampler 的机制，如果没有，需要在 server 启动参数中直接改为 `--temp 0`。

### P1：对齐 assistant prompt 格式

将 `stream_decode` 中的 assistant prompt 改为和 omni-cli 的 `text_only_decode` 一致：

**方案 A**（不用 TTS template）：直接使用 `<|im_end|>\n<|im_start|>assistant\n`

**方案 B**（用 TTS template 对齐 omni-cli）：使用 `<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>`

需要确认 omni-cli 使用哪个格式精度更好。

### P1：简化 decode 路径

为评测场景增加一个 text-only decode 模式，跳过 TTS hidden states 收集和 token 过滤逻辑，与 omni-cli 的 `text_only_decode` 对齐。

### P2：修复 Prompt 双前缀

```python
# eval_cpp_pipeline.py build_prompt()
# 改为直接拼接（不添加 key 前缀），对齐 omni-cli：
for choice in choices:
    options_prompt += f"{choice}\n"
```

### P2：移除额外 `\n` prompt 注入

```python
# eval_cpp_http_client.py
# prefill_image / prefill_audio 的 prompt 改为空字符串：
frame_prompt: str = ""
audio_prompt: str = ""
```

### P3：评测时使用同步模式

在 `omni_init` 后、评测开始前，将 `async` 设为 `false`（可通过新增 API 或在 server 端添加配置项实现）。

---

## 验证步骤

1. 先只修复 P0（采样参数），对比一轮结果
2. 如果仍有差距，逐步修复 P1/P2
3. 使用 `--limit 50` 在小样本上快速验证每步修复的效果
4. 目标：与 omni-cli 的 `--eval-daily-omni` 结果一致（误差 <1%）
