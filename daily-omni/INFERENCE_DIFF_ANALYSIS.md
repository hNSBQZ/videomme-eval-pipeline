# Daily-Omni 推理端差异分析：Server Pipeline vs Python evalkit vs omni-cli

对比三个推理路径在 Daily-Omni 评测上的差异。

- **对齐基准**：`~/Video-MME/evalkit` + `~/o45-py`（Python HuggingFace 推理）
- **Server Pipeline**：Python 处理音频图片 → HTTP 发送给 `llama-server` → 异步推理
- **omni-cli**：`~/llama.cpp-omni` 下的 `omni-cli.cpp --eval-daily-omni` → 同步推理

---

## 已确认与 Python 对齐的项（已排除）

以下差异项经验证后确认 Server Pipeline 已与 Python evalkit 一致：

| # | 差异项 | 结论 |
|---|--------|------|
| 1 | 采样参数 | ✅ Python 实际也是 `do_sample=True`（evalkit 传 `sampling=False` 但 o45-py 参数名是 `do_sample`，不生效）。两边均为 temp=0.7, top_p=0.8, top_k=100, repeat_penalty=1.02。已修复 `--top-p`/`--top-k` 传入 server。 |
| 2 | assistant prompt 格式 | ✅ Python tokenizer chat_template 在 `enable_thinking=False, use_tts_template=True` 时输出 `<think>\n\n</think>\n\n<\|tts_bos\|>`，与 Server Pipeline 的 `stream_decode` 一致。omni-cli 硬编码的 `<\|spk_bos\|><\|spk\|><\|spk_eos\|>` 反而跟 Python 不一致。 |
| 3 | Prompt 选项格式 | ✅ Python evalkit 的 `_build_options_prompt` 也是 `f"{key}. {choice}\n"` + `.rstrip()`，双前缀行为一致。 |
| 4 | 额外 `\n` token | ✅ Python `chat()` 中 `omni_mode=False`（evalkit 传 `omni_input` 但参数名是 `omni_mode`，不生效），走 `"\n".join(cur_msgs)`，每帧/音频间也有 `\n`。已修复模板前导 `\n`，消除媒体→文本边界的双 `\n`。 |
| 5 | Whisper KV Cache | ✅ 双方均正确处理（reset API / 手动 clear）。 |
| 6 | decode 路径 | ✅ **已排除**。详见下方排查结论。 |
| 7 | prefill 异步同步 | ✅ **已排除**。详见下方排查结论。 |

---

## 已排查并排除的差异项

### ~~差异 1~~（已排除）：decode 实现路径

**原假设**：omni-cli 使用 `text_only_decode`，Server Pipeline 走完整 TTS pipeline 的 `stream_decode`，可能引入 token 过滤或状态变化。

**排查结论：三者 decode 逻辑一致，无实质性差异。**

1. **omni-cli 和 server 调用的是同一个 `stream_decode` 函数**（omni-cli.cpp:195），不存在独立的 `text_only_decode`。

2. **`use_tts=false` 时 TTS 相关逻辑被完全旁路**：
   - TTS/T2W 线程不创建（omni.cpp:9110 `if ... ctx_omni->use_tts`）
   - LLMOut 不推送给 TTS 队列（omni.cpp:9493 `if ctx_omni->use_tts && ...`）

3. **采样路径与"干净文本生成"等价**：
   - `sample_with_hidden_and_token` 中，非双工模式下 logits 不被修改（`length_penalty` 默认 1.0，no-op）
   - 核心采样调用 `common_sampler_sample()` 与标准 llama.cpp 完全一致
   - `eval_id_with_hidden` 对比 `eval_id` 唯一区别是临时开启 `llama_set_embeddings(ctx, true)` 提取 hidden states，**不改变 forward pass 的计算或 logits**

4. **`is_valid_tts_token` 过滤只影响 TTS 数据收集**，不影响文本 response 拼接。所有 token 都会进入 `response += tmp_str`，特殊 token 由后续 post-processing 清理。

### ~~差异 2~~（已排除）：同步 vs 异步 prefill

**原假设**：异步模式增加了复杂度，prefill 执行顺序可能不一致。

**排查结论：异步同步逻辑正确，FIFO 保序，不影响推理结果。**

1. **FIFO 保序**：Python 客户端串行发 HTTP prefill 请求（request-response），server handler 持 `octx_mutex` 推入 FIFO 队列，LLM 线程按序消费。prefill 顺序严格为 `image_0 → audio_0 → image_1 → audio_1 → ... → text_prompt`。

2. **decode 正确等待所有 prefill 完成**：`stream_decode` 通过 `g_decode_cv.wait(lock, []{ return prefill_done; })` 阻塞，LLM 线程只在队列清空后才设置 `prefill_done = true`（omni.cpp:4444-4457）。

3. **互斥保护到位**：
   - server handler 持 `octx_mutex` 防止并发请求
   - 队列操作持 `llm_thread_info->mtx`
   - text_queue 由 `text_mtx` 保护

4. **两个理论瑕疵（不影响实际结果）**：
   - `need_speek` 在 `stream_decode` 中未持锁写入，但 `cv.notify_all()` 后的锁获取提供了足够的 memory barrier
   - `llm_thread_func` 分支1 释放锁后进入分支2 的 `queue.empty()` 无锁读取，但 Daily-Omni 场景下 decode 时无并发 prefill

---

## 剩余排查方向

以上排查确认 decode 路径和 async prefill 逻辑无问题。如果 Server Pipeline 与 Python 仍有分数差异，需从以下方向继续排查：

| # | 方向 | 说明 |
|---|------|------|
| 1 | 采样参数实际生效值 | 确认 temp/top_p/top_k/repeat_penalty 是否真正传入了 `ctx_sampler`（可在 `sample_with_hidden_and_token` 中打印 logits 分布验证） |
| 2 | 随机种子 | Python 和 C++ 的 RNG 初始状态是否一致，seed 是否固定 |
| 3 | 模型量化精度差异 | GGUF 量化格式 vs Python FP16/BF16 的数值偏差可能导致累积误差 |
| 4 | 视觉/音频编码器对齐 | C++ 侧的 vision/audio encoder 输出是否与 Python HF 版精确一致 |

---

## 验证步骤

1. 使用 `--limit 50` 在小样本上对比 Server Pipeline vs Python 的逐题输出
2. 针对上述剩余方向逐一排查
3. 目标：与 Python evalkit 结果一致（误差 <1%）
