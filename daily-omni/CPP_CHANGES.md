# C++ 推理端需要确认/修改的事项

本文档记录 Daily-Omni 评测流水线中，C++ 推理端（llama-server / llama.cpp-omni）可能需要调整的部分。

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

## 2. 交错 prefill 支持

### Daily-Omni 的 content 结构
```
[frame_0, audio_seg_0, frame_1, audio_seg_1, ..., frame_N-1, audio_seg_N-1, text_prompt]
```

Python 端按此顺序交替调用 `prefill_image` 和 `prefill_audio`，cnt 从 0 递增。

### 需确认
- [ ] 服务端是否正确处理交替的 image/audio prefill（即 cnt=0 是图片，cnt=1 是音频，cnt=2 是图片...）
- [ ] 每次 prefill 的 `prompt="\n"` 是否会正确插入换行 token（对齐 Python `"\n".join`）
- [ ] 最终 decode 前的 KV cache 内容是否与 Python 端一致

## 3. max_slice_nums 参数

### 差异
- VideoMME: `max_slice_nums=0`
- Daily-Omni: `max_slice_nums=1`（视频场景不分块，但值不同）

### 需确认
- [ ] `max_slice_nums=1` 在 C++ 端的行为是否等同于 Python 端（不对图片进行分块处理）

## 4. repeat_penalty 参数

### 差异
- VideoMME: `repeat_penalty=1.02`
- Daily-Omni: `repeat_penalty=1.0`（即不使用 repeat penalty）

### 处理
服务端启动参数中 `--repeat-penalty 1.0`，在 `.env` 或启动命令中调整即可。

## 5. Thinking 模式

### Python 端行为
Daily-Omni 使用 `enable_thinking=False`，模板自动注入空的 thinking 块：
```
<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
```

### 需确认
- [ ] C++ 端 decode 时是否自动注入 `<think>\n\n</think>\n\n`，或者需要在 prompt 尾部手动拼接
- [ ] `<|tts_bos|>` token 是否正确处理

## 6. 后处理

### Python 端
```python
response = response.replace("<|tts_eos|>", "").strip()
```

### C++ 端
- [ ] 确认 decode 输出是否可能包含 `<|tts_eos|>` token（如果有，Python 端已处理）

---

## 优先级排序

| 优先级 | 事项 | 说明 |
|--------|------|------|
| **P0** | 音频 prefill 支持 | 无此功能则 Daily-Omni 无法测评 |
| **P0** | 交错 prefill | 必须确保 image/audio 交替 prefill 正确 |
| **P1** | max_slice_nums=1 | 影响图片处理方式 |
| **P2** | thinking 模式注入 | 可能影响生成质量 |
| **P3** | repeat_penalty 调整 | 启动参数修改即可 |
