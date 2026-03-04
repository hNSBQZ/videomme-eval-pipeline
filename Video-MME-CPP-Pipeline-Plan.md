# Video-MME CPP Benchmark Pipeline — 设计方案

## 一、目标

使用 llama.cpp-omni 的 server 模式，在 8 卡 GPU 上并行评测 Video-MME benchmark（900 个视频、2700 道题），
生成与 `output_test_template.json` 格式一致的结构化结果，供后续评分。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Python 主控脚本                            │
│  1. 读取 parquet 数据集（900 视频 × 3 题）                    │
│  2. 按 video_id 分组，均匀分配到 8 个 GPU worker              │
│  3. 启动 8 个 llama-server 进程（每个绑定一张 GPU）            │
│  4. 8 个线程池并发处理，每个 worker 串行处理分配的视频          │
│  5. 收集结果，合并输出 JSON                                   │
└─────────────────────────────────────────────────────────────┘
         │ HTTP API
         ▼
┌─────────────────────────────────────────────────────────────┐
│              llama-server (× 8, 每卡一个)                     │
│  /v1/stream/omni_init   — 初始化 omni 上下文                  │
│  /v1/stream/reset       — 清空 KV cache（每个视频前）          │
│  /v1/stream/prefill     — 填充图片帧 + 文本 prompt             │
│  /v1/stream/decode      — 生成文字答案                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、单个视频的评测流程

对每个视频，有 3 道题。由于 MiniCPM 评测时**每次只送一道题**，因此每个 video_id 循环 3 次：

```
for question in video.questions:
    1. POST /v1/stream/reset       → 清空 KV cache
    2. 视频帧 prefill（N 帧图片依次填充）
       for frame_path in frame_paths:
           POST /v1/stream/prefill {
               audio_path_prefix: "",     # 无音频
               img_path_prefix: frame_path,
               cnt: frame_index           # 帧序号
           }
    3. 文本 prompt prefill（问题 + 选项 + 指令）
       POST /v1/stream/prefill {
           audio_path_prefix: "",
           img_path_prefix: "",
           cnt: N,                        # 接续帧序号
           prompt: "问题\nA. ...\nB. ...\n..."  ← 需要新增此字段
       }
    4. POST /v1/stream/decode {stream: true} → SSE 收集文字回答
    5. 从回答中提取 A/B/C/D
```

---

## 四、视频预处理（Python 侧）

直接复用 evalkit 已有的 `encode_video()` 函数做帧采样，参数对齐：

| 参数 | 值 | 说明 |
|---|---|---|
| MAX_NUM_FRAMES | 64 | 最多 64 帧 |
| MAX_FPS | 1.0 | 每秒最多 1 帧 |

采样后将 PIL.Image 保存为临时 JPG 文件，传路径给 server 做 vision encoding。

---

## 五、需要修改/新增的 Server 端代码

### 5.1 prefill 支持"仅图片无音频"

当前 `stream_prefill` 要求 `audio_path_prefix`，但 videomme 场景只有图片帧没有音频。
需要确认：**当 `audio_path_prefix` 为空字符串时，`stream_prefill` 是否能正常跳过音频处理。**

查看 `omni.cpp:8875`，已有判断：
```cpp
if (aud_fname.length() > 0) { ... }  // 只在有音频时处理
```

所以传空字符串即可跳过音频。**无需修改。**

### 5.2 prefill 支持注入文本 prompt（关键新增）

当前 prefill API 只处理图片和音频的 embedding，没有注入纯文本 prompt 的能力。
videomme 需要在帧序列之后注入文本问题，格式为：

```
{question}
A. xxx
B. xxx
C. xxx
D. xxx
Please select the correct answer from the options above. Only respond with the letter.
```

**方案：在 `handle_stream_prefill_impl` 中新增 `prompt` 字段支持。**

当 `prompt` 非空时，在 prefill 尾部调用 `eval_string()` 将文本 tokenize 并填入 KV cache。

需要在 `stream_prefill()` 函数中增加一个 `prompt` 参数，在图片/音频 embedding 之后、return 之前追加文本 eval。

### 5.3 decode 仅输出文字（禁用 TTS）

decode 时 server 已支持 `use_tts=false`（在 `omni_init` 时设置）。
当 `use_tts=false` 时：
- 不启动 TTS/T2W 线程
- `stream_decode` 中走 `else` 分支：`std::string prompt = "<|im_end|>\n<|im_start|>assistant\n";`
  （不含 `<think>` 和 `<|tts_bos|>`）
- LLM 直接生成文字，通过 SSE 返回

**这正是我们需要的。只需在 `omni_init` 时设置 `use_tts: false` 即可。**

### 5.4 decode 控制生成长度

需要限制 `n_predict`（max_tokens）为 128，与 Python 评测一致。
`omni_init` 已支持 `n_predict` 参数。

### 5.5 解码策略对齐

**Python 版实际配置**（`sampling=False` 时，见 `modeling_minicpmo.py:696-700`）：

```python
generation_config = {
    "num_beams": 3,           # beam search, beam width = 3
    "repetition_penalty": 1.2,
}
```

这不是纯贪心，而是 **beam search (num_beams=3) + repetition_penalty=1.2**。

**llama.cpp 的限制**：`common_sampler` 不支持 beam search，只支持 greedy / top-k / top-p 等逐 token 采样。

**CPP 版对齐方案**：
- `temperature = 0`（greedy argmax，最接近 beam search 的单路近似）
- `repetition_penalty = 1.2`（与 Python 对齐）
- 通过 server 启动参数 `--temp 0 --repeat-penalty 1.2` 设置，
  或在 `omni_init` 请求中传递 sampling 配置

LLM sampler 在 `omni_init` 时通过 `common_sampler_init(model, params->sampling)` 创建（`omni.cpp:3575`），
`params->sampling` 来自 llama-server 的命令行参数。

**注意**：beam search → greedy 的降级可能导致少量答案差异，这是预期内的。后续可通过对比实验量化差异。

### 5.6 reset 的 system_prompt_initialized 重置

`/v1/stream/reset` 清空了 KV cache 和 `n_past`，但没有重置 `system_prompt_initialized`。
这意味着 reset 后再 prefill，不会重新初始化 system prompt（包含 ref_audio 的 voice clone prompt）。

对于 videomme 评测，我们**不需要音频**，但 system prompt 仍然需要在每轮重新初始化。

**方案：在 reset 中增加 `system_prompt_initialized = false` 的重置。**

### 5.7 prefill 支持跳过 system prompt（`skip_system_prompt`）（✅ 已实现）

**问题：** 当前 llama.cpp-omni 的 `stream_prefill(index=0)` 会**硬编码注入 voice clone system prompt**：

```
<|im_start|>system
模仿音频样本的音色并生成新的内容。
<|audio_start|>[ref_audio_embed]<|audio_end|>你的任务是用这种声音模式来当一个助手。...
<|im_end|>
<|im_start|>user
```

而 Python Video-MME 评测配置的 `system_prompt = ""`，即**不发送任何 system message**，token 序列直接从 `<|im_start|>user\n` 开始。

**方案（已实现）：** 在 `stream_prefill()` 中新增 `skip_system_prompt` 参数（默认 `false`）。

当 `skip_system_prompt=true` 且 `index=0` 时：
- 跳过整段 voice clone system prompt 初始化（不加载 ref_audio、不 eval voice_clone_prompt/assistant_prompt）
- 仅 eval `<|im_start|>user\n` 开始用户回合
- 标记 `system_prompt_initialized = true`，使后续帧进入正常处理分支

**修改文件：**
- `omni.h`：函数签名新增 `bool skip_system_prompt = false`
- `omni.cpp`：在原 system prompt 初始化逻辑前插入 `skip_system_prompt` 短路分支
- `server.cpp`：`handle_stream_prefill_impl` 解析请求中的 `skip_system_prompt` 字段并传递

**API 用法：**
```json
POST /v1/stream/prefill
{
    "audio_path_prefix": "",
    "img_path_prefix": "frame_0.jpg",
    "cnt": 0,
    "max_slice_nums": 1,
    "skip_system_prompt": true
}
```

**对齐效果（`skip_system_prompt=true`）：**
```
<|im_start|>user
<unit><image>[帧1 embedding]</image>
<unit><image>[帧2 embedding]</image>
...
{question}\n{options}\nPlease select...
<|im_end|>
<|im_start|>assistant
[模型生成]
```

与 Python 版 `system_prompt=""` 时的 token 序列完全一致。

---

## 六、Server 端代码修改总结

| 修改点 | 文件 | 说明 |
|---|---|---|
| ① prefill 支持文本 prompt | `omni.cpp` `stream_prefill()` | 新增 `prompt` 参数，在末尾 `eval_string` |
| ② server API 传递 prompt | `server.cpp` `handle_stream_prefill_impl` | 读取请求中的 `prompt` 字段并传递 |
| ③ reset 重置 system_prompt | `server.cpp` `handle_stream_reset_impl` | 追加 `system_prompt_initialized = false` |
| ④ 解码策略对齐 | server 启动参数 | `--temp 0 --repeat-penalty 1.2`（近似 Python beam search） |
| ⑤ prefill 跳过 system prompt | `omni.h` `omni.cpp` `server.cpp` | 新增 `skip_system_prompt` 参数，跳过 voice clone system prompt，仅 eval `<\|im_start\|>user\n`（✅ 已实现） |

---

## 七、Python Pipeline 设计

### 7.1 目录结构

```
Video-MME/
├── eval_cpp_pipeline.py          # 主控脚本
├── eval_cpp_worker.py            # 单 GPU worker 逻辑
├── eval_cpp_video_prep.py        # 视频预处理（帧采样 + 保存 JPG）
└── eval_cpp_config.py            # 配置（模型路径、GPU 数量等）
```

### 7.2 主控脚本流程 `eval_cpp_pipeline.py`

```python
def main():
    # 1. 加载 parquet 数据集
    dataset = load_parquet("Video-MME/videomme/test-00000-of-00001.parquet")
    
    # 2. 按 video_id 分组 → 900 个 video group
    video_groups = group_by_video_id(dataset)
    
    # 3. 将 900 个 video group 均匀分成 8 份
    chunks = split_into_n(video_groups, n=8)
    
    # 4. 启动 8 个 llama-server 进程
    servers = []
    for gpu_id in range(8):
        port = 8080 + gpu_id
        proc = start_server(gpu_id, port, model_path, ctx_size=8192,
                            extra_args="--temp 0 --repeat-penalty 1.2")
        servers.append((proc, port))
    
    # 5. 等待 server 就绪
    wait_all_servers_ready(servers)
    
    # 6. 每个 server 调用 omni_init（media_type=2, use_tts=false）
    for gpu_id, (proc, port) in enumerate(servers):
        omni_init(port, media_type=2, use_tts=False, n_predict=128)
    
    # 7. 8 线程并发处理
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for gpu_id, chunk in enumerate(chunks):
            port = 8080 + gpu_id
            futures.append(pool.submit(process_chunk, port, chunk, gpu_id))
        for f in futures:
            results.extend(f.result())
    
    # 8. 输出 JSON
    save_results(results, "output_videomme_cpp.json")
    
    # 9. 停止 server
    stop_all_servers(servers)
```

### 7.3 单视频处理流程 `process_video()`

```python
def process_video(port, video_id, video_path, questions):
    """处理单个视频的所有题目"""
    
    # 1. 视频帧采样（只做一次，3 道题共享）
    frames = encode_video(video_path, MAX_NUM_FRAMES=64, MAX_FPS=1.0)
    
    # 2. 保存帧为临时 JPG
    frame_paths = save_frames_as_jpg(frames, tmp_dir, video_id)
    
    results = []
    for q in questions:
        # 3. Reset KV cache
        http_post(port, "/v1/stream/reset")
        
        # 4. 逐帧 prefill 图片（max_slice_nums=1，不做图片切片）
        #    第一帧 (cnt=0) 传 skip_system_prompt=true，跳过 voice clone system prompt
        for idx, frame_path in enumerate(frame_paths):
            req = {
                "audio_path_prefix": "",
                "img_path_prefix": frame_path,
                "cnt": idx,
                "max_slice_nums": 1
            }
            if idx == 0:
                req["skip_system_prompt"] = True
            http_post(port, "/v1/stream/prefill", req)
        
        # 5. Prefill 文本 prompt
        prompt = build_prompt(q["question"], q["options"])
        http_post(port, "/v1/stream/prefill", {
            "audio_path_prefix": "",
            "img_path_prefix": "",
            "cnt": len(frame_paths),
            "prompt": prompt
        })
        
        # 6. Decode 获取回答
        response_text = http_post_sse(port, "/v1/stream/decode", {
            "stream": True,
            "round_idx": 0
        })
        
        # 7. 提取答案字母
        answer = extract_answer(response_text)
        
        results.append({
            "question_id": q["question_id"],
            "task_type": q["task_type"],
            "question": q["question"],
            "options": q["options"],
            "answer": q["answer"],
            "response": response_text
        })
    
    # 清理临时文件
    cleanup_frames(frame_paths)
    return results
```

### 7.4 Prompt 模板

与 Python 评测完全一致：

```python
def build_prompt(question, options):
    """构建 Video-MME prompt"""
    options_text = "\n".join(options)
    return f"{question}\n{options_text}\nPlease select the correct answer from the options above. Only respond with the letter."
```

注意：Python 版中 `{media}` 占位符在 content 列表中被帧序列替换，文本 prompt 放在帧序列之后。
在 CPP server 中，图片通过 prefill API 填充，文本通过 prompt 参数注入，效果等价。

---

## 八、关键对齐项检查清单

| 对齐项 | Python 版 | CPP 版（计划） | 状态 |
|---|---|---|---|
| System Prompt | `system_prompt=""` （不发送 system message） | prefill 传 `skip_system_prompt=true`，跳过 voice clone system prompt | ✅ 已实现 |
| 帧采样 | 64帧, 1fps, uniform_sample | 复用 Python encode_video() | ✅ 复用 |
| 图片切片 | max_slice_nums=1（不切片） | prefill 传 max_slice_nums=1 | ✅ API 已支持 |
| Prompt 模板（user） | `{media}\n{question}\n{options}\nPlease select...` | 完全一致 | ✅ |
| 解码方式 | sampling=False → beam search (num_beams=3, rep_penalty=1.2) | greedy (temp=0) + rep_penalty=1.2（llama.cpp 无 beam search） | ⚠️ 近似 |
| max_tokens | 128 | omni_init n_predict=128 | ✅ API 已支持 |
| 每题独立 | 每题独立推理 | 每题前 reset KV cache | ✅ |
| TTS | 无 | use_tts=false | ✅ |
| 输出格式 | output_test_template.json | 完全一致 | ✅ |

---

## 九、TODO List

### Phase 1: Server 端修改

- [ ] **T1.1** 在 `stream_prefill()` 中新增 `prompt` 参数，支持在帧 embedding 之后注入纯文本
- [ ] **T1.2** 在 `handle_stream_prefill_impl` 中解析 `prompt` 字段并传给 `stream_prefill()`
- [ ] **T1.3** 在 `handle_stream_reset_impl` 中追加 `system_prompt_initialized = false` 重置
- [ ] **T1.4** 配置解码策略：server 启动参数 `--temp 0 --repeat-penalty 1.2`（近似 Python 的 beam search num_beams=3 + repetition_penalty=1.2）
- [x] **T1.5** 在 `stream_prefill()` 中新增 `skip_system_prompt` 参数，跳过 voice clone system prompt（对齐 Python `system_prompt=""`）
- [ ] **T1.6** 编译验证 server 修改无编译错误

### Phase 2: Python Pipeline 基础框架

- [ ] **T2.1** 创建 `eval_cpp_pipeline.py` 主控脚本，实现数据集加载、分组、分配
- [ ] **T2.2** 实现 server 进程管理（启动、健康检查、停止）
- [ ] **T2.3** 实现视频帧采样与临时 JPG 保存逻辑（复用 evalkit encode_video）
- [ ] **T2.4** 实现 HTTP 客户端封装（prefill / decode / reset / omni_init）
- [ ] **T2.5** 实现 SSE 流式接收 decode 结果

### Phase 3: 评测逻辑

- [ ] **T3.1** 实现单视频处理流程（reset → 多帧 prefill → prompt prefill → decode → 解析答案）
- [ ] **T3.2** 实现 8 路并发处理（ThreadPoolExecutor）
- [ ] **T3.3** 实现进度监控与日志
- [ ] **T3.4** 实现结果结构化输出（对齐 output_test_template.json 格式）
- [ ] **T3.5** 实现答案提取逻辑（从模型输出中提取 A/B/C/D）

### Phase 4: 测试与验证

- [ ] **T4.1** 单卡单视频 e2e 测试（用 1 个视频验证完整流程）
- [ ] **T4.2** 多卡并行测试（2 卡，验证并行逻辑）
- [ ] **T4.3** 对比 Python 版和 CPP 版在相同输入下的输出差异
- [ ] **T4.4** 全量 900 视频评测并生成结果文件

---

## 十、风险与注意事项

1. **Context Size**: 64 帧图片 + prompt 的总 token 数可能很大（每帧约 96 token），需要 n_ctx >= 8192。
   如果 `max_slice_nums=1`（不切片），每帧是一个 overview chunk（96 token）+ 标记开销。
   估算：64 × ~100 + prompt ~200 ≈ 6600 token，8192 应该够用。

2. **内存/显存**: 每卡需要加载完整模型（LLM + Vision + Audio encoder），需确认 8 卡时显存足够。
   由于 `use_tts=false`，不加载 TTS 模型，可节省显存。

3. **System Prompt 与 Ref Audio**: 即使 `use_tts=false`，当前的 system prompt 初始化流程仍会尝试加载
   ref_audio。对于纯 vision 场景，可能需要简化 system prompt 初始化，或确保 ref_audio 文件存在但不影响结果。

4. **media_type**: 应设为 2（omni=audio+vision），因为需要 vision encoder。虽然不传音频，
   但 media_type=1 可能不加载 vision encoder。需要验证 media_type=2 + 无音频的组合是否正常。

5. **Prompt 注入位置**: 文本 prompt 需要在所有图片帧之后、decode 的 `<|im_end|>\n<|im_start|>assistant\n` 之前。
   这意味着最后一次 prefill 带 prompt 时，不应触发 system prompt 初始化（index 不为 0）。
