# Daily-Omni 评测 Python 实现对齐文档（C++ 移植参考）

> 本文档基于 evalkit 中 Daily-Omni 数据集的完整评测流程，从视频帧采样、音频采样、提示词模板、推理参数、数据集格式、评测流水线、评估指标等维度进行深度剖析，供 C++ 评估架构开发时对齐使用。

---

## 目录

1. [视频帧采样逻辑](#1-视频帧采样逻辑)
2. [音频采样逻辑](#2-音频采样逻辑)
3. [音视频交错（Interleave）逻辑](#3-音视频交错interleave逻辑)
4. [提示词模板](#4-提示词模板)
5. [推理/采样参数](#5-推理采样参数)
6. [数据集格式](#6-数据集格式)
7. [评测流水线（Pipeline）](#7-评测流水线pipeline)
8. [评估指标与评分逻辑](#8-评估指标与评分逻辑)
9. [OOM 重试机制](#9-oom-重试机制)
10. [多卡并行策略](#10-多卡并行策略)
11. [C++ 移植注意事项](#11-c-移植注意事项)

---

## 1. 视频帧采样逻辑

### 1.1 核心参数

| 参数 | Daily-Omni 值 | 含义 |
|------|-------------|------|
| `max_frames` | **64** | 最大采样帧数上限 |
| `max_fps` | **1.0** | 采样率上限（帧/秒） |

### 1.2 采样算法 `_sample_video_frame_indices`

**源码位置**: `o_e_Kit/utils/utils.py:49-86`

采样策略根据视频时长分两种情况：

#### 情况 A：长视频（`duration > max_frames`，即时长 > 64 秒）

```
step = 0.1  # 0.1 秒粒度
num_steps = int(duration / step)
timestamps = [round(i * 0.1, 1) for i in range(num_steps)]   # [0.0, 0.1, 0.2, ..., duration)
frame_idx  = [min(int(ts * avg_fps), total_frames - 1) for ts in timestamps]

# 如果候选帧数超过 max_frames，则均匀下采样到 max_frames
if len(frame_idx) > max_frames:
    frame_idx  = uniform_sample(frame_idx, max_frames)
    timestamps = uniform_sample(timestamps, max_frames)
```

#### 情况 B：短视频（`duration <= max_frames`，即时长 <= 64 秒）

```
int_duration = int(duration)
frame_idx  = [int(i * avg_fps) for i in range(int_duration)]   # 每秒取 1 帧
timestamps = [float(i) for i in range(int_duration)]            # [0.0, 1.0, 2.0, ...]
```

#### `uniform_sample` 均匀采样函数

```python
def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]
```

逻辑：将原列表均匀分为 `n` 段，每段取中间位置的元素。

### 1.3 视频解码后端优先级

按以下顺序尝试，第一个成功的即为最终结果：

| 优先级 | 后端 | 库 | 说明 |
|--------|------|----|------|
| 1 | **decord** | `decord.VideoReader` | 主后端，基于 FFmpeg 的 Python 绑定，支持随机帧访问 |
| 2 | **FFmpeg CLI** | `subprocess` 调用 `ffmpeg` | 通过 `-vf fps={max_fps} -vframes {max_frames}` 抽帧到临时 JPEG |
| 3 | **torchvision** | `torchvision.io.read_video` | 读取全部帧后 `torch.linspace` 均匀采样 |

### 1.4 帧格式

- 所有后端最终输出 `PIL.Image.Image` 对象（RGB 模式）
- decord: `vr.get_batch(indices).asnumpy()` → `Image.fromarray(v.astype("uint8")).convert("RGB")`
- FFmpeg: JPEG 文件 → `Image.open().convert("RGB")`
- torchvision: Tensor `(T,C,H,W)` → numpy → `Image.fromarray().convert("RGB")`

### 1.5 C++ 对齐要点

```
C++ 需复现：
1. 使用 FFmpeg C API（或 decord C++ 接口）获取 avg_fps 和 total_frames
2. 实现 _sample_video_frame_indices 的两分支逻辑（长视频 vs 短视频）
3. 实现 uniform_sample（均匀采样到 max_frames）
4. seek 到目标帧并解码为 RGB 图像
5. 输出格式：每帧为 HxWx3 的 uint8 RGB 数组
```

---

## 2. 音频采样逻辑

### 2.1 核心参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `audio_sr` | **16000** (16kHz) | 目标采样率 |
| `speed` | 1.0（默认） | 变速倍数，OOM 重试时递增 |
| `trim_end` | 0.0（默认） | 截断末尾秒数，OOM 重试时递增 |

### 2.2 音频加载 `load_audio`

**源码位置**: `o_e_Kit/utils/utils.py:415-445`

```python
def load_audio(wav_path, sr=16000, speed=1.0, trim_end=0.0):
    # 1. 使用 mmap 加速大文件读取，回退到普通加载
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        waveform = librosa.load(m, sr=sr)[0]   # 重采样到 16kHz，返回 float32 numpy array

    # 2. 变速处理（OOM 重试时 speed > 1.0）
    if speed > 1.0:
        waveform = speedup_audio(waveform, sr, speed)

    # 3. 截断处理（OOM 重试时 trim_end > 0）
    if trim_end > 0:
        waveform = trim_audio_segment(waveform, sr, trim_end)
    
    return waveform
```

### 2.3 音频查找规则

`load_av=true` 时，`_load_waveform_and_duration`（及基于它的 `load_video_and_audio` / 音视频交错）按 **优先级** 解析音频文件：

**1. 显式 `audio_path`（优先）**

若调用方传入且文件存在，直接使用该路径。Daily-Omni 的 `build_content` 会把数据集 JSONL 里的 `WavPath`（即 `paths['audio_path']`）传给交错逻辑，例如：

```
/data/Videos/{video_id}/{video_id}_audio.wav   ← Daily-Omni 实际落盘名称
```

**2. 从视频路径派生（回退、兼容其它数据集）**

未传 `audio_path` 或路径不存在时，去掉视频扩展名后依次尝试 `.wav`、`.mp3`、`.m4a`、`.flac`：

```
视频路径: /data/Videos/{video_id}/{video_id}_video.mp4
候选:     {video_id}_video.wav → .mp3 → .m4a → .flac
```

若数据集只提供 `{video_id}_audio.wav` 而不存在 `{video_id}_video.*` 音频，**必须**走上面的显式 `audio_path`，否则交错会拿不到波形、退化为仅视频帧（见 `docs/fix-audio-interleave.md`）。

**C++ 对齐**：与 Python 一致——交错/对齐音频时，若有独立音轨字段（如 `WavPath`），应优先用该路径；再回退到「与 mp4 同 stem」的侧车文件。

### 2.4 音频变速 `speedup_audio`

```
优先级 1: pyrubberband（保持音高的高质量时域拉伸）
优先级 2: librosa.resample（改变音高，但足够用于 OOM 场景）
```

### 2.5 音频截断 `trim_audio_segment`

```python
def trim_audio_segment(waveform, sr, trim_end):
    samples_to_trim = int(trim_end * sr)
    min_samples = int(0.1 * sr)  # 至少保留 0.1s
    if len(waveform) - samples_to_trim >= min_samples:
        return waveform[:-samples_to_trim]
    elif len(waveform) > min_samples:
        return waveform[:min_samples]
    return waveform
```

### 2.6 C++ 对齐要点

```
C++ 需复现：
1. 使用 FFmpeg C API 或 libsndfile 加载音频并重采样到 16kHz mono float32
2. 音频输出格式：一维 float32 数组，值域通常 [-1.0, 1.0]
3. 变速：可使用 librubberband C 库（保持音高拉伸）
4. 截断：简单的数组切片，保留至少 0.1s (1600 samples)
```

---

## 3. 音视频交错（Interleave）逻辑

### 3.1 核心参数

| 参数 | Daily-Omni 值 | 含义 |
|------|-------------|------|
| `interleave_fps` | **1.0** | 交错频率，>0 时启用交错模式 |
| `load_av` | **true** | 是否加载音频 |

### 3.2 交错流程 `load_video_and_audio_interleaved`

**源码位置**: `o_e_Kit/utils/utils.py:711-777`

```
输入: video_path, max_frames=64, max_fps=1.0, audio_sr=16000

步骤:
1. get_video_frame_audio_segments():
   a. 解码视频帧 → [frame_0, frame_1, ..., frame_N-1]，得到 timestamps [t_0, t_1, ..., t_N-1]
   b. 加载完整音频波形
   c. 按时间戳切分音频:
      - segment_i = audio[t_i * sr : t_{i+1} * sr]  (i < N-1)
      - segment_last = audio[t_{N-1} * sr : audio_end]
   d. 对齐: num_pairs = min(len(frames), len(audio_segments))

2. 可选: 对每段音频进行变速/截断（OOM 重试）

3. 构建交错列表:
   media = [frame_0, audio_seg_0, frame_1, audio_seg_1, ..., frame_N-1, audio_seg_N-1]
```

### 3.3 最终 content 结构

```
content = [
    PIL.Image,     # frame_0
    np.ndarray,    # audio_segment_0 (float32, 16kHz)
    PIL.Image,     # frame_1
    np.ndarray,    # audio_segment_1
    ...,
    PIL.Image,     # frame_N-1
    np.ndarray,    # audio_segment_N-1
    str            # 文本 prompt
]
```

### 3.4 音频切分详细逻辑

```python
for i, start_time in enumerate(timestamps):
    if i < len(timestamps) - 1:
        end_time = timestamps[i + 1]
    else:
        end_time = audio_duration  # 最后一段延伸到音频结尾

    start_sample = max(0, int(start_time * audio_sr))
    end_sample   = max(start_sample, int(end_time * audio_sr))
    segment = waveform[start_sample:end_sample]
```

### 3.5 C++ 对齐要点

```
C++ 需复现：
1. 视频帧采样得到帧列表和时间戳列表
2. 加载完整音频波形
3. 按时间戳切分音频为与帧一一对应的段
4. 构建交错的 media 序列: [frame, audio_seg, frame, audio_seg, ...]
5. 最后追加文本 prompt
```

---

## 4. 提示词模板

### 4.1 Daily-Omni MCQ 模板

**配置来源**: `o_e_Kit/configs/omni_generation_configs_nosys_interleave.json`

```
{media}
Carefully read the following question and select the letter corresponding to the correct answer.Highlight the applicable choices without giving explanations.
{question}
Options:
{options}
Please select the correct answer from the options above. Only respond with the letter.
```

| 占位符 | 替换内容 | 处理方式 |
|--------|---------|---------|
| `{media}` | 音视频交错内容 | 在 `build_content` 中被移除（`.replace("{media}", "")`），媒体内容放在 content 列表前面 |
| `{question}` | 问题文本 | 直接字符串替换 |
| `{options}` | 选项文本 | `_build_options_prompt` 格式化 |

### 4.2 选项格式化 `_build_options_prompt`

```python
def _build_options_prompt(self, choices: list) -> str:
    KEYS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    options_prompt = ''
    for key, choice in zip(KEYS[:len(choices)], choices):
        options_prompt += f'{key}. {choice}\n'
    return options_prompt
```

**注意**: choices 数组中的内容已经**不包含**字母前缀（如 `"选项内容"`），函数自行添加 `A.`, `B.` 前缀。但实际数据中 choices 字段格式为 `["A. xxx", "B. xxx", ...]`，即**已包含**字母前缀，所以最终输出可能出现 `A. A. xxx` 的重复。这是当前 Python 实现的一个需要注意的行为。

### 4.3 System Prompt

Daily-Omni 的 `system_prompt` 为**空字符串**（`""`），不添加 system 消息。

### 4.4 消息结构

```python
msgs = [
    {
        "role": "user",
        "content": [
            PIL.Image,     # frame_0
            np.ndarray,    # audio_seg_0
            PIL.Image,     # frame_1
            np.ndarray,    # audio_seg_1
            ...,
            str            # 最终 prompt 文本（已去除 {media} 占位符）
        ]
    }
]
```

### 4.5 content → chat template → token IDs（完整链路）

`_generate_chat` 把 §4.4 的 `msgs` 传给模型的 `chat()` 方法（`modeling_minicpmo.py`），经过以下三步转为最终的 token 序列：

#### 步骤 1：content 列表 → 占位符文本

`chat()` 遍历 `msg["content"]`，按类型替换：

| Python 类型 | 替换为 |
|------------|--------|
| `PIL.Image` | `<image>./</image>` |
| `np.ndarray` | `<audio>./</audio>` |
| `str` | 原样保留 |

替换后用 **`"\n".join(cur_msgs)`** 拼接为一个字符串。

> **注意**：evalkit 传递的参数名为 `omni_input=True`，但模型 `chat()` 的形参名为 `omni_mode`（默认 `False`）。
> 由于名称不匹配，`omni_input` 被 `**kwargs` 吞掉，`omni_mode` 始终为 `False`，
> 因此**永远走 `"\n".join` 分支**（而非 `"".join`）。
> 这不是 bug，前人测评也是这样跑的，C++ 对齐时需保持一致。

拼接结果示例（30 帧交错，`\n` 分隔）：

```
<image>./</image>\n<audio>./</audio>\n<image>./</image>\n<audio>./</audio>\n...\n<image>./</image>\n<audio>./</audio>\nCarefully read the following question...
```

#### 步骤 2：apply_chat_template → chatml 格式

套用 Jinja2 模板（`tokenizer_config.json` 中的 `chat_template`），生成 chatml 格式字符串：

```
<|im_start|>user\n{步骤1的拼接文本}<|im_end|>\n<|im_start|>assistant\n<|tts_bos|>
```

- 无 system prompt（Daily-Omni 配置 `system_prompt: ""`）
- `use_tts_template=True` → 生成提示末尾追加 `<|tts_bos|>`

#### 步骤 3：processor 替换占位符 → token IDs

`processing_minicpmo.py` 的 `_convert_to_tensors` 对步骤 2 的文本做最终替换：

| 占位符 | 替换为 | 说明 |
|--------|--------|------|
| `<image>./</image>` | `<image><unk>×64</image>` | `image_feature_size=64`；Daily-Omni `use_image_id=False`，不加 `<image_id>` 前缀 |
| `<audio>./</audio>` | `<\|audio_start\|><unk>×N<\|audio_end\|>` | N 取决于该段音频波形长度和 `audio_pool_step` |

替换后的各片段用 `"".join` 拼接（**此处无额外分隔符**），再 tokenize 得到 `input_ids`。

最终 token 序列结构（30 帧交错）：

```
<|im_start|> user \n
<image> <unk>×64 </image> \n      ← frame_0
<|audio_start|> <unk>×N₀ <|audio_end|> \n   ← audio_seg_0
<image> <unk>×64 </image> \n      ← frame_1
<|audio_start|> <unk>×N₁ <|audio_end|> \n   ← audio_seg_1
...
<image> <unk>×64 </image> \n      ← frame_29
<|audio_start|> <unk>×N₂₉ <|audio_end|> \n  ← audio_seg_29
Carefully read the following question...
<|im_end|> \n
<|im_start|> assistant \n
<think> \n \n </think> \n \n      ← 空 thinking 块（enable_thinking=False 时自动注入，见 §5.2.1）
<|tts_bos|>
```

> 其中 `\n` 是步骤 1 中 `"\n".join` 引入的，在 tokenize 后对应 `\n` 的 token ID。

### 4.6 最终 prompt 文本示例

假设 question="What is happening in the video?"，choices=["The cat is sleeping", "The dog is running", "A bird is singing", "People are talking"]：

```
Carefully read the following question and select the letter corresponding to the correct answer.Highlight the applicable choices without giving explanations.
What is happening in the video?
Options:
A. The cat is sleeping
B. The dog is running
C. A bird is singing
D. People are talking
Please select the correct answer from the options above. Only respond with the letter.
```

### 4.7 C++ 对齐要点

```
C++ 需复现：
1. 精确复现模板字符串（注意标点符号、换行符、空格）
2. {media} 占位符移除后的 prompt 放在 content 末尾
3. 选项格式化: "A. {choice}\n"
4. 无 system prompt
5. 交错元素之间用 \n 连接（对应 Python 的 "\n".join，非 "".join）
6. chatml 外壳: <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n<|tts_bos|>
7. 图片占位: <image><unk>×64</image>（不加 <image_id>，因为 use_image_id=False）
8. 音频占位: <|audio_start|><unk>×N<|audio_end|>（N 取决于音频长度和 audio_pool_step）
```

---

## 5. 推理/采样参数

### 5.1 生成参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `sampling` | **False** | **贪心解码**，不使用随机采样 |
| `max_new_tokens` | **128** | 最大生成 token 数 |
| `temperature` | N/A | 贪心模式下不适用 |
| `top_p` | N/A | 贪心模式下不适用 |
| `top_k` | N/A | 贪心模式下不适用 |

### 5.2 模型特殊参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_tts_template` | **True** | 使用 TTS 模板（模型内部格式） |
| `use_image_id` | **False** | 视频场景下不使用 image_id |
| `max_slice_nums` | **1** | 视频场景下图片不分块（纯图片时为 9） |
| `omni_input` | **True** | 当同时有音频和视觉且 `interleave_fps > 0` 时为 True。注意：evalkit 传 `omni_input`，但模型 `chat()` 形参为 `omni_mode`，名称不匹配导致实际未生效（见 §4.5） |
| `merge_audio_from_same_content` | **True** | 合并同一 content 下的音频片段 |
| `enable_thinking` | **False**（默认值） | 见下方 §5.2.1 详细说明 |

#### 5.2.1 Thinking 模式说明

evalkit 调用 `model.chat()` 时 **未传 `enable_thinking`**，该参数默认值为 `False`。

`enable_thinking=False` 在两个层面产生影响：

**1. chat_template（Jinja2）**

```jinja2
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

当 `enable_thinking=False` 时，模板会在 `assistant` 回复起始处**自动注入空的 thinking 块** `<think>\n\n</think>\n\n`，
相当于跳过思考、直接进入回答。

**2. streaming 路径的 `think_str`**（仅在 o45-py 基于 Qwen3 LLM 时）

```python
self.think_str = "<think>\n\n</think>\n\n"  # Qwen3ForCausalLM 时设置

# streaming generate 中:
bos_input = "".join([
    "<|im_end|>\n<|im_start|>assistant\n",
    "" if enable_thinking else self.think_str,   # enable_thinking=False → 注入空 think 块
    "<|tts_bos|>" if use_tts_template else "",
])
```

**结论**：Daily-Omni 测评 **不进行 thinking**。模型收到的 prompt 末尾结构为：

```
...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
```

即空的 `<think></think>` 块 + `<|tts_bos|>`，模型直接输出答案（如 `B`），不产生推理链。

**C++ 对齐**：生成 prompt 时需在 `assistant\n` 后注入 `<think>\n\n</think>\n\n`，再拼 `<|tts_bos|>`。

### 5.3 模型精度

- 权重精度: **BFloat16** (`torch.bfloat16`)
- 推理模式: `model.eval()` + 无梯度

### 5.4 随机种子

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

### 5.5 模型配置

**文件**: `configs/model_config/minicpm4o_3b_batch.json`

```json
{
    "audio_pool_step": 5,
    "audio_chunk_length": -1,
    "init_tts": false
}
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `audio_pool_step` | 5 | 音频特征池化步长 |
| `audio_chunk_length` | -1 | 音频分块长度，-1 表示不分块 |
| `init_tts` | false | 不初始化 TTS 模块 |

### 5.6 C++ 对齐要点

```
C++ 需复现：
1. 贪心解码 (argmax)，无需实现 temperature/top_p/top_k
2. max_new_tokens = 128
3. BFloat16 推理精度
4. 固定随机种子 0
5. 后处理：去除 "<|tts_eos|>" 标记并 strip
6. 不启用 thinking：在 assistant 头部注入空 think 块 <think>\n\n</think>\n\n（见 §5.2.1）
```

---

## 6. 数据集格式

### 6.1 原始格式（qa.json）

Daily-Omni 原始数据为 JSON 数组：

```json
[
    {
        "video_id": "abc123",
        "Question": "What is the person doing in the video?",
        "Choice": ["Cooking", "Reading", "Sleeping", "Dancing"],
        "Answer": "A",
        "Type": "Event Sequence",
        "content_parent_category": "Lifestyle",
        "content_fine_category": "Cooking",
        "video_category": "Howto & Style",
        "video_duration": "30s"
    },
    ...
]
```

### 6.2 转换后格式（daily_omni.jsonl）

通过 `scripts/convert_daily_omni.py` 转换为 JSONL（每行一条 JSON）：

```json
{
    "dataset_type": "mcq",
    "dataset_name": "daily_omni",
    "question": "What is the person doing in the video?",
    "choices": ["Cooking", "Reading", "Sleeping", "Dancing"],
    "gt_answer": "A",
    "WavPath": "Videos/abc123/abc123_audio.wav",
    "VideoPath": "Videos/abc123/abc123_video.mp4",
    "qa_type": "Event Sequence",
    "content_parent_category": "Lifestyle",
    "content_fine_category": "Cooking",
    "video_category": "Howto & Style",
    "video_duration": "30s",
    "video_id": "abc123"
}
```

### 6.3 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dataset_type` | string | 是 | 固定为 `"mcq"` |
| `dataset_name` | string | 是 | 固定为 `"daily_omni"` |
| `question` | string | 是 | 问题文本 |
| `choices` | string[] | 是 | 选项内容列表 |
| `gt_answer` | string | 是 | 正确答案字母，如 `"A"`, `"B"` |
| `WavPath` | string | 是 | 音频相对路径（相对于 `data_prefix_dir`） |
| `VideoPath` | string | 是 | 视频相对路径（相对于 `data_prefix_dir`） |
| `qa_type` | string | 否 | 问题类型（分组字段） |
| `content_parent_category` | string | 否 | 内容大类（分组字段） |
| `content_fine_category` | string | 否 | 内容细类 |
| `video_category` | string | 否 | 视频类别（分组字段） |
| `video_duration` | string | 否 | 视频时长标签（分组字段） |
| `video_id` | string | 否 | 视频 ID |

### 6.4 文件目录结构

```
daily-omni/
├── daily_omni.jsonl          # 转换后的标注文件
├── qa.json                   # 原始标注（可选保留）
└── Videos/
    ├── {video_id_1}/
    │   ├── {video_id_1}_video.mp4
    │   └── {video_id_1}_audio.wav
    ├── {video_id_2}/
    │   ├── {video_id_2}_video.mp4
    │   └── {video_id_2}_audio.wav
    └── ...
```

### 6.5 数据集统计

- 总样本数: **1197** 条
- 任务类型: MCQ（选择题）
- 选项数量: 通常 4 个（A/B/C/D）

### 6.6 C++ 对齐要点

```
C++ 需复现：
1. JSONL 逐行解析（每行一个 JSON 对象）
2. 必要字段: question, choices, gt_answer, WavPath, VideoPath
3. 路径拼接: data_prefix_dir + WavPath / VideoPath
4. 元数据字段用于分组评估统计
```

---

## 7. 评测流水线（Pipeline）

### 7.1 整体流程

```
eval_main.py
  ├─ parse_args()                     # 解析命令行参数
  ├─ load_model()                     # 加载 MiniCPM_o_ou 模型
  └─ run_all_evaluations()
       └─ evaluate_omni_datasets()
            ├─ load_dataset("daily_omni")
            │   └─ OmniEvalDataset(annotation_path, data_prefix_dir)
            │       └─ _load_annotations()  # 读 JSONL，提取 paths + annotation
            │
            ├─ infer_and_evaluate()
            │   ├─ create_dataloader(dataset)
            │   ├─ run_inference()
            │   │   └─ for batch in dataloader:
            │   │       └─ model.generate_chat(paths, items, "daily_omni")
            │   │           └─ _generate_chat()
            │   │               ├─ build_messages()
            │   │               │   ├─ get_generation_config("daily_omni")
            │   │               │   ├─ 模板替换 {question}, {options}
            │   │               │   └─ build_content()
            │   │               │       └─ load_video_and_audio_interleaved()
            │   │               │           ├─ get_video_frame_audio_segments()
            │   │               │           │   ├─ _load_video_frames_with_decord()
            │   │               │           │   ├─ _load_waveform_and_duration()
            │   │               │           │   └─ 按时间戳切分音频
            │   │               │           └─ [frame, audio_seg, frame, audio_seg, ..., prompt]
            │   │               │
            │   │               └─ model.chat(msgs, sampling=False, max_new_tokens=128, ...)
            │   │
            │   └─ save_predictions()   # → results/{model_name}/{time}/daily_omni.json
            │
            └─ evaluate_dataset("daily_omni", answer_file_path)
                └─ OmniMCQEvaluator(dataset_name="daily_omni")
                    └─ MQAEvaluator.evaluate()
```

### 7.2 数据加载阶段

```python
# OmniEvalDataset.__getitem__ 返回三元组:
(idx, paths, annotation)

# paths 示例:
{
    "video_path": "/cache/.../Videos/abc123/abc123_video.mp4",
    "audio_path": "/cache/.../Videos/abc123/abc123_audio.wav"
}

# annotation 示例:
{
    "dataset_type": "mcq",
    "dataset_name": "daily_omni",
    "question": "...",
    "choices": ["A. ...", "B. ...", ...],
    "gt_answer": "A",
    "qa_type": "Event Sequence",
    ...
}
```

### 7.3 推理输出格式

```json
{
    "predictions": [
        {
            "prediction": "A",
            "annotation": {
                "question": "...",
                "choices": [...],
                "gt_answer": "A",
                "qa_type": "..."
            },
            "audio_speed": 1.0,
            "audio_trim_end": 0.0
        },
        ...
    ],
    "job_id": "...",
    "dataset_name": "daily_omni"
}
```

### 7.4 输出后处理

```python
response = response.replace("<|tts_eos|>", "").strip()
```

### 7.5 C++ 对齐要点

```
C++ 需复现的完整 pipeline：
1. 读取 JSONL → 构建 (paths, annotation) 列表
2. 对每个样本：
   a. 加载视频帧 (max_frames=64, max_fps=1.0)
   b. 加载音频并按帧时间戳切分 (sr=16000)
   c. 构建交错 media 序列
   d. 拼接 prompt 文本
   e. 送入模型推理 (greedy, max_tokens=128)
   f. 后处理 response
3. 保存预测结果为 JSON
4. 运行评估器计算 accuracy
```

---

## 8. 评估指标与评分逻辑

### 8.1 评估器

**类**: `OmniMCQEvaluator`（继承自 `MQAEvaluator`）

评分流程：
1. **规则匹配**（Rule-based）：尝试从模型输出中提取答案字母
2. **Sentence Transformer**（可选）：语义相似度匹配
3. **LLM 后备**（可选）：使用大模型判断正误

### 8.2 答案提取

MQAEvaluator 从模型回复中提取答案字母，常见规则：
- 直接匹配单字母: `"A"`, `"B"` 等
- 从句子中提取: `"The answer is A"` → `"A"`
- 匹配选项内容: 如果回复包含某选项全文

### 8.3 分组统计字段

Daily-Omni 按以下字段进行分组统计：

| 字段 | 含义 | 示例值 |
|------|------|--------|
| `qa_type` | 问题类型 | Event Sequence, Spatial Understanding, ... |
| `content_parent_category` | 内容大类 | Lifestyle, Technology, ... |
| `video_category` | 视频类别 | Howto & Style, Science, ... |
| `video_duration` | 视频时长 | 30s, 1min, ... |

### 8.4 指标输出

```
总体准确率: XX.XX%
总样本数: 1197
正确数: XXX

按 qa_type 分组:
  Event Sequence: XX.XX% (XX/XX)
  Spatial Understanding: XX.XX% (XX/XX)
  ...

按 video_duration 分组:
  30s: XX.XX% (XX/XX)
  1min: XX.XX% (XX/XX)
  ...
```

### 8.5 C++ 对齐要点

```
C++ 需复现：
1. 答案字母提取（regex + 规则匹配）
2. 正确性判定: extracted_answer == gt_answer
3. 分组统计: 按 qa_type, content_parent_category, video_category, video_duration
4. 计算总体 accuracy 和各分组 accuracy
```

---

## 9. OOM 重试机制

### 9.1 OOM 策略配置

```json
{
    "oom_strategy": "speed",
    "max_audio_speed": 5.0,
    "max_audio_trim_end": 0.8,
    "audio_speed_increment": 0.2,
    "audio_trim_increment": 0.2
}
```

### 9.2 重试流程

```
初始: audio_speed=1.0, audio_trim_end=0.0

while True:
    构建 content (使用当前 audio_speed, audio_trim_end)
    try:
        response = model.chat(...)
        return response
    except OOM:
        清理显存
        if strategy == "speed":
            audio_speed += 0.2  (直到 5.0x)
        elif strategy == "trim":
            audio_trim_end += 0.2  (无上限)
        elif strategy == "speed_then_trim":
            先 speed 到上限，再 trim
```

### 9.3 C++ 对齐要点

```
C++ 可简化处理：
1. 如果 C++ 侧内存管理更可控，可不实现 OOM 重试
2. 如需对齐，记录每个样本的 audio_speed 和 audio_trim_end
3. 对于基准测试对齐，应使用 speed=1.0, trim_end=0.0（默认无重试）
```

---

## 10. 多卡并行策略

### 10.1 数据并行（Data Parallel）

- 使用 `torchrun` 启动多进程
- 每个进程绑定一张 GPU
- 数据集按 rank 切分

```bash
GPUS=4 bash scripts/run_daily_omni_o45py.sh
```

### 10.2 模型并行（Model Parallel）

- 单进程，使用 `accelerate.dispatch_model` 跨多卡
- GPU 0: 编码器 (apm, vpm, resampler, audio_projection)
- GPU 1~N: LLM 各层均匀分布

```bash
MP=2 bash scripts/run_daily_omni_o45py.sh
```

### 10.3 混合并行

```bash
MP=3 DP=2 bash scripts/run_daily_omni_o45py.sh  # 3卡模型并行 × 2组 = 6卡
```

### 10.4 模型结构与大小

| 模块 | 功能 | 大小 (BF16) | 占比 |
|------|------|------------|------|
| `apm` | 音频编码器 | 0.57 GiB | 3.3% |
| `vpm` | 视觉编码器 | 0.78 GiB | 4.5% |
| `resampler` | 跨模态投影 | 0.35 GiB | 2.0% |
| `audio_projection` | 音频投影 | 0.05 GiB | 0.3% |
| `llm` | 语言模型 (8B params) | 15.25 GiB | 87.4% |
| **总计** | | **~17.5 GiB** | 100% |

---

## 11. C++ 移植注意事项

### 11.1 关键依赖库映射

| Python 库 | C++ 替代方案 | 用途 |
|-----------|-------------|------|
| `decord` / `torchvision` | FFmpeg C API (`libavformat`, `libavcodec`) | 视频解码 |
| `librosa` | FFmpeg + `libsamplerate` / `libsndfile` | 音频加载与重采样 |
| `pyrubberband` | `librubberband` (C API) | 音频变速（保持音高） |
| `PIL.Image` | `stb_image` / OpenCV C++ | 图像处理 |
| `numpy` | `Eigen` / 裸数组 | 数值计算 |
| `transformers` | ONNX Runtime / TensorRT / llama.cpp | 模型推理 |
| `jsonlines` | `nlohmann/json` / `rapidjson` | JSON 解析 |

### 11.2 数据类型对齐

| Python 类型 | C++ 对齐类型 | 说明 |
|------------|-------------|------|
| `PIL.Image (RGB)` | `uint8[H][W][3]` | HWC 布局的 RGB 图像 |
| `np.ndarray (float32)` | `float[]` / `std::vector<float>` | 音频波形，16kHz |
| `str` | `std::string` | 文本 prompt |
| `List[Any]` | `std::vector<std::variant<Image, Audio, Text>>` | content 序列 |

### 11.3 精度对齐检查清单

| 检查项 | 对齐标准 |
|--------|---------|
| 帧采样索引 | 给定相同视频，C++ 和 Python 产出的 frame_idx 列表完全一致 |
| 时间戳 | timestamps 一致（浮点容差 < 1e-6） |
| 音频重采样 | 16kHz float32 波形，与 librosa 输出差异 < 1e-4 (L2 norm) |
| 音频切分 | 切分点一致（`int(timestamp * 16000)`） |
| prompt 文本 | 字节级一致 |
| 贪心解码 | 相同输入 → 相同 token 序列 |
| 答案提取 | 相同 response → 相同 extracted_answer |

### 11.4 建议的 C++ 模块划分

```
cpp_evalkit/
├── core/
│   ├── video_sampler.h/.cpp      # 视频帧采样（对齐 _sample_video_frame_indices）
│   ├── audio_loader.h/.cpp       # 音频加载与重采样（对齐 load_audio）
│   ├── av_interleaver.h/.cpp     # 音视频交错（对齐 load_video_and_audio_interleaved）
│   └── uniform_sampler.h/.cpp    # 均匀采样工具函数
├── prompt/
│   ├── template.h/.cpp           # 模板替换
│   └── options_builder.h/.cpp    # 选项格式化
├── dataset/
│   ├── jsonl_reader.h/.cpp       # JSONL 读取
│   └── dataset.h/.cpp            # 数据集抽象
├── model/
│   ├── inference.h/.cpp          # 模型推理接口
│   └── tokenizer.h/.cpp          # tokenizer
├── eval/
│   ├── answer_extractor.h/.cpp   # 答案提取
│   ├── mcq_evaluator.h/.cpp      # MCQ 评估器
│   └── group_stats.h/.cpp        # 分组统计
└── pipeline/
    └── runner.h/.cpp             # 完整 pipeline 编排
```

### 11.5 验证方法

建议的对齐验证流程：

1. **单元对齐**: 分别对比视频采样、音频加载、prompt 构建的输出
2. **端到端对齐**: 对比相同输入下 Python 和 C++ 的模型输入 tensor
3. **结果对齐**: 对比最终 accuracy（允许因浮点精度造成的微小差异）

```
# Python 侧导出中间结果
OMNI_DEBUG_CHAT_INPUT=1 python eval_main.py ...

# 对比:
- frame_idx: 帧索引列表
- timestamps: 时间戳列表
- audio_segments: 各段音频长度
- prompt_text: 最终 prompt 字符串
- model_input_tokens: tokenizer 输出的 token ids
- model_output: 生成的 response
```

---

## 附录 A: 关键代码路径索引

| 功能 | 源文件 | 关键函数/类 |
|------|--------|------------|
| 帧采样核心 | `o_e_Kit/utils/utils.py:49-86` | `_sample_video_frame_indices` |
| 均匀采样 | `o_e_Kit/utils/utils.py:43-47` | `uniform_sample` |
| decord 解码 | `o_e_Kit/utils/utils.py:89-129` | `_load_video_frames_with_decord` |
| FFmpeg 解码 | `o_e_Kit/utils/utils.py:208-304` | `_load_video_frames_with_ffmpeg` |
| torchvision 解码 | `o_e_Kit/utils/utils.py:132-205` | `_load_video_frames_with_torchvision` |
| 音频加载 | `o_e_Kit/utils/utils.py:415-445` | `load_audio` |
| 音频变速 | `o_e_Kit/utils/utils.py:448-473` | `speedup_audio` |
| 音频截断 | `o_e_Kit/utils/utils.py:685-708` | `trim_audio_segment` |
| 音视频对齐切分 | `o_e_Kit/utils/utils.py:607-682` | `get_video_frame_audio_segments` |
| 交错加载 | `o_e_Kit/utils/utils.py:711-777` | `load_video_and_audio_interleaved` |
| 生成配置 | `o_e_Kit/configs/omni_generation_configs_nosys_interleave.json` | `daily_omni` 条目 |
| 模型配置 | `configs/model_config/minicpm4o_3b_batch.json` | 全局 |
| 消息构建 | `o_e_Kit/models/minicpm/minicpmo_ou.py` | `build_messages`, `build_content` |
| 推理入口 | `o_e_Kit/models/minicpm/minicpmo_ou.py` | `_generate_chat` |
| 选项格式化 | `o_e_Kit/models/minicpm/minicpmo_ou.py` | `_build_options_prompt` |
| 数据集定义 | `o_e_Kit/datasets/omni_datasets.py` | `OmniEvalDataset` |
| 数据转换 | `scripts/convert_daily_omni.py` | `convert` |
| 评估器 | `o_e_Kit/utils/metrics/evaluator_omni.py` | `OmniMCQEvaluator` |

## 附录 B: 生成配置完整参数（Daily-Omni）

```json
{
    "user_prompt": "{media}\nCarefully read the following question and select the letter corresponding to the correct answer.Highlight the applicable choices without giving explanations.\n{question}\nOptions:\n{options}\nPlease select the correct answer from the options above. Only respond with the letter.",
    "system_prompt": "",
    "max_tokens": 128,
    "max_frames": 64,
    "max_fps": 1.0,
    "load_av": true,
    "interleave_fps": 1.0
}
```

解析后的完整配置字典：

```python
{
    "max_tokens": 128,
    "max_frames": 64,
    "max_fps": 1.0,
    "user_prompt": "...(同上)...",
    "system_prompt": "",
    "load_av": True,
    "keep_placeholder": False,      # 默认值
    "interleave_fps": 1.0,
    "use_image_id": False,          # 默认值
    "max_slice_nums": 1,            # 默认值（视频场景）
}
```
