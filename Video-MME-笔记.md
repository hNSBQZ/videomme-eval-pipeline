# Video-MME Benchmark — MiniCPM 评测笔记

## 一、数据集结构

### Parquet 文件

路径：`Video-MME/videomme/test-00000-of-00001.parquet`

- **2700 道题**，**900 个视频**（每个视频 3 道题）
- 字段：

| 字段 | 含义 |
|---|---|
| `video_id` | 视频编号（如 001） |
| `duration` | 时长类别：`short` / `medium` / `long` |
| `domain` | 领域（6 类：Knowledge, Film & Television, Sports Competition, Artistic Performance, Life Record, Multilingual） |
| `sub_category` | 细分子类 |
| `videoID` | YouTube video ID → 对应 `data/` 里的 mp4 文件名 |
| `question_id` | 题目编号（如 001-1） |
| `task_type` | 题目类型（12 种：Counting Problem, Action Recognition, OCR Problems 等） |
| `question` | 问题文本 |
| `options` | 四个选项（A/B/C/D） |
| `answer` | 正确答案 |

### 目录结构

```
Video-MME/
├── data/          # 视频 (.mp4)，文件名 = videoID
├── subtitle/      # 字幕 (.srt)，744个，所有 long 视频都有
└── videomme/      # parquet 标注文件（问题+答案）
```

三者通过 `videoID` 关联。

---

## 二、字幕处理

**字幕不烧录进视频，作为纯文本拼到 prompt 中。**

官方做法：根据采样帧的时间戳，从 `.srt` 提取对应字幕，拼到 prompt：

```
This video's subtitles are listed below:
[对应帧时间戳的字幕文本]
Select the best answer to the following multiple-choice question based on the video...
```

> **当前 evalkit 未实现字幕加载**，MiniCPM 评测时 prompt 中没有字幕。如需带字幕评测需自行添加。
> 官方推荐工具：[video-slicer](https://github.com/look4u-ok/video-slicer)

---

## 三、MiniCPM 帧采样

### 参数

| 参数 | 值 |
|---|---|
| max_frames | 64 |
| max_fps | 1.0 |
| scale_resolution | 448（模型输入时缩放） |

### 采样逻辑

- **短视频**（≤64秒）：1fps 取帧，有多少秒就多少帧
- **长视频**（>64秒）：先按 0.1s 粒度生成候选帧，再 `uniform_sample` 均匀降采样到 64 帧
- 采样时**不做 resize**，保持原始分辨率；MiniCPM 模型侧以 `scale_resolution=448` 缩放
- 解码器优先级：decord → ffmpeg → torchvision

---

## 四、MiniCPM 推理流程

### 数据流

```
视频文件 (.mp4)
    │
    ▼
encode_video(video_path, MAX_NUM_FRAMES=64, MAX_FPS=1)
    │  短视频: 1fps（如30秒→30帧）
    │  长视频: 均匀采样到64帧
    ▼
List[PIL.Image]  (原始分辨率, ≤64帧)
    │
    ▼
build_content() 拼接:
    [img1, img2, ..., imgN, "问题\n选项\nPlease select..."]
    │
    ▼
msgs = [{"role": "user", "content": [img1, ..., imgN, "prompt"]}]
    │
    ▼
model.chat(msgs=msgs, sampling=False, max_new_tokens=128,
           max_slice_nums=1, use_image_id=False)
    │
    ▼
返回 "A" / "B" / "C" / "D"
```

### Prompt 模板

```
{视频帧序列}
{question}
A. xxx
B. xxx
C. xxx
D. xxx
Please select the correct answer from the options above. Only respond with the letter.
```

### 关键细节

- 每次只送一道题，2700 题循环 2700 次
- 视频帧放前面，文本放后面
- `max_slice_nums=1`：视频帧不做图片切片（区别于图片模式的 9 切片）
- `sampling=False`：beam search（num_beams=3, repetition_penalty=1.2），非纯贪心（见 `modeling_minicpmo.py:696-700`）

### MiniCPM 配置对比

| 参数 | videomme（纯视频） | videomme_short（音视频） |
|---|---|---|
| max_frames | 64 | 64 |
| load_av | **false** | **true** |
| interleave_fps | 1.0 | 1.0 |
| max_tokens | 128 | 128 |

`videomme_short` 会从视频中提取音频，按时间戳交错排列：`[frame1, audio_seg1, frame2, audio_seg2, ...]`

---

## 五、关键代码位置

| 功能 | 文件 | 函数 |
|---|---|---|
| 帧采样 | `evalkit/o_e_Kit/utils/utils.py` | `encode_video()`, `_sample_video_frame_indices()` |
| 配置 | `evalkit/o_e_Kit/configs/omni_generation_configs.json` | videomme / videomme_short 节点 |
| Prompt 构建 | `evalkit/o_e_Kit/models/minicpm/minicpmo_ou.py` | `build_messages()`, `build_content()` |
| 推理 | `evalkit/o_e_Kit/models/minicpm/minicpmo_ou.py` | `_generate_chat()` → `model.chat()` |
| 数据集 | `evalkit/o_e_Kit/datasets/omni_datasets.py` | `OmniEvalDataset` |
| 路径注册 | `evalkit/o_e_Kit/utils/args/dataset_args.py` | videomme 相关参数 |

---

## 六、CPP 版本测试待办

- [ ] 确认 CPP 版本的帧采样逻辑是否与 Python 版一致（64帧、1fps、uniform_sample）
- [ ] 确认 CPP 版本的图片预处理是否对齐（scale_resolution=448）
- [ ] 确认 Prompt 模板完全一致
- [ ] 确认解码方式对齐（贪心解码、max_tokens=128）
- [ ] 对比 Python 版和 CPP 版在相同输入下的输出是否一致
