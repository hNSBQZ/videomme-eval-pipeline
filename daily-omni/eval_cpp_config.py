"""
Daily-Omni CPP 评测 Pipeline 配置
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ==================== 路径配置 ====================

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLAMA_SERVER_BIN = os.environ.get(
    "LLAMA_SERVER_BIN",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "build", "bin", "llama-server"),
)

LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf", "llm", "MiniCPM-o-4_5-llm-Q4_K_M.gguf"),
)
GGUF_MODEL_DIR = os.environ.get(
    "GGUF_MODEL_DIR",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf"),
)

# 数据集
DATASET_DIR = os.environ.get(
    "DATASET_DIR",
    os.path.join(os.path.expanduser("~"), "daily-omni"),
)
ANNOTATION_PATH = os.environ.get(
    "ANNOTATION_PATH",
    os.path.join(DATASET_DIR, "daily_omni.jsonl"),
)

# 输出
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "output_daily_omni_cpp.json")

# 临时文件目录（帧 JPG + 音频 WAV 片段）
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_media")
FRAME_TEMP_DIR = os.path.join(TEMP_DIR, "frames")
AUDIO_TEMP_DIR = os.path.join(TEMP_DIR, "audio_segments")

# ==================== Server 配置 ====================

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
BASE_PORT = int(os.environ.get("BASE_PORT", "8080"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "40960"))

# ==================== 评测参数（对齐 evalkit Daily-Omni 配置）====================

MAX_NUM_FRAMES = 64
MAX_FPS = 1.0
MAX_SLICE_NUMS = 1          # 视频场景不分块
MAX_TOKENS = 128
AUDIO_SR = 16000             # 音频采样率

# 解码策略：对齐 Python sampling=False (greedy)
TEMPERATURE = 0.0
REPEAT_PENALTY = 1.0         # Daily-Omni 不设 repeat_penalty

# Server omni_init 参数
MEDIA_TYPE = 2               # omni = audio + vision
USE_TTS = False

# ==================== Prompt 模板（对齐 omni_generation_configs_nosys_interleave.json）====================

USER_PROMPT_TEMPLATE = (
    "Carefully read the following question and select the letter corresponding to the correct answer."
    "Highlight the applicable choices without giving explanations.\n"
    "{question}\n"
    "Options:\n{options}"
    "Please select the correct answer from the options above. Only respond with the letter."
)

# ==================== 超时与重试 ====================

SERVER_STARTUP_TIMEOUT = 300
SERVER_HEALTH_INTERVAL = 2
HTTP_TIMEOUT = 300
SSE_READ_TIMEOUT = 120
