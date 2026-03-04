"""
Video-MME CPP 评测 Pipeline 配置
"""
import os

# ==================== 路径配置 ====================

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLAMA_SERVER_BIN = os.path.join(PROJ_ROOT, "llama.cpp-omni", "build", "llama-server")

# 模型文件
LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf", "llm", "MiniCPM-o-4_5-llm-Q4_K_M.gguf"),
)
GGUF_MODEL_DIR = os.environ.get(
    "GGUF_MODEL_DIR",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf"),
)

# 数据集
PARQUET_PATH = os.path.join(PROJ_ROOT, "Video-MME", "videomme", "test-00000-of-00001.parquet")
VIDEO_DATA_DIR = os.path.join(PROJ_ROOT, "Video-MME", "data")

# 输出
OUTPUT_DIR = os.path.join(PROJ_ROOT, "cpp-eval", "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "output_videomme_cpp.json")

# 临时帧文件目录
FRAME_TEMP_DIR = os.path.join(PROJ_ROOT, "cpp-eval", "tmp_frames")

# ==================== Server 配置 ====================

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
BASE_PORT = int(os.environ.get("BASE_PORT", "8080"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))

# ==================== 评测参数 ====================

MAX_NUM_FRAMES = 64
MAX_FPS = 1.0
MAX_SLICE_NUMS = 1
MAX_TOKENS = 128

# 解码策略：greedy + repetition_penalty（近似 Python beam search）
TEMPERATURE = 0.0
REPEAT_PENALTY = 1.2

# Server omni_init 参数
MEDIA_TYPE = 2       # omni = audio + vision（需要 vision encoder）
USE_TTS = False

# ==================== Prompt 模板 ====================

USER_PROMPT_TEMPLATE = (
    "{question}\n{options}\n"
    "Please select the correct answer from the options above. Only respond with the letter."
)

# ==================== 超时与重试 ====================

SERVER_STARTUP_TIMEOUT = 300   # 等待 server 启动（秒）
SERVER_HEALTH_INTERVAL = 2     # 健康检查轮询间隔（秒）
HTTP_TIMEOUT = 300             # HTTP 请求超时（秒），视频 prefill 可能较慢
SSE_READ_TIMEOUT = 120         # SSE 流式读取超时（秒）
