"""
llama-server 进程生命周期管理：启动、健康检查、停止
"""
import os
import time
import signal
import subprocess
import logging
from glob import glob
from typing import List, Optional

import requests

from eval_cpp_config import (
    LLAMA_SERVER_BIN, LLM_MODEL_PATH, GGUF_MODEL_DIR,
    BASE_PORT, CTX_SIZE,
    TEMPERATURE, REPEAT_PENALTY,
    SERVER_STARTUP_TIMEOUT, SERVER_HEALTH_INTERVAL,
)

logger = logging.getLogger(__name__)


class ServerInstance:
    """封装单个 llama-server 进程"""

    def __init__(self, gpu_id: int, port: int, proc: subprocess.Popen, log_file):
        self.gpu_id = gpu_id
        self.port = port
        self.proc = proc
        self.log_file = log_file
        self.base_url = f"http://127.0.0.1:{port}"

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


def _rotate_server_logs(log_dir: str, gpu_id: int, keep_rotated_logs: int = 5) -> str:
    """
    轮转 server 日志，避免单文件无限变大。

    - 当前运行始终写入: server_gpu{gpu_id}.log
    - 若旧文件存在且非空，先重命名为带时间戳历史日志
    - 仅保留最近 keep_rotated_logs 份历史日志
    """
    os.makedirs(log_dir, exist_ok=True)
    active_log = os.path.join(log_dir, f"server_gpu{gpu_id}.log")

    if os.path.exists(active_log) and os.path.getsize(active_log) > 0:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rotated_log = os.path.join(log_dir, f"server_gpu{gpu_id}_{ts}.log")
        os.replace(active_log, rotated_log)

    rotated_pattern = os.path.join(log_dir, f"server_gpu{gpu_id}_*.log")
    rotated_logs = sorted(glob(rotated_pattern), key=os.path.getmtime, reverse=True)
    for old_log in rotated_logs[keep_rotated_logs:]:
        try:
            os.remove(old_log)
        except OSError as e:
            logger.warning(f"Failed to remove old log {old_log}: {e}")

    return active_log


def start_server(
    gpu_id: int,
    port: int,
    model_path: str = LLM_MODEL_PATH,
    ctx_size: int = CTX_SIZE,
    log_dir: Optional[str] = None,
    keep_rotated_logs: int = 5,
) -> ServerInstance:
    """
    启动一个 llama-server 进程，绑定到指定 GPU。

    返回 ServerInstance 对象。
    """
    if not os.path.isfile(LLAMA_SERVER_BIN):
        raise FileNotFoundError(f"llama-server binary not found: {LLAMA_SERVER_BIN}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    extra_ld = os.environ.get("EXTRA_LD_LIBRARY_PATH", "")
    if extra_ld:
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{extra_ld}:{ld_path}" if ld_path else extra_ld

    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--n-gpu-layers", "999",
        "--temp", str(TEMPERATURE),
        "--repeat-penalty", str(REPEAT_PENALTY),
        "--host", "0.0.0.0",
    ]

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    log_path = _rotate_server_logs(log_dir, gpu_id, keep_rotated_logs=keep_rotated_logs)
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    logger.info(f"Starting server on GPU {gpu_id}, port {port}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    return ServerInstance(gpu_id=gpu_id, port=port, proc=proc, log_file=log_f)


def wait_server_ready(server: ServerInstance, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """轮询健康检查，等待 server 就绪。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not server.is_alive():
            logger.error(f"Server GPU {server.gpu_id} exited prematurely (exit code {server.proc.returncode})")
            return False
        if server.health_check():
            logger.info(f"Server GPU {server.gpu_id} ready (port {server.port}), took {time.time()-t0:.1f}s")
            return True
        time.sleep(SERVER_HEALTH_INTERVAL)
    logger.error(f"Server GPU {server.gpu_id} health check timeout after {timeout}s")
    return False


def stop_server(server: ServerInstance):
    """优雅停止 server 进程。"""
    if server.proc.poll() is not None:
        logger.info(f"Server GPU {server.gpu_id} already exited")
    else:
        logger.info(f"Stopping server GPU {server.gpu_id} (pid={server.proc.pid})")
        try:
            os.killpg(os.getpgid(server.proc.pid), signal.SIGTERM)
            server.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"Server GPU {server.gpu_id} did not stop gracefully, sending SIGKILL")
            os.killpg(os.getpgid(server.proc.pid), signal.SIGKILL)
            server.proc.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error stopping server GPU {server.gpu_id}: {e}")
    if server.log_file and not server.log_file.closed:
        server.log_file.close()


def start_all_servers(
    num_gpus: int,
    base_port: int = BASE_PORT,
    model_path: str = LLM_MODEL_PATH,
    ctx_size: int = CTX_SIZE,
    keep_rotated_logs: int = 5,
) -> List[ServerInstance]:
    """启动所有 GPU 上的 server 并等待就绪。"""
    servers: List[ServerInstance] = []
    for gpu_id in range(num_gpus):
        port = base_port + gpu_id
        srv = start_server(
            gpu_id,
            port,
            model_path=model_path,
            ctx_size=ctx_size,
            keep_rotated_logs=keep_rotated_logs,
        )
        servers.append(srv)

    logger.info(f"Waiting for {num_gpus} servers to become ready...")
    failed = []
    for srv in servers:
        if not wait_server_ready(srv):
            failed.append(srv.gpu_id)

    if failed:
        logger.error(f"Failed to start servers on GPUs: {failed}")
        stop_all_servers(servers)
        raise RuntimeError(f"Server startup failed on GPUs: {failed}")

    logger.info(f"All {num_gpus} servers ready")
    return servers


def stop_all_servers(servers: List[ServerInstance]):
    """停止所有 server。"""
    for srv in servers:
        try:
            stop_server(srv)
        except Exception as e:
            logger.warning(f"Error stopping server GPU {srv.gpu_id}: {e}")
    logger.info("All servers stopped")
