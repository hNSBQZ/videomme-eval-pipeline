"""
llama-server HTTP 客户端封装（Daily-Omni 版）。

在 VideoMME 版基础上扩展：
  - prefill_audio: 音频段 prefill
  - prefill_interleaved: 交错 prefill 图片帧 + 音频段
"""
import json
import logging
from typing import Any, Union

import requests

from eval_cpp_config import (
    GGUF_MODEL_DIR, MEDIA_TYPE, USE_TTS, MAX_TOKENS, MAX_SLICE_NUMS,
    HTTP_TIMEOUT, SSE_READ_TIMEOUT,
)

logger = logging.getLogger(__name__)


class OmniServerClient:
    """封装对单个 llama-server 实例的所有 HTTP 调用。"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _dump_payload(self, payload: Any) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return repr(payload)

    def _post_json(
        self,
        api_path: str,
        payload: dict,
        timeout: Union[int, tuple],
        stream: bool = False,
    ) -> requests.Response:
        url = f"{self.base_url}{api_path}"
        logger.debug("HTTP REQUEST POST %s payload=%s", url, self._dump_payload(payload))
        resp = self.session.post(url, json=payload, timeout=timeout, stream=stream)
        logger.debug("HTTP RESPONSE POST %s status=%s", url, resp.status_code)
        resp.raise_for_status()
        return resp

    # ==================== omni_init ====================

    def omni_init(
        self,
        media_type: int = MEDIA_TYPE,
        use_tts: bool = USE_TTS,
        n_predict: int = MAX_TOKENS,
        model_dir: str = GGUF_MODEL_DIR,
    ) -> dict:
        payload = {
            "media_type": media_type,
            "use_tts": use_tts,
            "n_predict": n_predict,
            "model_dir": model_dir,
        }
        resp = self._post_json("/v1/stream/omni_init", payload, timeout=HTTP_TIMEOUT)
        result = resp.json()
        logger.info(f"omni_init OK: {result}")
        return result

    # ==================== reset ====================

    def reset(self) -> dict:
        resp = self._post_json("/v1/stream/reset", {}, timeout=HTTP_TIMEOUT)
        result = resp.json()
        logger.debug(f"reset OK: {result}")
        return result

    # ==================== prefill: 图片 ====================

    def prefill_image(
        self,
        img_path: str,
        cnt: int,
        max_slice_nums: int = MAX_SLICE_NUMS,
        skip_system_prompt: bool = False,
        frame_prompt: str = "\n",
    ) -> dict:
        """
        POST /v1/stream/prefill — 图片帧 prefill

        frame_prompt="\n" 对齐 Python "\n".join 时每帧后的换行分隔。
        """
        payload = {
            "audio_path_prefix": "",
            "img_path_prefix": img_path,
            "cnt": cnt,
            "max_slice_nums": max_slice_nums,
            "prompt": frame_prompt,
        }
        if skip_system_prompt:
            payload["skip_system_prompt"] = True
        resp = self._post_json("/v1/stream/prefill", payload, timeout=HTTP_TIMEOUT)
        return resp.json()

    # ==================== prefill: 音频 ====================

    def prefill_audio(
        self,
        audio_path: str,
        cnt: int,
        audio_prompt: str = "\n",
    ) -> dict:
        """
        POST /v1/stream/prefill — 音频段 prefill

        audio_prompt="\n" 对齐 Python "\n".join 时每段音频后的换行分隔。
        """
        payload = {
            "audio_path_prefix": audio_path,
            "img_path_prefix": "",
            "cnt": cnt,
            "prompt": audio_prompt,
        }
        resp = self._post_json("/v1/stream/prefill", payload, timeout=HTTP_TIMEOUT)
        return resp.json()

    # ==================== prefill: 文本 ====================

    def prefill_text(self, prompt: str, cnt: int) -> dict:
        payload = {
            "audio_path_prefix": "",
            "img_path_prefix": "",
            "cnt": cnt,
            "prompt": prompt,
        }
        resp = self._post_json("/v1/stream/prefill", payload, timeout=HTTP_TIMEOUT)
        return resp.json()

    # ==================== decode (SSE) ====================

    def decode(self, round_idx: int = 0) -> str:
        payload = {
            "stream": True,
            "round_idx": round_idx,
        }
        resp = self._post_json("/v1/stream/decode", payload, stream=True,
                               timeout=(HTTP_TIMEOUT, SSE_READ_TIMEOUT))
        return self._collect_sse_text(resp)

    def _collect_sse_text(self, resp: requests.Response) -> str:
        fragments = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning(f"SSE JSON parse error: {data_str}")
                continue

            content = event.get("content", "")
            if content:
                fragments.append(content)

            if event.get("stop", False):
                break

        full_text = "".join(fragments)
        logger.debug(f"decode result: {full_text!r}")
        return full_text

    # ==================== 便捷方法：交错 prefill ====================

    def prefill_interleaved(
        self,
        frame_paths: list,
        audio_paths: list,
        skip_system_prompt: bool = True,
        max_slice_nums: int = MAX_SLICE_NUMS,
        frame_prompt: str = "\n",
        audio_prompt: str = "\n",
    ) -> int:
        """
        交错 prefill 图片帧和音频段。

        按 [frame_0, audio_0, frame_1, audio_1, ...] 顺序 prefill，
        对齐 Python content = [PIL.Image, np.ndarray, PIL.Image, np.ndarray, ..., str] 结构。

        返回总 prefill 次数（即下一步 prefill_text 应使用的 cnt 值）。
        """
        cnt = 0
        num_pairs = min(len(frame_paths), len(audio_paths)) if audio_paths else len(frame_paths)

        for i in range(num_pairs):
            self.prefill_image(
                img_path=frame_paths[i],
                cnt=cnt,
                max_slice_nums=max_slice_nums,
                skip_system_prompt=(skip_system_prompt and i == 0),
                frame_prompt=frame_prompt,
            )
            cnt += 1

            if audio_paths and i < len(audio_paths):
                self.prefill_audio(
                    audio_path=audio_paths[i],
                    cnt=cnt,
                    audio_prompt=audio_prompt,
                )
                cnt += 1

        # 如果帧数多于音频段数，处理剩余帧
        for i in range(num_pairs, len(frame_paths)):
            self.prefill_image(
                img_path=frame_paths[i],
                cnt=cnt,
                max_slice_nums=max_slice_nums,
                skip_system_prompt=(skip_system_prompt and cnt == 0),
                frame_prompt=frame_prompt,
            )
            cnt += 1

        return cnt

    def close(self):
        self.session.close()
