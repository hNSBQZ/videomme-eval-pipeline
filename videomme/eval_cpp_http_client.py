"""
llama-server HTTP 客户端封装。

覆盖所有 omni streaming API：
  - omni_init    初始化 omni 上下文
  - reset        清空 KV cache
  - prefill      填充图片帧 / 文本 prompt
  - decode       生成文字（支持 SSE 流式）
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
        """把请求 payload 序列化为可读字符串，便于全链路排查。"""
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
        """
        统一封装 POST JSON 请求，并记录完整请求内容与响应状态。
        """
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
        """
        POST /v1/stream/omni_init

        初始化 omni 上下文，加载 vision/audio encoder。
        """
        payload = {
            "media_type": media_type,
            "use_tts": use_tts,
            "n_predict": n_predict,
            "model_dir": model_dir,
        }
        resp = self._post_json(
            api_path="/v1/stream/omni_init",
            payload=payload,
            timeout=HTTP_TIMEOUT,
        )
        result = resp.json()
        logger.info(f"omni_init OK: {result}")
        return result

    # ==================== reset ====================

    def reset(self) -> dict:
        """
        POST /v1/stream/reset

        清空 KV cache，为下一道题做准备。
        """
        resp = self._post_json(
            api_path="/v1/stream/reset",
            payload={},
            timeout=HTTP_TIMEOUT,
        )
        result = resp.json()
        logger.debug(f"reset OK: {result}")
        return result

    # ==================== prefill ====================

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

        Args:
            img_path: 图片绝对路径
            cnt: 帧序号（从 0 开始）
            max_slice_nums: 图片切片数（1=不切片，对齐 Python 评测）
            skip_system_prompt: 第一帧时传 True，跳过 voice clone system prompt
            frame_prompt: 图片帧后追加的文本（默认换行，对齐每帧分隔）
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

        resp = self._post_json(
            api_path="/v1/stream/prefill",
            payload=payload,
            timeout=HTTP_TIMEOUT,
        )
        return resp.json()

    def prefill_text(self, prompt: str, cnt: int) -> dict:
        """
        POST /v1/stream/prefill — 文本 prompt prefill

        在所有图片帧之后调用，将问题+选项注入 KV cache。
        """
        payload = {
            "audio_path_prefix": "",
            "img_path_prefix": "",
            "cnt": cnt,
            "prompt": prompt,
        }
        resp = self._post_json(
            api_path="/v1/stream/prefill",
            payload=payload,
            timeout=HTTP_TIMEOUT,
        )
        return resp.json()

    # ==================== decode (SSE) ====================

    def decode(self, round_idx: int = 0) -> str:
        """
        POST /v1/stream/decode (stream=true)

        发起 SSE 流式解码，收集所有文本 fragment 并拼接返回。

        SSE 事件格式：
          data: {"content": "xxx", "stop": false, ...}
          ...
          data: [DONE]

        返回完整的模型输出文本。
        """
        payload = {
            "stream": True,
            "round_idx": round_idx,
        }
        resp = self._post_json(
            api_path="/v1/stream/decode",
            payload=payload,
            stream=True,
            timeout=(HTTP_TIMEOUT, SSE_READ_TIMEOUT),
        )
        return self._collect_sse_text(resp)

    def _collect_sse_text(self, resp: requests.Response) -> str:
        """从 SSE 流中收集所有文本 fragment。"""
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

    # ==================== 便捷方法 ====================

    def prefill_all_frames(
        self,
        frame_paths: list,
        skip_system_prompt: bool = True,
        max_slice_nums: int = MAX_SLICE_NUMS,
        frame_prompt: str = "\n",
    ) -> None:
        """
        依次 prefill 所有图片帧。

        第一帧 (cnt=0) 传 skip_system_prompt=True。
        """
        for idx, path in enumerate(frame_paths):
            self.prefill_image(
                img_path=path,
                cnt=idx,
                max_slice_nums=max_slice_nums,
                skip_system_prompt=(skip_system_prompt and idx == 0),
                frame_prompt=frame_prompt,
            )

    def close(self):
        self.session.close()
