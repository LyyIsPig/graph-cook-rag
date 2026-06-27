"""
结构化日志（P0）。
- JSON 格式输出到 stdout，方便后续接 ELK / Loki / Langfuse 关联。
- request_id 通过 contextvars 在单次请求/单条链路内透传，便于全链路排障。
"""

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar

# 全局请求 ID（默认 "-" 表示非请求上下文，如启动阶段或 CLI）
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class StructuredFormatter(logging.Formatter):
    """把每条日志渲染成一行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: "str | None" = None) -> logging.Logger:
    """
    初始化根日志：单 handler、JSON 格式、统一 level。
    幂等：重复调用只替换 handler，不会叠加。
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    return logging.getLogger("graph_rag")


class RequestContext:
    """请求上下文：进入时生成/绑定 request_id，退出时还原。"""

    def __init__(self, request_id: "str | None" = None):
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self._token = None

    def __enter__(self) -> str:
        self._token = request_id_var.set(self.request_id)
        return self.request_id

    def __exit__(self, *exc_info):
        if self._token is not None:
            request_id_var.reset(self._token)


def get_request_id() -> str:
    return request_id_var.get()
