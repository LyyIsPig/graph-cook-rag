"""
Prometheus 业务指标（P5 监控）。
指标命名空间 `graphrag_`，分两类：
- HTTP 层：请求计数、延迟直方图（middleware 自动记）
- 业务层：缓存命中(L1/L2/L3/MISS)、路由策略分布、拒答原因、LLM 调用、各阶段耗时(gate/route/retrieve/generate)
导出 /metrics 供 Prometheus 抓取；也可直接 curl 看裸文本。
"""

import time
from contextlib import contextmanager

from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)

# ---- HTTP 层（FastAPI middleware 自动记录）----
HTTP_REQUESTS = Counter(
    "graphrag_http_requests_total", "HTTP 请求总数",
    ["endpoint", "method", "status"])
HTTP_LATENCY = Histogram(
    "graphrag_http_request_duration_seconds", "HTTP 请求端到端延迟(秒)",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))

# ---- 业务层 ----
CACHE_HITS = Counter(
    "graphrag_cache_hits_total", "答案缓存命中情况",
    ["layer"])  # layer: L1 / L2 / L3 / MISS
ROUTE_STRATEGY = Counter(
    "graphrag_route_strategy_total", "路由策略选择分布",
    ["strategy"])  # hybrid_traditional / graph_rag / combined
REFUSALS = Counter(
    "graphrag_refusals_total", "拒答次数",
    ["reason"])  # vector_low / no_retrieval / check_error / prompt_refusal
LLM_CALLS = Counter(
    "graphrag_llm_calls_total", "LLM 调用次数(按用途)",
    ["stage"])  # stage: analyze_query / extract_keywords / generate
STAGE_LATENCY = Histogram(
    "graphrag_stage_duration_seconds", "单次问答各阶段耗时(秒)",
    ["stage"],  # gate / route / retrieve / generate / cache_lookup
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))


def metrics_body() -> tuple:
    """返回 (bytes, content_type) 供 /metrics 端点。"""
    return generate_latest(), CONTENT_TYPE_LATEST


@contextmanager
def stage_timer(stage: str):
    """计时一个阶段并记入 STAGE_LATENCY。用法：with stage_timer('generate'): ..."""
    t0 = time.time()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(stage=stage).observe(time.time() - t0)
