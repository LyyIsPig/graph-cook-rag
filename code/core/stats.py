"""
轻量请求统计（P3 前端监控条 / P5 指标的 JSON 源）。
进程内、线程安全、滚动窗口：QPS(近60s)、p50/p95/p99、缓存命中分层、路由分布、拒答。
不依赖解析 Prometheus 文本，直接给 /api/stats 返回 JSON。
"""

import threading
import time
from collections import deque, defaultdict


class RequestStats:
    def __init__(self, window: int = 60, maxlen: int = 400):
        self._lock = threading.Lock()
        self.window = window
        self.recent = deque(maxlen=maxlen)        # (ts, latency_ms)
        self.route = defaultdict(int)             # strategy -> count
        self.cache = defaultdict(int)             # L1/L2/L3/MISS -> count
        self.refusals = defaultdict(int)          # reason -> count
        self.total = 0
        self.errors = 0

    def record_request(self, latency_ms: float, status: int = 200):
        with self._lock:
            self.total += 1
            if status >= 500:
                self.errors += 1
            self.recent.append((time.time(), latency_ms))

    def record_route(self, strategy):
        if strategy:
            with self._lock:
                self.route[str(strategy)] += 1

    def record_cache(self, layer):
        # layer: 'L1'/'L2'/'L3'(可能带 sim 后缀如 'L3(0.95)')/'MISS'/None
        key = (layer or "MISS").split("(")[0] if layer else "MISS"
        with self._lock:
            self.cache[key] += 1

    def record_refusal(self, reason):
        if reason:
            with self._lock:
                self.refusals[str(reason)] += 1

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            recent = [(t, l) for (t, l) in self.recent if now - t <= self.window]
            lats = sorted(l for _, l in recent)
            qps = len(recent) / self.window if self.window else 0.0
            route = dict(self.route)
            cache = dict(self.cache)
            refusals = dict(self.refusals)
            total, errors = self.total, self.errors

        def pct(xs, p):
            if not xs:
                return 0.0
            return xs[max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))]

        cache_total = sum(cache.values()) or 1
        hits = cache.get("L1", 0) + cache.get("L2", 0) + cache.get("L3", 0)
        return {
            "window_s": self.window,
            "qps": round(qps, 2),
            "recent_requests": len(lats),
            "p50_ms": round(pct(lats, 50), 1),
            "p95_ms": round(pct(lats, 95), 1),
            "p99_ms": round(pct(lats, 99), 1),
            "total_requests": total,
            "errors": errors,
            "cache_hit_rate": round(hits / cache_total, 3),
            "cache_layers": cache,
            "route_distribution": route,
            "refusals": refusals,
        }
