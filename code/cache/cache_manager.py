"""
多级缓存管理器（P1）。

四级缓存（命名空间前缀 `grag:`）：
  - L1 进程内 LRU        answer/route    normalized query     容量 cache_l1_size
  - L2 Redis 精确        完整答案+溯源   `grag:answer:<qh>`   TTL cache_ttl_answer
  - L3 语义              相似 query 复用 BGE 余弦 >= 阈值     与答案同 TTL
  - L4 路由决策          analyze_query 结果 `grag:route:<qh>` TTL cache_ttl_route
  (+) embedding 缓存     query 向量      `grag:emb:<qh>`      TTL cache_ttl_embedding

降级：cache_enabled=False 或 Redis 不可达 → get_* 返回 None（miss）、set_* no-op，系统照常工作。
线程安全：FastAPI 同步端点跑在线程池，L1/metrics/语义索引用 RLock 保护；Redis 客户端自带连接池。
"""

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .redis_client import RedisClient

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, redis_client: RedisClient, embedding_fn: Callable[[str], Any],
                 config, namespace: str = "grag"):
        self.r = redis_client
        self.embed = embedding_fn          # str -> list[float] / np.ndarray
        self.cfg = config
        self.ns = namespace
        self._lock = threading.RLock()

        # L1 进程内 LRU（answer 用）
        self._l1: "OrderedDict[str, dict]" = OrderedDict()
        # L3 语义索引的进程内镜像：[(qhash, query_text, embedding_np)]，懒加载自 Redis
        self._sem_index = []
        self._sem_loaded = False
        # 埋点
        self._metrics = defaultdict(int)

    # ---------- 基础工具 ----------
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.cache_enabled)

    @property
    def redis_on(self) -> bool:
        return self.enabled and self.r.available

    @staticmethod
    def _normalize(query: str) -> str:
        # 去所有空白 + 转小写：'宫保鸡丁 怎么做' / '宫保鸡丁怎么做' 视为同一 query
        return re.sub(r"\s+", "", query).strip().lower()

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _key(self, kind: str, qh: str) -> str:
        return f"{self.ns}:{kind}:{qh}"

    def _set_l1(self, qh: str, payload: dict):
        """写入 L1 并维持容量上限（LRU 淘汰最旧）。"""
        with self._lock:
            self._l1[qh] = payload
            self._l1.move_to_end(qh)
            while len(self._l1) > self.cfg.cache_l1_size:
                self._l1.popitem(last=False)

    # ---------- 答案缓存：L1 + L2 ----------
    def get_answer(self, query: str) -> Optional[Tuple[dict, str]]:
        """命中返回 (payload, layer)；payload={answer, strategy, sources, latency_ms, refused}。"""
        if not self.enabled:
            return None
        qh = self._sha(self._normalize(query))
        with self._lock:
            if qh in self._l1:                      # L1
                self._l1.move_to_end(qh)
                self._metrics["answer_hit_l1"] += 1
                return self._l1[qh], "L1"
        if self.redis_on:                          # L2
            raw = self.r.get(self._key("answer", qh))
            if raw:
                try:
                    payload = json.loads(raw)
                    self._set_l1(qh, payload)
                    with self._lock:
                        self._metrics["answer_hit_l2"] += 1
                    return payload, "L2"
                except (ValueError, TypeError) as e:
                    logger.warning(f"答案缓存反序列化失败，丢弃: {e}")
        with self._lock:
            self._metrics["answer_miss"] += 1
        return None

    def set_answer(self, query: str, payload: dict):
        if not self.enabled:
            return
        qh = self._sha(self._normalize(query))
        self._set_l1(qh, payload)
        if self.redis_on:
            self.r.setex(self._key("answer", qh), self.cfg.cache_ttl_answer,
                         json.dumps(payload, ensure_ascii=False))

    # ---------- 语义缓存：L3 ----------
    def get_semantic_answer(self, query: str) -> Optional[Tuple[dict, float, str]]:
        """返回 (payload, similarity, 'L3') 或 None。embedding 经 L3 embedding 缓存去重计算。"""
        if not self.redis_on:
            return None
        qv = self._embed_cached(query)
        if qv is None:
            return None
        self._ensure_sem_loaded()
        with self._lock:
            index_snapshot = list(self._sem_index)
        if not index_snapshot:
            with self._lock:
                self._metrics["answer_miss"] += 1
            return None
        sim, payload = self._best_match(qv, index_snapshot)
        if payload is not None and sim >= self.cfg.cache_semantic_threshold:
            with self._lock:
                self._metrics["answer_hit_sem"] += 1
            return payload, sim, "L3"
        with self._lock:
            self._metrics["answer_miss"] += 1
        return None

    def register_semantic(self, query: str, payload: dict):
        """新答案落库时登记进语义索引（payload + 向量 + 原文），供后续相似 query 复用。"""
        if not self.redis_on:
            return
        qv = self._embed_cached(query)
        if qv is None:
            return
        qh = self._sha(self._normalize(query))
        self.r.hset(f"{self.ns}:sem:payload", qh, json.dumps(payload, ensure_ascii=False))
        self.r.hset(f"{self.ns}:sem:embed", qh, json.dumps(qv.astype(np.float32).tolist()))
        self.r.hset(f"{self.ns}:sem:text", qh, query)
        with self._lock:
            # 去重替换同 qh 的旧条目
            self._sem_index = [(q, t, v) for (q, t, v) in self._sem_index if q != qh]
            self._sem_index.append((qh, query, qv))

    def _best_match(self, qv: np.ndarray, index_snapshot) -> Tuple[float, Optional[dict]]:
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        best_sim, best_qh = -1.0, None
        for qh, _txt, vec in index_snapshot:
            vn = vec / (np.linalg.norm(vec) + 1e-12)
            sim = float(np.dot(qn, vn))
            if sim > best_sim:
                best_sim, best_qh = sim, qh
        if best_qh is None:
            return -1.0, None
        raw = self.r.hget(f"{self.ns}:sem:payload", best_qh)
        if not raw:
            return best_sim, None
        try:
            return best_sim, json.loads(raw)
        except (ValueError, TypeError):
            return best_sim, None

    def _ensure_sem_loaded(self):
        """首次访问语义缓存时从 Redis 拉一次全量索引做进程内镜像（小规模 O(n) 余弦）。"""
        with self._lock:
            if self._sem_loaded:
                return
            self._sem_loaded = True  # 标记先置位，避免异常后反复重试
        if not self.redis_on:
            return
        try:
            embeds = self.r.hgetall(f"{self.ns}:sem:embed")
            texts = self.r.hgetall(f"{self.ns}:sem:text")
            idx = []
            for qh, ej in embeds.items():
                try:
                    vec = np.asarray(json.loads(ej), dtype=np.float32)
                    idx.append((qh, texts.get(qh, ""), vec))
                except (ValueError, TypeError):
                    continue
            with self._lock:
                self._sem_index = idx
            logger.info(f"语义索引加载完成: {len(idx)} 条历史 query")
        except Exception as e:
            logger.warning(f"加载语义索引失败: {e}")

    def _embed_cached(self, query: str) -> Optional[np.ndarray]:
        """带 embedding 缓存的向量化：相同 query 不重算（roadmap L3 embedding 缓存）。"""
        qh = self._sha(self._normalize(query))
        if self.redis_on:
            raw = self.r.get(self._key("emb", qh))
            if raw:
                try:
                    return np.asarray(json.loads(raw), dtype=np.float32)
                except (ValueError, TypeError):
                    pass
        try:
            vec = np.asarray(self.embed(query), dtype=np.float32)
        except Exception as e:
            logger.warning(f"embedding 计算失败，语义缓存跳过: {e}")
            return None
        if self.redis_on:
            self.r.setex(self._key("emb", qh), self.cfg.cache_ttl_embedding,
                         json.dumps(vec.astype(np.float32).tolist()))
        return vec

    # ---------- 路由决策缓存：L4 ----------
    def get_route(self, query: str) -> Optional[dict]:
        """命中返回缓存的路由分析 dict（含最终 recommended_strategy）；否则 None。"""
        if not self.redis_on:
            return None
        qh = self._sha(self._normalize(query))
        raw = self.r.get(self._key("route", qh))
        if not raw:
            with self._lock:
                self._metrics["route_miss"] += 1
            return None
        try:
            with self._lock:
                self._metrics["route_hit"] += 1
            return json.loads(raw)
        except (ValueError, TypeError) as e:
            logger.warning(f"路由缓存反序列化失败，丢弃: {e}")
            return None

    def set_route(self, query: str, analysis_dict: dict):
        if not self.redis_on:
            return
        qh = self._sha(self._normalize(query))
        self.r.setex(self._key("route", qh), self.cfg.cache_ttl_route,
                     json.dumps(analysis_dict, ensure_ascii=False))

    # ---------- 失效 & 埋点 ----------
    def invalidate_all(self) -> int:
        """知识库 rebuild 时调用：清空 L1、语义镜像、Redis 命名空间。返回删除键数。"""
        with self._lock:
            self._l1.clear()
            self._sem_index.clear()
            self._sem_loaded = False
        if not self.redis_on:
            return 0
        n = self.r.delete_keys(f"{self.ns}:*")
        logger.info(f"缓存命名空间 {self.ns}: 已清空 {n} 个键")
        return n

    def stats(self) -> dict:
        with self._lock:
            m = dict(self._metrics)
            l1_size = len(self._l1)
            sem_size = len(self._sem_index)
        answer_total = (m.get("answer_hit_l1", 0) + m.get("answer_hit_l2", 0)
                        + m.get("answer_hit_sem", 0) + m.get("answer_miss", 0))
        answer_hit = answer_total - m.get("answer_miss", 0)
        return {
            "enabled": self.enabled,
            "redis_available": self.r.available,
            "answer_hit_rate": round(answer_hit / answer_total, 3) if answer_total else 0.0,
            "l1_size": l1_size,
            "semantic_index_size": sem_size,
            "semantic_threshold": self.cfg.cache_semantic_threshold,
            "metrics": m,
        }
